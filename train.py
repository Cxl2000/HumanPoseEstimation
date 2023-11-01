import json
import os
import datetime

import torch
from torch.utils import data
import numpy as np

from model.HRNet import HighResolutionNet
import utils.transforms as transforms
from utils.my_dataset_coco import CocoKeypoint
from loss.loss import KpLoss


coco_path = "../dataset/COCO2017"
train_img_path = "../dataset/COCO2017/train2017"
train_ann_path = "../dataset/COCO2017/annotations_trainval2017/person_keypoints_train2017.json"
person_keypoints_path = "./person_keypoints.json"

num_joints = 17
input_size = (256, 192)
heatmap_hw = (input_size[0] / 4, input_size[1] / 4)


batch_size = 1
learn_rate = 0.001
learn_rate_steps = [170, 200]
learn_rate_gamma = 0.1
weight_decay = 0.0001
amp = True
epochs = 300

if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))
    model = HighResolutionNet().to(device)

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 人体关键点索引、权重等信息
    with open(person_keypoints_path, "r") as f:
        person_kps_info = json.load(f)

    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((num_joints))

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.VisibleFilter(),
            transforms.HalfBody(upper_body_ids = person_kps_info["upper_body_ids"], lower_body_ids = person_kps_info["lower_body_ids"]),
            transforms.CropKpsAndAffineTransform(fixed_size=input_size),
            transforms.RandomHorizontalFlip(matched_parts = person_kps_info["flip_pairs"]),
            transforms.KeypointToHeatMap(num_joints = num_joints, heatmap_hw=heatmap_hw, gaussian_sigma=2),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.VisibleFilter(),
            transforms.CropKpsAndAffineTransform(fixed_size=input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    train_dataset = CocoKeypoint(coco_path, "train", transforms=data_transform["train"], fixed_size=input_size)
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)

    val_dataset = CocoKeypoint(coco_path, "val", transforms=data_transform["val"], fixed_size=input_size)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=nw,
                                      collate_fn=val_dataset.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=learn_rate,
                                  weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler() if amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=learn_rate_steps, gamma=learn_rate_gamma)


    for epoch in range(0, epochs):
        for i, [images, targets] in enumerate(train_data_loader):
            print(i, len(targets[0]["keypoints"]))














