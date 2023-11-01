import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F



class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target



class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class VisibleFilter(object):
    """
        根据关键点可见度过滤关键点
        返回的target关键点个数可能小于17，但增加了关键点id以防混淆
        标签数据预处理必须加上此功能
    """
    def __init__(self,threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, image, target):
        keypoints = []
        keypointsId = []
        vis = []
        for i, v in enumerate(target["visible"]):
            if v > self.threshold:
                keypoints.append(target["keypoints"][i])
                keypointsId.append(i)
                vis.append(target["visible"][i])
        target["keypoints"] = keypoints
        target["keypointsId"] = keypointsId
        target["visible"] = vis
        
        return image, target


class HalfBody(object):
    def __init__(self, p: float = 0.3, upper_body_ids=None, lower_body_ids=None):
        assert upper_body_ids is not None
        assert lower_body_ids is not None
        self.p = p
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

    def __call__(self, image, target):
        # 根据概率确定是否选择半身
        if random.random() < self.p:
            kps = target["keypoints"]
            kpsId = target["keypointsId"]
            kpsVis = target["visible"]
            # 50%的概率选择上或下半身
            if random.random() < 0.5:
                selectId = self.upper_body_ids
            else:
                selectId = self.lower_body_ids

            # 对keypoints进行归类
            selected_kps = []
            selected_kpsId = []
            selected_kpsVis = []
            for i, keypointsId in enumerate(target["keypointsId"]):
                if keypointsId in selectId:
                    selected_kps.append(kps[i])
                    selected_kpsId.append(kpsId[i])
                    selected_kpsVis.append(kpsVis[i])

            # 如果选择半身后发现点数太少则选择全身
            if len(selected_kpsId) >= 2:
                target["keypoints"] = selected_kps
                target["keypointsId"] = selected_kpsId
                target["visible"] = selected_kpsVis

        return image, target

class CropKpsAndAffineTransform(object):
    """ 
        通过关键点截取图片，并对图片进行随机缩放旋转
    """
    def __init__(self,
                 p: float = 0.5,
                 scale: Tuple[float, float] = (0.8, 1.2),
                 rotation: Tuple[int, int] = (-30, 30),
                 fixed_size: Tuple[int, int] = (256, 192)):
        self.p = p  # 缩放旋转的概率
        self.scale = random.uniform(*scale)         # 随机缩放系数
        self.rotation = random.uniform(*rotation)   # 随机旋转度数
        self.fixed_size = fixed_size

    def __call__(self, img, target):
        """
            根据图片关键点信息裁剪图片
            防止关键点在图片边界，应扩大裁剪区域(扩大倍数为固定常数)
            扩大倍数再乘以随机比例缩放倍数可以达到随机缩放效果
            裁剪完成后应缩放为fixed_size大小(保持高宽比不变)
        """
        
        # 确定裁剪区域
        xmin, ymin = np.min(target["keypoints"], axis = 0).astype(np.int16)
        xmax, ymax = np.max(target["keypoints"], axis = 0).astype(np.int16)
        # 只有一个关键点，或关键点在一条线上，防止裁剪区域面积为0
        if xmin == xmax:
            xmin -= self.fixed_size[1] // 2
            xmax += self.fixed_size[1] // 2
        if ymin == ymax:
            ymin -= self.fixed_size[0] // 2
            ymax += self.fixed_size[0] // 2

        # 增加裁剪区域  防止关键点在边界
        expand_const = 1.5
        w = (xmax - xmin) * expand_const
        h = (ymax - ymin) * expand_const

        # scale 缩放  根据概率确定是否缩放
        if random.random() < self.p:
            w, h = w * self.scale, h * self.scale

        # 增加hw控制图片比例  不能超出图片边界
        hw_ratio = self.fixed_size[0]/self.fixed_size[1]
        if h/w < hw_ratio:
            h = w * hw_ratio
        else:
            w = h / hw_ratio
        # 防止超边界裁剪
        bbox_xmin = max(int((xmin + xmax) / 2 - w*0.5), 0)
        bbox_ymin = max(int((ymin + ymax) / 2 - h*0.5), 0)
        bbox_xmax = min(int((xmin + xmax) / 2 + w*0.5), target["image_width"])
        bbox_ymax = min(int((ymin + ymax) / 2 + h*0.5), target["image_height"])
        # 裁剪图像  image: H W C
        image = img[bbox_ymin : bbox_ymax, bbox_xmin : bbox_xmax]
        # 若由于裁剪坐标超出边界导致比例不正确，则填充图片
        image_h, image_w= image.shape[0], image.shape[1]
        image_hw_ratio = image_h / image_w
        if image_hw_ratio < hw_ratio:
            border_size = int((image_w * hw_ratio - image_h) / 2)
            image = cv2.copyMakeBorder(image, border_size, border_size, 0, 0, cv2.BORDER_CONSTANT,value=(255,255,255))
            bbox_ymin -= border_size
            bbox_ymax += border_size
        else:
            border_size = int((image_h / hw_ratio - image_w) / 2)
            image = cv2.copyMakeBorder(image, 0, 0, border_size, border_size, cv2.BORDER_CONSTANT,value=(255,255,255))
            bbox_xmin -= border_size
            bbox_xmax += border_size
        # 计算裁剪填充后图像与对应关键点新坐标
        target["keypoints"] -= np.array([bbox_xmin, bbox_ymin])
        target["image_width"] = bbox_xmax - bbox_xmin
        target["image_height"] = bbox_ymax - bbox_ymin

        img_w = target["image_width"]
        img_h = target["image_height"]
        p_center = np.array([img_w/2, img_h/2])
        p2_src = np.array([0, img_h/2])    # 左中
        p3_src = np.array([img_w/2, 0])    # 上中

        dst_center = np.array([self.fixed_size[1] / 2, self.fixed_size[0] / 2])
        dst_p2 = np.array([0, self.fixed_size[0] / 2])  # 左中
        dst_p3 = np.array([self.fixed_size[1] / 2, 0])  # 上中
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

        # rotate 顺时针旋转  根据概率确定是否旋转
        if random.random() < self.p:
            angle = self.rotation  # 角度制
            angle = angle / 180 * math.pi  # 弧度制
            p2_rotate = p_center + np.array([-(p_center[0] - p2_src[0]) * math.cos(angle), (p_center[0] - p2_src[0]) * math.sin(angle)])
            p3_rotate = p_center + np.array([-(p_center[1] - p3_src[1]) * math.sin(angle), -(p_center[1] - p3_src[1]) * math.cos(angle)])
            pts_rotate = np.stack([p_center, p2_rotate, p3_rotate]).astype(np.float32)
            trans = cv2.getAffineTransform(pts_rotate, dst)  # 计算正向仿射变换矩阵
        else:
            pts_src = np.stack([p_center, p2_src, p3_src]).astype(np.float32)
            trans = cv2.getAffineTransform(pts_src, dst)  # 计算正向仿射变换矩阵

        resize_img = cv2.warpAffine(image,
                                    trans,
                                    tuple(self.fixed_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR)

        # 根据变换矩阵重新计算关键点位置
        kps = target["keypoints"]
        ones = np.ones((kps.shape[0], 1), dtype=float)
        kp_m = np.concatenate([kps, ones], axis=1)
        target["keypoints"] = np.dot(kp_m, trans.T)

        # 过滤超出图片边界的关键点
        resize_img_xmax = resize_img.shape[1]
        resize_img_ymax = resize_img.shape[0]
        kps = []
        kpsId = []
        kpsVis = []
        for i, keypoints in  enumerate(target["keypoints"]):
            if keypoints[0] < 0 or keypoints[0] > resize_img_xmax or keypoints[1] < 0 or keypoints[1] > resize_img_ymax:
                continue
            else:
                kps.append(keypoints)
                kpsId.append(target["keypointsId"][i])
                kpsVis.append(target["visible"][i])

        # 若变换后关键点全部丢失，则取消变换，仅仅缩放为指定大小
        if len(kps) == 0:
            pts_src = np.stack([p_center, p2_src, p3_src]).astype(np.float32)
            trans = cv2.getAffineTransform(pts_src, dst)  # 计算正向仿射变换矩阵
            resize_img = cv2.warpAffine(image,
                                    trans,
                                    tuple(self.fixed_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR)
            target["keypoints"] = np.dot(kp_m, trans.T)
            return resize_img, target

        target["keypoints"] = kps
        target["keypointsId"] = kpsId
        target["visible"] = kpsVis

        return resize_img, target

class RandomHorizontalFlip(object):
    """随机对输入图片进行水平翻转，注意该方法必须接在 AffineTransform 后"""
    def __init__(self, p: float = 0.5, matched_parts: list = None):
        assert matched_parts is not None
        self.p = p
        self.matched_parts = matched_parts

    def __call__(self, image, target):
        if random.random() < self.p:
            # [h, w, c]
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            keypoints = np.array(target["keypoints"])
            keypointsId = target["keypointsId"]

            width = image.shape[1]

            # Flip horizontal
            keypoints[:, 0] = width - keypoints[:, 0] - 1

            # Change left-right parts
            for parts in self.matched_parts:
                for kpsId in keypointsId:
                    if kpsId == parts[0]:
                        kpsId = parts[1]
                    elif kpsId == parts[1]:
                        kpsId = parts[0]

            target["keypoints"] = keypoints

        return image, target

class KeypointToHeatMap(object):
    def __init__(self,
                 num_joints: int = 17,
                 heatmap_hw: Tuple[int, int] = (256 // 4, 192 // 4),
                 gaussian_sigma: int = 2):
        self.num_joints = num_joints
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3

        # generate gaussian kernel(not normalized)
        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        # print(kernel)

        self.kernel = kernel

    def __call__(self, image, target):
        kps = np.array(target["keypoints"])
        kpsId = target["keypointsId"]
        num_kps = self.num_joints

        heatmap = np.zeros((int(num_kps), int(self.heatmap_hw[0]), int(self.heatmap_hw[1])))
        heatmap_kps = (kps / 4 + 0.5).astype(np.int16)  # 取整
        for kp_id in kpsId:
            i = 0
            x, y = heatmap_kps[i]   # 热图中关键点坐标
            i += 1
            # 热图中高斯核矩形坐标
            xmin = x - self.kernel_radius
            ymin = y - self.kernel_radius
            xmax = x + self.kernel_radius + 1
            ymax = y + self.kernel_radius + 1
            gs_xmin = gs_ymin = 0
            gs_xmax = gs_ymax = 2 * self.kernel_radius + 1
            # 防止填充高斯核出界(关键点在比较边界位置)
            if xmin < 0:
                gs_xmin -= xmin
                xmin = 0
            if ymin < 0:
                gs_ymin -= ymin
                ymin = 0
            if xmax > self.heatmap_hw[1]:
                gs_xmax -= (xmax - self.heatmap_hw[1])
                xmax = self.heatmap_hw[1]
            if ymax > self.heatmap_hw[0]:
                gs_ymax -= (ymax - self.heatmap_hw[0])
                ymax = self.heatmap_hw[0]

            # 将高斯核有效区域复制到heatmap对应区域
            heatmap[int(kp_id), int(ymin):int(ymax), int(xmin):int(xmax)] = self.kernel[int(gs_ymin):int(gs_ymax), int(gs_xmin):int(gs_xmax)]

        # C H W
        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)

        return image, target






























