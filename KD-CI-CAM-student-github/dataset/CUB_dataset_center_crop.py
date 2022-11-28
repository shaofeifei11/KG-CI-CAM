import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import config as cfg
import numpy as np
import copy
from torchvision.transforms import functional as F
import numbers

class ResizedBBoxCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        #resize to 256
        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                img = copy.deepcopy(img)
                ow, oh = w, h
            if w < h:
                ow = size
                oh = int(size*h/w)
            else:
                oh = size
                ow = int(size*w/h)
        else:
            ow, oh = size[::-1]
            w, h = img.size


        intersec = copy.deepcopy(bbox)
        ratew = ow / w
        rateh = oh / h
        intersec[0] = bbox[0]*ratew
        intersec[2] = bbox[2]*ratew
        intersec[1] = bbox[1]*rateh
        intersec[3] = bbox[3]*rateh

        #intersec = normalize_intersec(i, j, h, w, intersec)
        return (oh, ow), intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        size, crop_bbox = self.get_params(img, bbox, self.size)
        return F.resize(img, self.size, self.interpolation), crop_bbox

class CenterBBoxCrop(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size

        self.interpolation = interpolation

    @staticmethod
    def get_params(img, bbox, size):
        #center crop
        if isinstance(size, numbers.Number):
            output_size = (int(size), int(size))

        w, h = img.size
        th, tw = output_size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        intersec = compute_intersec(i, j, th, tw, bbox)
        intersec = normalize_intersec(i, j, th, tw, intersec)

        #intersec = normalize_intersec(i, j, h, w, intersec)
        return i, j, th, tw, intersec

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, th, tw, crop_bbox = self.get_params(img, bbox, self.size)
        return F.center_crop(img, self.size), crop_bbox

def compute_intersec(i, j, h, w, bbox):
    '''
    intersection box between croped box and GT BBox
    '''
    intersec = copy.deepcopy(bbox)

    intersec[0] = max(j, bbox[0])
    intersec[1] = max(i, bbox[1])
    intersec[2] = min(j + w, bbox[2])
    intersec[3] = min(i + h, bbox[3])
    return intersec

def normalize_intersec(i, j, h, w, intersec):
    '''
    return: normalize into [0, 1]
    '''

    intersec[0] = (intersec[0] - j) / w
    intersec[2] = (intersec[2] - j) / w
    intersec[1] = (intersec[1] - i) / h
    intersec[3] = (intersec[3] - i) / h
    return intersec

class CUB_Dataset_Center_Crop_Test(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, input_size):
        self.dir = "../../CI-CAM-final/dataset/CUB_200_2011"
        self.image_list = []
        self.label_list = []
        self.bbox_list = []
        self.img_size = []
        self.input_size = input_size
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

        self.list_path = os.path.join(self.dir, "datalist", "test_list.txt")
        self.func_transforms = transforms.Compose([transforms.Resize((input_size, input_size)),
                                                   transforms.CenterCrop(cfg.crop_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean_vals, std_vals)
                                                   ])

        self.resize_bbox_crop = ResizedBBoxCrop((self.input_size, self.input_size))
        self.center_bbox_crop = CenterBBoxCrop((cfg.crop_size))
        self.read_labeled_image_list()
        self.read_bbox_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        # image = Image.open(img_name)  # 因为有些图片不是三通道的，所以需要转为RGB三通道
        image = self.func_transforms(image).float()

        return image, torch.tensor(self.label_list[idx]), torch.tensor(self.img_size[idx])

    def read_bbox_list(self):
        with open(os.path.join(self.dir, "datalist", "test_bounding_box.txt"), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].split(' ')
                gt_bbox = [float(line[0]), float(line[1]), float(line[2]), float(line[3])]  # [xmin, ymin, xmax, ymax]
                raw_img = Image.open(self.image_list[i]).convert('RGB')
                raw_img_i, gt_bbox_i = self.resize_bbox_crop(raw_img, gt_bbox)
                _, gt_bbox_i = self.center_bbox_crop(raw_img_i, gt_bbox_i)
                w, h = cfg.crop_size, cfg.crop_size
                gt_bbox_i[0] = gt_bbox_i[0] * w
                gt_bbox_i[2] = gt_bbox_i[2] * w
                gt_bbox_i[1] = gt_bbox_i[1] * h
                gt_bbox_i[3] = gt_bbox_i[3] * h
                self.bbox_list.append([gt_bbox_i]) #[[xmin, ymin, xmax, ymax]]
                self.img_size.append([float(line[4]), float(line[5])])  # [height, width]

    def read_labeled_image_list(self):
        with open(self.list_path, 'r') as f:
            for line in f:
                image, label = line.strip("\n").split(';')
                self.image_list.append(os.path.join(self.dir, "images", image.strip()))
                self.label_list.append(int(label.strip()))