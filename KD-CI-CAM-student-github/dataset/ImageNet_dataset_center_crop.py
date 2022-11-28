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

class ImageNet_Dataset_Center_Crop_Test(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, input_size=256):
        self.dir = "../../CI-CAM-final/dataset/ILSVRC2016"
        self.name_list = []
        self.label_list = []
        self.bbox_list = []
        self.img_size = []
        self.resize_bbox_crop = ResizedBBoxCrop((input_size, input_size))
        self.center_bbox_crop = CenterBBoxCrop((cfg.crop_size))
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

        _imagenet_pca = {}
        _imagenet_pca["eigval"] = torch.from_numpy(np.asarray([0.2175, 0.0188, 0.0045]))
        _imagenet_pca["eigvec"] = torch.from_numpy(np.asarray([
                                        [-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]
                                    ]))

        self.list_path = os.path.join(self.dir, "datalist", "val.txt")
        self.func_transforms = transforms.Compose([transforms.Resize(input_size),
                                                   transforms.CenterCrop(cfg.crop_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean_vals, std_vals)
                                                   ])
        self.read_labeled_name_list()
        self.read_bbox_list()
        self.read_imagesize_list()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        image = Image.open(img_name).convert('RGB')
        # image = Image.open(img_name)  # 因为有些图片不是三通道的，所以需要转为RGB三通道
        image = self.func_transforms(image).float()

        return image, torch.tensor(self.label_list[idx]), torch.tensor(self.img_size[idx])

    def read_bbox_list(self):
        with open(os.path.join(self.dir, "datalist", "bounding_boxes.txt"), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].split(' ')
                bbox_number = len(line[1:]) // 4
                bboxes = []
                raw_img = Image.open(self.name_list[i]).convert('RGB')
                for j in range(bbox_number):
                    gt_bbox = [float(line[j*4 + 1]),
                               float(line[j*4 + 2]),
                               float(line[j*4 + 1]) + float(line[j*4 + 3]),
                               float(line[j*4 + 2]) + float(line[j*4 + 4])
                              ]
                    raw_img_i, gt_bbox_i = self.resize_bbox_crop(raw_img, gt_bbox)
                    _, gt_bbox_i = self.center_bbox_crop(raw_img_i, gt_bbox_i)
                    w, h = cfg.crop_size, cfg.crop_size
                    gt_bbox_i[0] = gt_bbox_i[0] * w
                    gt_bbox_i[2] = gt_bbox_i[2] * w
                    gt_bbox_i[1] = gt_bbox_i[1] * h
                    gt_bbox_i[3] = gt_bbox_i[3] * h
                    bboxes.append(gt_bbox_i)  # [xmin, ymin, xmax, ymax]
                self.bbox_list.append(bboxes)  # [[xmin, ymin, xmax, ymax]]

    def read_imagesize_list(self):
        with open(os.path.join(self.dir, "datalist", "sizes.txt"), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].split(' ')
                self.img_size.append([float(line[2]), float(line[1])])  # [height, width]

    def read_labeled_name_list(self):
        basedir = os.path.join(self.dir, "images", "val")
        with open(self.list_path, 'r') as f:
            for line in f:
                ls = line.strip("\n").split(' ')
                self.name_list.append(os.path.join(basedir, ls[-2]))
                self.label_list.append(int(ls[-1]))


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))