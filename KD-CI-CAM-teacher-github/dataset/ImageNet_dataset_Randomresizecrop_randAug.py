import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import config as cfg
import numpy as np
from RandAugment import RandAugment

class ImageNet_Dataset_Randomresizecrop_RandAug(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, input_size=256, train=True, crop=False, danet=False, cam_img=False, min_scale=0.2, max_scale=1.0):
        self.dir = "../../CI-CAM-final/dataset/ILSVRC2016"
        self.train = train
        self.name_list = []
        self.label_list = []
        self.bbox_list = []
        self.img_size = []
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

        _imagenet_pca = {}
        _imagenet_pca["eigval"] = torch.from_numpy(np.asarray([0.2175, 0.0188, 0.0045]))
        _imagenet_pca["eigvec"] = torch.from_numpy(np.asarray([
                                        [-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]
                                    ]))

        if cam_img:
            self.list_path = os.path.join(self.dir, "datalist", "train.txt")
            self.func_transforms = transforms.Compose([transforms.Resize((cfg.crop_size, cfg.crop_size)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean_vals, std_vals)
                                                       ])
        else:
            if self.train:
                self.list_path = os.path.join(self.dir, "datalist", "train.txt")
                if danet:
                    self.func_transforms = transforms.Compose(
                        [transforms.RandomResizedCrop(cfg.crop_size, scale=(min_scale, max_scale)),  # 04-26 将resize+randCrop 改为了 randomResizeCrop，且改了scale
                         transforms.RandomHorizontalFlip(),
                         # transforms.RandomVerticalFlip(),  # add at 2022-04-19 106
                         transforms.ToTensor(),
                         transforms.ColorJitter(0.4, 0.4, 0.4),
                         Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                         transforms.Normalize(mean_vals, std_vals)
                         ])
                else:
                    self.func_transforms = transforms.Compose(
                        [transforms.RandomResizedCrop(cfg.crop_size, scale=(min_scale, max_scale)),  # 04-26 将resize+randCrop 改为了 randomResizeCrop，且改了scale
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean_vals, std_vals)
                         ])
            else:
                self.list_path = os.path.join(self.dir, "datalist", "val.txt")
                if crop:
                    self.func_transforms = transforms.Compose([transforms.Resize(input_size),
                                                               transforms.CenterCrop(cfg.crop_size),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean_vals, std_vals)
                                                               ])
                else:
                    self.func_transforms = transforms.Compose([transforms.Resize((cfg.crop_size, cfg.crop_size)),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean_vals, std_vals)
                                                               ])
                self.read_bbox_list()
                self.read_imagesize_list()

        self.func_transforms.transforms.insert(0, RandAugment(3, 10))
        self.read_labeled_name_list()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_name = self.name_list[idx]
        image = Image.open(img_name).convert('RGB')
        # image = Image.open(img_name)  # 因为有些图片不是三通道的，所以需要转为RGB三通道
        image = self.func_transforms(image).float()
        if self.train:
            return image, torch.tensor(self.label_list[idx])
        else:
            return image, torch.tensor(self.label_list[idx]), torch.tensor(self.img_size[idx])

    def read_bbox_list(self):
        with open(os.path.join(self.dir, "datalist", "bounding_boxes.txt"), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].split(' ')
                bbox_number = len(line[1:]) // 4
                bboxes = []
                for j in range(bbox_number):
                    bboxes.append([float(line[j*4 + 1]),
                                   float(line[j*4 + 2]),
                                   float(line[j*4 + 1]) + float(line[j*4 + 3]),
                                   float(line[j*4 + 2]) + float(line[j*4 + 4])
                                  ])  # [xmin, ymin, xmax, ymax]
                self.bbox_list.append(bboxes)  # [[xmin, ymin, xmax, ymax]]

    def read_imagesize_list(self):
        with open(os.path.join(self.dir, "datalist", "sizes.txt"), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i].split(' ')
                self.img_size.append([float(line[2]), float(line[1])])  # [height, width]

    def read_labeled_name_list(self):
        if self.train:
            basedir = os.path.join(self.dir, "images", "train")
        else:
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