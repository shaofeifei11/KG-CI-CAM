import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import config as cfg
import numpy as np
import random
from RandAugment import RandAugment

class CUB_Dataset_KD_RandomReizeCrop_2(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, danet=False, zero=True, cls_min_scale=0.2, loc_min_scale=0.1, distillation="random"):
        self.dir = "../../CI-CAM-final/dataset/CUB_200_2011"
        self.image_list = []
        self.label_list = []
        self.zero = zero
        self.distillation = distillation
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]

        _imagenet_pca = {}
        _imagenet_pca["eigval"] = torch.from_numpy(np.asarray([0.2175, 0.0188, 0.0045]))
        _imagenet_pca["eigvec"] = torch.from_numpy(np.asarray([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]
        ]))

        self.list_path = os.path.join(self.dir, "datalist", "train_list.txt")
        if danet:
            self.func_transforms_stu = transforms.Compose(
                [transforms.RandomResizedCrop(cfg.crop_size, scale=(cls_min_scale, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.4, 0.4, 0.4),
                 transforms.ToTensor(),
                 Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                 transforms.Normalize(mean_vals, std_vals)
                 ])
            self.func_transforms_cls = transforms.Compose(
                    [transforms.RandomResizedCrop(cfg.crop_size, scale=(cls_min_scale, 1.0)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ColorJitter(0.4, 0.4, 0.4),
                     transforms.ToTensor(),
                     Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                     transforms.Normalize(mean_vals, std_vals)
                     ])
            self.func_transforms_loc = transforms.Compose(
                [transforms.RandomResizedCrop(cfg.crop_size, scale=(loc_min_scale, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(0.4, 0.4, 0.4),
                 transforms.ToTensor(),
                 Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                 transforms.Normalize(mean_vals, std_vals)
                 ])
        else:
            self.func_transforms_stu = transforms.Compose(
                [transforms.RandomResizedCrop(cfg.crop_size, scale=(cls_min_scale, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean_vals, std_vals)
                 ])
            self.func_transforms_cls = transforms.Compose(
                    [transforms.RandomResizedCrop(cfg.crop_size, scale=(cls_min_scale, 1.0)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean_vals, std_vals)
                     ])
            self.func_transforms_loc = transforms.Compose(
                [transforms.RandomResizedCrop(cfg.crop_size, scale=(loc_min_scale, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean_vals, std_vals)
                 ])

        self.read_labeled_image_list()
        if cfg.args.randAug == 1 or cfg.args.randAug == 4 or cfg.args.randAug == 5 or cfg.args.randAug == 7:
            self.func_transforms_stu.transforms.insert(0, RandAugment(3, 10))
        if cfg.args.randAug == 2 or cfg.args.randAug == 4 or cfg.args.randAug == 6 or cfg.args.randAug == 7:
            self.func_transforms_cls.transforms.insert(0, RandAugment(3, 10))
        if cfg.args.randAug == 3 or cfg.args.randAug == 5 or cfg.args.randAug == 6 or cfg.args.randAug == 7:
            self.func_transforms_loc.transforms.insert(0, RandAugment(3, 10))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert('RGB')
        # image = Image.open(img_name)  # 因为有些图片不是三通道的，所以需要转为RGB三通道
        # flag = random.randint(0, 2)
        if self.zero:
            if self.distillation =="random":
                flag = random.randint(0, 2)
            elif self.distillation == "none":
                flag = 0
            elif self.distillation =="cls":
                flag = random.randint(0, 1)
            elif self.distillation =="loc":
                flag = random.randint(0, 1)
                if flag == 1:
                    flag = 2
            # elif self.distillation =="all":
            #     flag = random.randint(0, 1)
            #     if flag == 1:
            #         flag = 3
        else:
            if self.distillation == "random":
                flag = random.randint(1, 2)
            elif self.distillation =="none":
                flag = 0
            elif self.distillation =="cls":
                flag = 1
            elif self.distillation =="loc":
                flag = 2
            # elif self.distillation =="all":
            #     flag = 3

        if flag == 0:
            image_tensor = self.func_transforms_stu(image).float()
        elif flag == 1:
            image_tensor = self.func_transforms_cls(image).float()
        elif flag == 2:
            image_tensor = self.func_transforms_loc(image).float()

        return image_tensor, torch.tensor(self.label_list[idx]), torch.tensor(flag)

    def read_labeled_image_list(self):
        with open(self.list_path, 'r') as f:
            for line in f:
                image, label = line.strip("\n").split(';')
                self.image_list.append(os.path.join(self.dir, "images", image.strip()))
                self.label_list.append(int(label.strip()))


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