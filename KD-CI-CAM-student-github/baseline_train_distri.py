import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

import config as cfg
from dataset.CUB_dataset_KD_randomresizecrop_2 import CUB_Dataset_KD_RandomReizeCrop_2
from dataset.ImageNet_dataset_KD_randomresizecrop import ImageNet_Dataset_KD_RandomReizeCrop
from baseline.baseline_cub_vgg16 import CUB_VGG16_Baseline
from baseline.baseline_cub_inceptionV3 import CUB_InceptionV3_NonLocal_Baseline
from baseline.baseline_imagenet_vgg16 import ImageNet_VGG16_Baseline
from option import args_parser
from baseline_test_center_crop import inference
from utils import net_load, net_save_classifier

def poly_lr_scheduler_baseline(optimizer, init_lr, iter, max_iter=100, power=0.9, dataset="imagenet", backbone_rate=0.1, decay_rate=0.5, decay_epoch=2, inception_reduce=0.1, max_num=100):
    lr = init_lr*(1 - iter/max_iter)**power
    if cfg.args.backbone == "inceptionV3" and iter >= 85:
        lr = lr * inception_reduce
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if dataset == "imagenet":
        optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] * backbone_rate
        shang = iter // decay_epoch
        new_rate = 1
        for i in range(shang):
            new_rate = new_rate * decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * new_rate
    else:
        optimizer.param_groups[-1]['lr'] = optimizer.param_groups[-1]['lr'] * backbone_rate
        shang = iter // decay_epoch
        lg = min(shang, max_num)
        new_rate = 1
        for i in range(lg):
            new_rate = new_rate * decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * new_rate

if __name__ == '__main__':
    args = args_parser()

    epoch = args.epoch
    lr = args.lr

    batch_size = args.batch_size
    gpus = args.gpu.replace("_", ",")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    print("gpus: ", gpus)

    if args.time == None:
        tim = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    else:
        tim = args.time
    logger = None
    net_dir = os.path.join("save_model", args.dir, tim)
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

    if args.dataset == "cub":
        train_data = CUB_Dataset_KD_RandomReizeCrop_2(danet=args.danet, zero=args.kd_fe_zero,
                                                      cls_min_scale=args.cls_min_scale, loc_min_scale=args.loc_min_scale,
                                                      distillation=args.distillation)
        if args.backbone == "vgg16_baseline":
            student_net = CUB_VGG16_Baseline(args)
        elif args.backbone == "inceptionV3_baseline":
            student_net = CUB_InceptionV3_NonLocal_Baseline(args)
    else:
        train_data = ImageNet_Dataset_KD_RandomReizeCrop(danet=args.danet, zero=args.kd_fe_zero,
                                                         cls_min_scale=args.cls_min_scale, cls_max_scale=args.cls_max_scale,
                                                         loc_min_scale=args.loc_min_scale, loc_max_scale=args.loc_max_scale,
                                                         distillation=args.distillation)
        if args.backbone == "vgg16_baseline":
            student_net = ImageNet_VGG16_Baseline(args)

    train_data_length = len(train_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)

    cls_criterion = nn.CrossEntropyLoss().to(cfg.device)
    begin_epoch = 0

    # load student
    model_path = args.model_path
    if model_path != None:
        print("\n\n########################################################################")
        print("model_path: ", model_path)
        print("model_args: ", args.dir)
        student_net = net_load(student_net, model_path)
        begin_epoch = int(model_path.split("_")[-1].split(".")[0])

    print("run here! ", len(gpus))
    if len(gpus) > 1:
        print("DataParallel! ")
        student_net = torch.nn.DataParallel(student_net).to(cfg.device)
        student_net_module = student_net.module
    else:
        print("one machine! ")
        student_net.to(cfg.device)
        student_net_module = student_net

    optimizer = torch.optim.Adam([{'params': student_net_module.up_classifier.parameters()},
                                  {'params': student_net_module.backbone.parameters()}], lr=lr)

    for e in range(begin_epoch, epoch):
        poly_lr_scheduler_baseline(optimizer, lr, e, max_iter=epoch, dataset=args.dataset,
                                 backbone_rate=args.backbone_rate, decay_rate=cfg.decay_rate,
                                 decay_epoch=args.decay_epoch, inception_reduce=args.inception_reduce,
                                 max_num=args.max_num)

        student_net.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_bbox_loss = 0

        for i, dat in enumerate(train_loader):
            images, labels, flags = dat
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            labels = labels.long()
            flag_numpy = flags.detach().cpu().numpy()
            optimizer.zero_grad()

            cam_up, out_up, pred_sort_up, pred_ids_up = student_net(images)

            ############ down ###########
            cls_loss_down = cls_criterion(out_up, labels)
            loss = cls_loss_down
            epoch_acc += float((pred_ids_up[:, 0].reshape(labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())

            print("epoch: ", e + 1, " | batch: ", i, "/", len(train_loader), " | loss: ", loss.item())

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        if logger == None:
            logger = SummaryWriter(os.path.join("tensorboard", args.dir, tim))

        if args.dataset == "cub":
            if (e >= 49 and (e+1) % 5 == 0) or epoch - (e+1) < 5:
                net_save_classifier(student_net_module, os.path.join(net_dir, 'net_train_{}_{}_{}.pth'.format(args.backbone, args.dataset, e + 1)))
            if (e+1) % 5 == 0:
                train_avg_epoch_acc = epoch_acc / train_data_length
                train_avg_epoch_cls_loss = epoch_loss / train_data_length
                train_avg_epoch_box_loss = epoch_bbox_loss / train_data_length
                print("Train Epoch: ", e+1, " | Train Avg Acc: ", train_avg_epoch_acc, " | Train Avg Cls Loss: ",
                      train_avg_epoch_cls_loss, " | Train Avg Box Loss: ", train_avg_epoch_box_loss)
                logger.add_scalar("Train Avg Acc", train_avg_epoch_acc, e + 1)
                logger.add_scalar("Train Avg Cls Loss", train_avg_epoch_cls_loss, e + 1)
                logger.add_scalar("Train Avg Box Loss", train_avg_epoch_box_loss, e + 1)
                c1, l1, c5, l5, gt_l = inference(student_net_module, args)
                logger.add_scalar("classification top 1 accuracy: ", c1, e + 1)
                logger.add_scalar("localization top 1 accuracy: ", l1, e + 1)
                logger.add_scalar("classification top 5 accuracy: ", c5, e + 1)
                logger.add_scalar("localization top 5 accuracy: ", l5, e + 1)
                logger.add_scalar("gt localization accuracy: ", gt_l, e + 1)

        else:
            net_save_classifier(student_net_module, os.path.join(net_dir, 'net_train_{}_{}_{}.pth'.format(args.backbone, args.dataset, e + 1)))
            if (e+1) % 1 == 0:
                train_avg_epoch_acc = epoch_acc / train_data_length
                train_avg_epoch_cls_loss = epoch_loss / train_data_length
                train_avg_epoch_box_loss = epoch_bbox_loss / train_data_length
                print("Train Epoch: ", e+1, " | Train Avg Acc: ", train_avg_epoch_acc, " | Train Avg Cls Loss: ",
                      train_avg_epoch_cls_loss, " | Train Avg Box Loss: ", train_avg_epoch_box_loss)
                logger.add_scalar("Train Avg Acc", train_avg_epoch_acc, e + 1)
                logger.add_scalar("Train Avg Cls Loss", train_avg_epoch_cls_loss, e + 1)
                logger.add_scalar("Train Avg Box Loss", train_avg_epoch_box_loss, e + 1)
                c1, l1, c5, l5, gt_l= inference(student_net_module, args)
                logger.add_scalar("classification top 1 accuracy: ", c1, e + 1)
                logger.add_scalar("localization top 1 accuracy: ", l1, e + 1)
                logger.add_scalar("classification top 5 accuracy: ", c5, e + 1)
                logger.add_scalar("localization top 5 accuracy: ", l5, e + 1)
                logger.add_scalar("gt localization accuracy: ", gt_l, e + 1)

    logger.close()
