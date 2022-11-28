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
from model.cub_inceptionV3_nonlocal import CUB_InceptionV3_NonLocal
from model.cub_vgg16 import CUB_VGG16
from model.imagenet_vgg16 import ImageNet_VGG16
from option import args_parser
from test_student_randomresizecrop import inference
from utils import net_load, net_save, old_create_attention
from utils import poly_lr_scheduler_kd


def _feature_kd_loss_1(source, target, mse):
    feat_kd_loss = mse(source, target)
    return feat_kd_loss

if __name__ == '__main__':
    args = args_parser()

    epoch = args.epoch
    lr = args.lr
    cls_kd_T = args.cls_teacher_T
    loc_kd_T = args.loc_teacher_T
    kd_alpha = args.kd_alpha

    kd_fe_loss = args.kd_fe_loss
    kd_fe_times = args.kd_fe_times

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
        if args.backbone == "vgg16":
            student_net = CUB_VGG16(args)
            cls_teacher = CUB_VGG16(args)
            loc_teacher = CUB_VGG16(args)
        elif args.backbone == "inceptionV3":
            student_net = CUB_InceptionV3_NonLocal(args)
            cls_teacher = CUB_InceptionV3_NonLocal(args)
            loc_teacher = CUB_InceptionV3_NonLocal(args)
    else:
        train_data = ImageNet_Dataset_KD_RandomReizeCrop(danet=args.danet, zero=args.kd_fe_zero,
                                                         cls_min_scale=args.cls_min_scale, cls_max_scale=args.cls_max_scale,
                                                         loc_min_scale=args.loc_min_scale, loc_max_scale=args.loc_max_scale,
                                                         distillation=args.distillation)
        if args.backbone == "vgg16":
            student_net = ImageNet_VGG16(args)
            cls_teacher = ImageNet_VGG16(args)
            loc_teacher = ImageNet_VGG16(args)

    train_data_length = len(train_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)

    cls_criterion = nn.CrossEntropyLoss().to(cfg.device)
    # box_criterion = nn.SmoothL1Loss().to(cfg.device)
    kld_criterion = nn.KLDivLoss(reduction="batchmean").to(cfg.device)
    mse_criterion = nn.MSELoss().to(cfg.device)
    smoothL1_criterion = nn.SmoothL1Loss().to(cfg.device)
    l1_criterion = nn.L1Loss().to(cfg.device)
    begin_epoch = 0

    # load teachers
    cls_teacher = net_load(cls_teacher, cfg.cls_teacher_path)
    loc_teacher = net_load(loc_teacher, cfg.loc_teacher_path)
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
        cls_teacher = torch.nn.DataParallel(cls_teacher).to(cfg.device)
        loc_teacher = torch.nn.DataParallel(loc_teacher).to(cfg.device)
        student_net_module = student_net.module
        cls_net_module = cls_teacher.module
        loc_net_module = loc_teacher.module
    else:
        print("one machine! ")
        student_net.to(cfg.device)
        cls_teacher.to(cfg.device)
        loc_teacher.to(cfg.device)
        student_net_module = student_net
        cls_net_module = cls_teacher
        loc_net_module = loc_teacher

    cls_teacher.train(False)
    loc_teacher.train(False)



    optimizer = torch.optim.Adam([{'params': student_net_module.up_classifier.parameters()},
                                 {'params': student_net_module.mask2attention.parameters()},
                                 {'params': student_net_module.backbone.parameters()}], lr=lr)

    for e in range(begin_epoch, epoch):
        poly_lr_scheduler_kd(optimizer, lr, e, max_iter=epoch, dataset=args.dataset,
                                 backbone_rate=args.backbone_rate, decay_rate=cfg.decay_rate,
                                 decay_epoch=args.decay_epoch, inception_reduce=args.inception_reduce,
                                 max_num=args.max_num)

        student_net.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_bbox_loss = 0

        cls_teacher.eval()
        loc_teacher.eval()

        for i, dat in enumerate(train_loader):
            images, labels, flags = dat
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            labels = labels.long()
            flag_numpy = flags.detach().cpu().numpy()
            optimizer.zero_grad()

            cam_up, out_up, pred_sort_up, pred_ids_up, cam_down, out_down, pred_sort_down, pred_ids_down = student_net(images)

            ############ down ###########
            cls_loss_down = cls_criterion(out_down, labels)
            loss = (1 - kd_alpha) * cls_loss_down
            epoch_acc += float((pred_ids_down[:, 0].reshape(labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())


            cls_kd_index = np.where(flag_numpy == 1)[0]
            loc_kd_index = np.where(flag_numpy == 2)[0]
            # print(flag_numpy)
            # print(cls_kd_index, len(cls_kd_index))
            # print(loc_kd_index, len(loc_kd_index))
            batch_size = flag_numpy.shape[0]

            if len(cls_kd_index) > 0:
                # cls teacher kd #
                _, _, _, _, _, cls_teacher_out_down, _, _ = cls_teacher(images)
                cls_teacher_out_down = cls_teacher_out_down.to(cfg.device).detach()
                # 参考mnist内容
                cls_KD_loss = kld_criterion(F.softmax(out_down[cls_kd_index] / cls_kd_T, dim=1),
                                            F.softmax(cls_teacher_out_down[cls_kd_index] / cls_kd_T, dim=1))
                loss = loss + kd_alpha * cls_KD_loss
            if len(loc_kd_index) > 0:
                # loc teacher kd #
                _, _, _, _, loc_teacher_cam_down, _, _, _ = loc_teacher(images)
                loc_teacher_cam_down = loc_teacher_cam_down.to(cfg.device).detach()
                cam_down_flatten = cam_down.reshape([batch_size, -1])
                loc_teacher_cam_down_flatten = loc_teacher_cam_down.reshape([batch_size, -1])
                # todo kd_fe_loss 1 和 7 效果差不多，但是 5 对 定位 更友好
                # if kd_fe_loss == 0:
                #     loc_KD_loss = _feature_kd_loss_0(F.sigmoid(cam_down_flatten[loc_kd_index] / loc_kd_T),
                #                                          F.sigmoid(loc_teacher_cam_down_flatten[loc_kd_index] / loc_kd_T), mse_criterion)
                if kd_fe_loss == 1:
                    loc_KD_loss = _feature_kd_loss_1(F.sigmoid(cam_down_flatten[loc_kd_index] / loc_kd_T),
                                                         F.sigmoid(loc_teacher_cam_down_flatten[loc_kd_index] / loc_kd_T), mse_criterion)
                # elif kd_fe_loss == 2:
                #     loc_KD_loss = _feature_kd_loss_0(F.sigmoid(cam_down_flatten[loc_kd_index] / loc_kd_T),
                #                                          F.sigmoid(loc_teacher_cam_down_flatten[loc_kd_index] / loc_kd_T), kld_criterion)
                elif kd_fe_loss == 3:
                    loc_KD_loss = _feature_kd_loss_1(F.sigmoid(cam_down_flatten[loc_kd_index] / loc_kd_T),
                                                         F.sigmoid(loc_teacher_cam_down_flatten[loc_kd_index] / loc_kd_T), kld_criterion)
                # elif kd_fe_loss == 4:
                #     loc_KD_loss = _feature_kd_loss_0(F.sigmoid(cam_down_flatten[loc_kd_index] / loc_kd_T),
                #                                          F.sigmoid(loc_teacher_cam_down_flatten[loc_kd_index] / loc_kd_T), smoothL1_criterion)
                elif kd_fe_loss == 5:
                    loc_KD_loss = _feature_kd_loss_1(F.sigmoid(cam_down_flatten[loc_kd_index] / loc_kd_T),
                                                         F.sigmoid(loc_teacher_cam_down_flatten[loc_kd_index] / loc_kd_T), smoothL1_criterion)
                # elif kd_fe_loss == 6:
                #     loc_KD_loss = _feature_kd_loss_0(F.sigmoid(cam_down_flatten[loc_kd_index] / loc_kd_T),
                #                                          F.sigmoid(loc_teacher_cam_down_flatten[loc_kd_index] / loc_kd_T), l1_criterion)
                elif kd_fe_loss == 7:
                    loc_KD_loss = _feature_kd_loss_1(F.sigmoid(cam_down_flatten[loc_kd_index] / loc_kd_T),
                                                         F.sigmoid(loc_teacher_cam_down_flatten[loc_kd_index] / loc_kd_T), l1_criterion)

                loss = loss + kd_alpha * loc_KD_loss * kd_fe_times
            # loc_KD_loss = _kld_criterion(cam_down_flatten / loc_kd_T, loc_teacher_cam_down_flatten / loc_kd_T) / (loc_kd_T**2)
            # l = {'type': feature_type, 'weight': float(weight), 'function': FeatureLoss(loss=nn.L1Loss())}

            if cfg.attention:
               student_net_module.update_mask(old_create_attention(labels=labels, pred_ids=pred_ids_up, cam=cam_up))

            print("epoch: ", e + 1, " | batch: ", i, "/", len(train_loader), " | loss: ", loss.item())

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        if logger == None:
            logger = SummaryWriter(os.path.join("tensorboard", args.dir, tim))

        if args.dataset == "cub":
            if (e >= 49 and (e+1) % 5 == 0) or epoch - (e+1) < 5:
                net_save(student_net_module, os.path.join(net_dir, 'net_train_{}_{}_{}.pth'.format(args.backbone, args.dataset, e + 1)))
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
            net_save(student_net_module, os.path.join(net_dir, 'net_train_{}_{}_{}.pth'.format(args.backbone, args.dataset, e + 1)))
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
