import os
import time

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

import config as cfg
from dataset.CUB_dataset_randomresizecrop_randaug import CUB_Dataset_RandomResizeCrop_RandAug
from dataset.ImageNet_dataset_Randomresizecrop_randAug import ImageNet_Dataset_Randomresizecrop_RandAug
from model.cub_inceptionV3_nonlocal_0702 import CUB_InceptionV3_NonLocal_0702
from model.cub_vgg16_0702 import CUB_VGG16_0702
from model.imagenet_vgg16_0702 import ImageNet_VGG16_0702
from option import args_parser
from test_center_crop import inference
from utils import net_load, net_save, old_create_attention, torch_cam_combination
from utils import poly_lr_scheduler_kd, poly_lr_scheduler_imagenet


def compute_entropy(x):
    """
    :param x: [sampler_batch_size, class_num]
    a = [[0.1, 0.1, 0.1, 0.1, 0.6],
          [0.2, 0.2, 0.2, 0.2, 0.2]]
    print(compute_entropy(np.asarray(a)))
    >> [1.77095059  2.32192809]
    :return:
    """
    k = torch.log2(x)
    where_are_inf = torch.isinf(k)
    k[where_are_inf] = 0

    where_are_nan = torch.isnan(k)
    k[where_are_nan] = -100

    # entropy = torch.mean((-1 * torch.sum(torch.multiply(x, k), dim=-1))).to(cfg.device)  # 2022-06-19 之前
    entropy = torch.mean((-1 * torch.mean(torch.multiply(x, k), dim=-1))).to(cfg.device)
    return entropy  # [sampler_batch_size * point_number]

if __name__ == '__main__':
    args = args_parser()

    epoch = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    gpus = args.gpu.replace("_", ",")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    print("gpus: ", gpus)
    cls_criterion = nn.CrossEntropyLoss().to(cfg.device)
    box_criterion = nn.SmoothL1Loss().to(cfg.device)
    sigmoid = nn.Sigmoid()

    if args.time == None:
        tim = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    else:
        tim = args.time
    logger = None
    net_dir = os.path.join("save_model", args.dir, tim)
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

    if args.dataset == "cub":
        if args.scale == 1:
            pass
            # train_data = CUB_Dataset_RandomResizeCrop(input_size=args.input_size, train=True, danet=args.danet,
            #                                           min_scale=args.min_scale, max_scale=args.max_scale)
        # elif args.scale == 2:
        #     train_data = CUB_Dataset_RandomScale(input_size=args.input_size, train=True, danet=args.danet,
        #                                          min_scale=args.min_scale, max_scale=args.max_scale)
        elif args.scale == 3:
            train_data = CUB_Dataset_RandomResizeCrop_RandAug(input_size=args.input_size, train=True, danet=args.danet,
                                                              min_scale=args.min_scale, max_scale=args.max_scale)
        else:
            pass
            # train_data = CUB_Dataset(input_size=args.input_size, train=True, danet=args.danet)
        if args.backbone == "vgg16":
            net = CUB_VGG16_0702(args)
        elif args.backbone == "inceptionV3":
            net = CUB_InceptionV3_NonLocal_0702(args)
    else:
        if args.scale == 1:
            pass
        #     train_data = ImageNet_Dataset_Randomresizecrop(input_size=args.input_size, train=True, danet=args.danet,
        #                                                    min_scale=args.min_scale, max_scale=args.max_scale)
        # # elif args.scale == 2:
        #     train_data = CUB_Dataset_RandomScale(input_size=args.input_size, train=True, danet=args.danet,
        #                                          min_scale=args.min_scale, max_scale=args.max_scale)
        elif args.scale == 3:
            train_data = ImageNet_Dataset_Randomresizecrop_RandAug(input_size=args.input_size, train=True,
                                                                   danet=args.danet,
                                                                   min_scale=args.min_scale, max_scale=args.max_scale)
        else:
            pass
            # train_data = ImageNet_Dataset(input_size=args.input_size, train=True, danet=args.danet)

        if args.backbone == "vgg16" and args.new_image_backbone == 0:
            net = ImageNet_VGG16_0702(args)

    train_data_length = len(train_data)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)

    begin_epoch = 0
    model_path = args.model_path
    if model_path != None:
        print("\n\n########################################################################")
        print("model_path: ", model_path)
        print("model_args: ", args.dir)
        net = net_load(net, model_path)
        begin_epoch = int(model_path.split("_")[-1].split(".")[0])

    print("run here! ", len(gpus))
    if len(gpus) > 1:
        print("DataParallel! ")
        net = torch.nn.DataParallel(net).to(cfg.device)
        net_module = net.module
    else:
        print("one machine! ")
        net = net.to(cfg.device)
        net_module = net

    if args.shared_classifier:
        optimizer = torch.optim.Adam([{'params': net_module.up_classifier.parameters()},
                                      {'params': net_module.mask2attention.parameters()},
                                      {'params': net_module.backbone.parameters()}], lr=lr)
    else:
        optimizer = torch.optim.Adam([{'params': net_module.up_classifier.parameters()},
                                      {'params': net_module.down_classifier.parameters()},
                                      {'params': net_module.mask2attention.parameters()},
                                      {'params': net_module.backbone.parameters()}], lr=lr)

    aux_rate = args.aux_rate

    # net.train()
    # init ########################
    if args.batch_size > 15 and args.dataset == "imagenet" and model_path == None:
        init_loader = DataLoader(train_data, batch_size=6, shuffle=False, num_workers=6)
        count = 0
        for i, dat in enumerate(init_loader):
            if count > 500:
                break
            else:
                images, labels = dat
                images, labels = images.to(cfg.device), labels.to(cfg.device)
                labels = labels.long()
                optimizer.zero_grad()
                cam_up, out_up, pred_sort_up, pred_ids_up, cam_down, out_down, pred_sort_down, pred_ids_down = net(
                    images)
                ############ up ###########
                out_up = out_up.float()
                cls_loss_up = cls_criterion(out_up, labels)
                loss = cls_loss_up
                cls_loss_up_cpu = cls_loss_up.item()
                print("init", count, cls_loss_up_cpu)
                loss.backward()
                optimizer.step()
                count = count + 1
        init_loader = None
    ###############################
    loader_change = False

    for e in range(begin_epoch, epoch):
        if args.decay:
            if args.dataset == "cub":
                poly_lr_scheduler_kd(optimizer, lr, e, max_iter=epoch, dataset=args.dataset,
                                     backbone_rate=args.backbone_rate, decay_rate=args.decay_rate,
                                     decay_epoch=args.decay_epoch, inception_reduce=args.inception_reduce,
                                     max_num=args.max_num)
            else:
                poly_lr_scheduler_imagenet(optimizer, lr, e, max_iter=epoch, dataset=args.dataset,
                                           backbone_rate=args.backbone_rate, decay_rate=args.decay_rate,
                                           decay_epoch=args.decay_epoch, inception_reduce=args.inception_reduce,
                                           max_num=args.max_num)

        net.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_bbox_loss = 0
        if e == 0 and batch_size > 30:
            train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=6)  # 第一个epoch不适合太大，不然会出现nan
            loader_change = True

        for i, dat in enumerate(train_loader):
            images, labels = dat
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            labels = labels.long()
            optimizer.zero_grad()
            cam_up, out_up, pred_sort_up, pred_ids_up, cam_down, out_down, pred_sort_down, pred_ids_down = net(
                images)
            diff_out, diff_ids, aux_out, aux_ids, fore_out, fore_ids, back_out = net.aux_classification(images)

            ############ up ###########
            cls_loss_up = cls_criterion(out_up, labels)
            loss = cls_loss_up
            cls_loss_up_cpu = cls_loss_up.item()
            # print("cls_loss_up_cpu", cls_loss_up_cpu)
            epoch_loss += cls_loss_up_cpu
            epoch_acc += float((pred_ids_up[:, 0].reshape(
                labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())
            ############ down ###########
            cls_loss_down = cls_criterion(out_down, labels)
            loss = loss + cls_loss_down
            cls_loss_down_cpu = cls_loss_down.item()
            # print("cls_loss_down_cpu", cls_loss_down_cpu)
            epoch_loss += cls_loss_down_cpu
            epoch_acc += float((pred_ids_down[:, 0].reshape(
                labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())
            ############ foreground-background diff, aux ############
            # diff_out, aux_out, fore_out, back_out = diff_out.float(), aux_out.float(), fore_out.float(), back_out.float()
            if "diff" in args.aux_type:
                # diff
                cls_loss_dif = cls_criterion(diff_out, labels)
                loss = loss + cls_loss_dif * aux_rate
                cls_loss_dif_cpu = cls_loss_dif.item()
                # print("cls_loss_dif_cpu", cls_loss_dif_cpu)
                epoch_loss += cls_loss_dif_cpu
                # epoch_acc += float((diff_ids[:, 0].reshape(
                #     labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())
            if "aux" in args.aux_type:
                # aux
                cls_loss_aux = cls_criterion(aux_out, labels)
                loss = loss + cls_loss_aux * aux_rate
                cls_loss_aux_cpu = cls_loss_aux.item()
                # print("cls_loss_aux_cpu", cls_loss_aux_cpu)
                epoch_loss += cls_loss_aux_cpu
                # epoch_acc += float((aux_ids[:, 0].reshape(
                #     labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())
            if "fore" in args.aux_type:
                # fore
                cls_loss_fore = cls_criterion(fore_out, labels)
                loss = loss + cls_loss_fore * aux_rate
                cls_loss_fore_cpu = cls_loss_fore.item()
                # print("cls_loss_fore_cpu", cls_loss_fore_cpu)
                epoch_loss += cls_loss_fore_cpu
                # epoch_acc += float((fore_ids[:, 0].reshape(
                #     labels.size()).detach().cpu().numpy() == labels.detach().cpu().numpy()).sum())
            if "back" in args.aux_type:
                # back
                cls_loss_back = compute_entropy(back_out)
                loss = loss - cls_loss_back * aux_rate
                cls_loss_back_cpu = cls_loss_back.item()
                # print("cls_loss_back_cpu", cls_loss_back_cpu)
                epoch_loss = epoch_loss + cls_loss_back_cpu

            if args.spa_loss == 1:
                activation_maps = torch_cam_combination(cam_down, None, pred_ids_down, args.function,
                                                        args.mean_num)  # [bz, h*w]
                # loss = loss + args.spa_loss_rate * torch.mean(torch.abs(activation_maps))
                loss = loss + args.spa_loss_rate * torch.mean(activation_maps)
            elif args.spa_loss == 2:
                bz, nc, h, w = cam_down.shape
                activation_maps_spa = torch_cam_combination(cam_down, None, pred_ids_down, args.function,
                                                            args.mean_num)  # [bz, h*w]
                activation_maps_spa = activation_maps_spa.reshape(bz, h, w)
                activation_maps_spa = net.upsample(activation_maps_spa.unsqueeze(dim=1).to(cfg.device)).to(
                    cfg.device).squeeze(
                    dim=1).to(
                    cfg.device)
                # loss = loss + args.spa_loss_rate * torch.mean(torch.abs(activation_maps_spa))
                loss = loss + args.spa_loss_rate * torch.mean(activation_maps_spa)
            elif args.spa_loss == 3:
                # 用了 sigmoid
                activation_maps = torch_cam_combination(cam_down, None, pred_ids_down, args.function,
                                                        args.mean_num)  # [bz, h*w]
                loss = loss + args.spa_loss_rate * torch.mean(sigmoid(activation_maps))
            elif args.spa_loss == 4:
                bz, nc, h, w = cam_down.shape
                activation_maps_spa = torch_cam_combination(cam_down, None, pred_ids_down, args.function,
                                                            args.mean_num)  # [bz, h*w]
                activation_maps_spa = activation_maps_spa.reshape(bz, h, w)
                activation_maps_spa = net.upsample(activation_maps_spa.unsqueeze(dim=1).to(cfg.device)).to(
                    cfg.device).squeeze(
                    dim=1).to(
                    cfg.device)
                # loss = loss + args.spa_loss_rate * torch.mean(torch.abs(activation_maps_spa))
                loss = loss + args.spa_loss_rate * torch.mean(sigmoid(activation_maps_spa))

            if args.attention:
               net_module.update_mask(old_create_attention(labels=labels, pred_ids=pred_ids_up, cam=cam_up))
            print("epoch: ", e + 1, " | batch: ", i, "/", len(train_loader), " | loss: ", loss.item())
            loss.backward()
            optimizer.step()

        if logger == None:
            logger = SummaryWriter(os.path.join("tensorboard", args.dir, tim))
        if args.dataset == "cub":
            if (e + 1) % 5 == 0:
                net_save(net_module, os.path.join(net_dir, 'net_train_{}_{}_{}.pth'.format(args.backbone, args.dataset, e + 1)))
            if (e+1) % 5 == 0:
                train_avg_epoch_acc = epoch_acc / train_data_length
                train_avg_epoch_cls_loss = epoch_loss / train_data_length
                train_avg_epoch_box_loss = epoch_bbox_loss / train_data_length
                print("Train Epoch: ", e+1, " | Train Avg Acc: ", train_avg_epoch_acc, " | Train Avg Cls Loss: ",
                      train_avg_epoch_cls_loss, " | Train Avg Box Loss: ", train_avg_epoch_box_loss)
                logger.add_scalar("Train Avg Acc", train_avg_epoch_acc, e + 1)
                logger.add_scalar("Train Avg Cls Loss", train_avg_epoch_cls_loss, e + 1)
                logger.add_scalar("Train Avg Box Loss", train_avg_epoch_box_loss, e + 1)
                c1, l1, c5, l5, gt_l = inference(net_module, args)
                logger.add_scalar("classification top 1 accuracy: ", c1, e+1)
                logger.add_scalar("localization top 1 accuracy: ", l1, e+1)
                logger.add_scalar("classification top 5 accuracy: ", c5, e+1)
                logger.add_scalar("localization top 5 accuracy: ", l5, e+1)
                logger.add_scalar("gt localization accuracy: ", gt_l, e + 1)
        else:
            net_save(net_module, os.path.join(net_dir, 'net_train_{}_{}_{}.pth'.format(args.backbone, args.dataset, e + 1)))
            if (e+1) % 1 == 0:
                train_avg_epoch_acc = epoch_acc / train_data_length
                train_avg_epoch_cls_loss = epoch_loss / train_data_length
                train_avg_epoch_box_loss = epoch_bbox_loss / train_data_length
                print("Train Epoch: ", e+1, " | Train Avg Acc: ", train_avg_epoch_acc, " | Train Avg Cls Loss: ",
                      train_avg_epoch_cls_loss, " | Train Avg Box Loss: ", train_avg_epoch_box_loss)
                logger.add_scalar("Train Avg Acc", train_avg_epoch_acc, e + 1)
                logger.add_scalar("Train Avg Cls Loss", train_avg_epoch_cls_loss, e + 1)
                logger.add_scalar("Train Avg Box Loss", train_avg_epoch_box_loss, e + 1)
                c1, l1, c5, l5, gt_l = inference(net_module, args)
                logger.add_scalar("classification top 1 accuracy: ", c1, e + 1)
                logger.add_scalar("localization top 1 accuracy: ", l1, e + 1)
                logger.add_scalar("classification top 5 accuracy: ", c5, e + 1)
                logger.add_scalar("localization top 5 accuracy: ", l5, e + 1)
                logger.add_scalar("gt localization accuracy: ", gt_l, e + 1)

                # seg_thr_list, c1_list, l1_list, c5_list, l5_list, gt_l_list = inference_multi_segthr(net_module, args)
                # logger.add_scalar("cls-1 accuracy", c1_list[0], e + 1)
                # logger.add_scalar("cls-5 accuracy", c5_list[0], e + 1)
                # for i in range(len(seg_thr_list)):
                #     logger.add_scalar(str(seg_thr_list[i]) + "_loc-1 accuracy", l1_list[i], e + 1)
                #     logger.add_scalar(str(seg_thr_list[i]) + "_loc-5 accuracy", l5_list[i], e + 1)
                #     logger.add_scalar(str(seg_thr_list[i]) + "_gt-know accuracy", gt_l_list[i], e + 1)

        if loader_change:
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6)
            loader_change = False

    logger.close()
