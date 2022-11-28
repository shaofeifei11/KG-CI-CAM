import torch
from torch.utils.data import DataLoader
from dataset.CUB_dataset_center_crop import CUB_Dataset_Center_Crop_Test
from dataset.ImageNet_dataset_center_crop import ImageNet_Dataset_Center_Crop_Test

from model.cub_vgg16_0702 import CUB_VGG16_0702
from model.imagenet_vgg16_0702 import ImageNet_VGG16_0702
from model.cub_inceptionV3_nonlocal_0702 import CUB_InceptionV3_NonLocal_0702

import config as cfg
from option import args_parser
from utils import pred_bbox_resize_imagesize, localization_acc, net_load
import os
import numpy as np
import time
from PIL import Image, ImageDraw

def inference(net, args):
    seg_thr = args.seg_thr
    # seg_thr = 0.25  # todo 为了应付 inceptionV3 的 center crop 测试
    print("\n\n\n# ", seg_thr, args.input_size)
    start_time = time.time()

    print_test_image_dir = os.path.join("print_test_image", str(seg_thr))
    if not os.path.exists(print_test_image_dir):
        os.makedirs(print_test_image_dir)

    net.train(False)
    # assist work in test using center crop for more accuracy classification
    if args.dataset == "cub":
        test_data_crop = CUB_Dataset_Center_Crop_Test(input_size=args.input_size)

    else:
        test_data_crop = ImageNet_Dataset_Center_Crop_Test(input_size=args.input_size)

    # begin test
    batch_size = 16
    test_loader = DataLoader(test_data_crop, batch_size=batch_size, shuffle=False, num_workers=6)
    local_acc0_sum = 0
    correct0_sum = 0
    local_acc5_sum = 0
    correct5_sum = 0
    gt_local_sum = 0

    net.eval()
    with torch.no_grad():
        for i, dat in enumerate(test_loader):
            image_list, label_list, img_size = dat
            image_list = image_list.to(cfg.device)
            label_list = label_list.to(cfg.device)
            img_size = img_size.to(cfg.device)
            _, _, _, pred_ids_up, _, _, _, pred_ids_down = net(image_list)
            batch_size = label_list.shape[0]  # update batch_size
            # pred_ids_up, pred_ids_down = net.update_pred_ids(pred_ids_up_assist[batch_size*i: batch_size*(i+1)], pred_ids_down_assist[batch_size*i: batch_size*(i+1)])
            label_numpy = label_list.detach().cpu().numpy()

            pred_ids = pred_ids_down
            # down
            pred_bbox = net.compute_rois_down(seg_thr=seg_thr, topk=cfg.top_k,
                                              combination=args.combination)
            # compute gt
            pred_gt_bbox = net.compute_gt_rois_down(seg_thr=seg_thr, labels=label_numpy,
                                                    combination=args.combination)

            pred_ids_numpy = pred_ids.detach().cpu().numpy()
            label_list_numpy = label_list.detach().cpu().numpy()
            for b in range(batch_size):
                bbox_list = torch.from_numpy(np.asarray(test_data_crop.bbox_list[i*batch_size+b : i*batch_size+b+1])).to(cfg.device)

                local_acc0 = 0
                local_acc5 = 0
                correct5 = 0
                local_acc_temp = 0
                correct_temp = 0

                for j in range(cfg.top_k):
                    # down
                    class_match = pred_ids_numpy[b, j] == label_list_numpy[b]
                    correct_temp += class_match
                    # pred_bbox_image_size = pred_bbox_resize_imagesize(pred_bbox[b][j], img_size[b])
                    pred_bbox_image_size = pred_bbox[b][j]
                    local_acc_temp += localization_acc(pred_bbox_image_size, bbox_list[0], class_match)
                    if j == 0:
                        correct0 = class_match
                        local_acc0 = localization_acc(pred_bbox_image_size, bbox_list[0], class_match)
                        # if correct0:
                        #     # output image
                        #     # PIL
                        #     img_name = test_data.name_list[i*batch_size + b]
                        #     pil_img = Image.open(img_name).convert('RGB')
                        #     rect = ImageDraw.ImageDraw(pil_img)
                        #     rec_bbox = list(pred_bbox_image_size.detach().cpu().numpy())
                        #     rect.rectangle(rec_bbox, outline="black", width=2)
                        #     pil_img.save(os.path.join(print_test_image_dir, str(i*batch_size + b)+"_PIL.png"))
                # down
                if local_acc_temp >= 1.0:
                    local_acc5 = 1.0
                local_acc0_sum = local_acc0_sum + local_acc0
                local_acc5_sum = local_acc5_sum + local_acc5
                if correct_temp >= 1:
                    correct5 = 1.0
                correct0_sum = correct0_sum + correct0
                correct5_sum = correct5_sum + correct5
                # compute gt
                # pred_gt_image_size = pred_bbox_resize_imagesize(pred_gt_bbox[b], img_size[b])
                pred_gt_image_size = pred_gt_bbox[b]
                gt_local_sum += localization_acc(pred_gt_image_size, bbox_list[0], True)

    data_length = len(test_data_crop)
    c1 = correct0_sum / data_length
    l1 = local_acc0_sum / data_length
    c5 = correct5_sum / data_length
    l5 = local_acc5_sum / data_length
    gt_l = gt_local_sum / data_length
    print("Inference Results: ")
    print("classification top 1 accuracy: ", c1)
    print("classification top 5 accuracy: ", c5)
    print("localization top 1 accuracy: ", l1)
    print("localization top 5 accuracy: ", l5)
    # compute gt
    print("gt localization accuracy: ", gt_l)

    print("run time： ", time.time() - start_time)
    return c1, l1, c5, l5, gt_l


if __name__ == '__main__':
    args = args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset == "cub":
        if args.backbone == "vgg16":
            net = CUB_VGG16_0702(args)
        elif args.backbone == "inceptionV3":
            net = CUB_InceptionV3_NonLocal_0702(args)
    else:
        if args.backbone == "vgg16":
            net = ImageNet_VGG16_0702(args)

    model_path = args.model_path
    if model_path != None:
        print("\n\n########################################################################")
        print("model_path: ", model_path)
        print("model_args: ", args.dir)
        net = net_load(net, model_path)
    print("\n")

    net.to(cfg.device)
    inference(net, args)