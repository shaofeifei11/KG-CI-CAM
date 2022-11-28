import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--attention_size", type=int, default=14, help="vgg16: 14; inceptionV3: 14")
    parser.add_argument("--danet", type=int, default=0, choices=[0, 1],
                        help="0: not use danet; 1: danet.")
    parser.add_argument("--update_rate", type=float, default=0.01,
                        help="update attention rate")
    parser.add_argument("--backbone_rate", default=0.1, type=float)
    parser.add_argument("--function", default='quadratic', type=str,
                        help='select CCAM function: linear, quadratic，mean')
    parser.add_argument("--mean_num", type=int, default=20,
                        help="mean num")
    parser.add_argument("--decay_epoch", type=int, default=2,
                        help="second decay epoch in imagenet")
    parser.add_argument("--decay_rate", type=float, default=0.5,
                        help="second decay rate in imagenet")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument("--seg_thr", type=float, default=0.0,
                        help="segmentation threshold")

    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--time", type=str, default=None,
                        help="procedure run time")
    parser.add_argument("--dataset", type=str, default="cub", choices=["cub", "imagenet"],
                        help="cub: CUB-200-2011; imagenet: ILSVRC 2016")

    parser.add_argument("--backbone", type=str, default="vgg16", choices=["vgg16", "inceptionV3", "vgg16Relu", "vgg_low"])

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--non_local", type=int, default=1, choices=[0, 1],
                        help="0: vgg16; 1: the vgg16 that adds non-local attention")
    parser.add_argument("--shared_classifier", type=int, default=1, choices=[0, 1],
                        help="0: not shared; 1: shared classifier")
    parser.add_argument("--combination", type=int, default=1, choices=[0, 1],
                        help="0: common cam; 1: use linear function to combination cam")
    parser.add_argument("--input_size", type=int, default=256,
                        help="image resize input size")

    parser.add_argument("--decay", type=int, default=1, choices=[0, 1],
                        help="0: learning rate is fixed; 1: learning rate is decay")
    parser.add_argument("--attention", type=int, default=1, choices=[0, 1],
                        help="0: not update attention; 1: update attention")
    parser.add_argument("--pretrain", type=int, default=1, choices=[0, 1],
                        help="0: not use pre-train vgg16 conv weight; 1: use pre-train vgg16 conv weight")
    parser.add_argument("--model_path", type=str, default=None,
                        help="None: random init. Else, load model path weight")

    parser.add_argument("--model_path_loc", type=str, default=None,
                        help="None: random init. Else, load model path weight")
    parser.add_argument("--input_size_loc", type=int, default=0,
                        help="None: random init. Else, load model path weight")

    parser.add_argument("--cls_selection", type=str, default="down", choices=["up", "down", "diff", "aux", "comb"],
                        help="None: random init. Else, load model path weight")

    parser.add_argument("--aux_rate", type=float, default=0.1, help="tt")

    parser.add_argument("--max_num", type=int, default=100, help="max number decay")
    parser.add_argument("--inception_reduce", type=float, default=0.1, help="tt")

    parser.add_argument("--sgd", type=int, default=0,  choices=[0, 1], help="0: adam, 1: sgd")

    parser.add_argument("--new_image_backbone", type=int, default=0, choices=[0, 1])

    parser.add_argument("--cutmix_prob", type=float, default=0.0, help="if <= 0, not use cutmix; eles using CutMix, https://github.com/clovaai/CutMix-PyTorch")
    parser.add_argument("--cutmix", type=int, default=1, choices=[1, 2])

    # parser.add_argument("--backbone_rate_begin", type=float, default=0.1) 没有用，取消了
    parser.add_argument("--scale", type=int, default=0, choices=[0, 1, 2, 3], help="1: randomResizeCrop; 2: randomScale; 3: RandomResizeCrop_RandAug")
    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=1.0)

    parser.add_argument("--spa_loss", type=int, default=0, choices=[0, 1, 2, 3, 4], help="using spa loss from FAM. 1: before upsample, 2: after upsample")
    parser.add_argument("--spa_loss_rate", type=float, default=0.04)

    parser.add_argument("--aux_type", type=str, default="diff_aux")

    args = parser.parse_args()
    if args.dataset == "cub":
        args.num_classes = 200
    else:
        args.num_classes = 1000

    args.dir = str(args.dataset) \
               + "_sh_" + str(args.shared_classifier) \
               + "_bz_" + str(args.batch_size) \
               + "_ipS_" + str(args.input_size) \
               + "_cpS_" + str(args.crop_size) \
               + "_atS_" + str(args.attention_size) \
               + "_bkB_" + str(args.backbone) \
               + "_bkR_" + str(args.backbone_rate) \
               + "_func_" + str(args.function) \
               + "_mN_" + str(args.mean_num) \
               + "_lr_" + str(args.lr) \
               + "_epc_" + str(args.epoch) \
               + "_dn_" + str(args.danet) \
               + "_dcE_" + str(args.decay_epoch) \
               + "_dcR_" + str(args.decay_rate) \
               + "_sgT_" + str(args.seg_thr) \
               + "_auxR_" + str(args.aux_rate) \
               + "_dcMn_" + str(args.max_num) \
               + "_incR_" + str(args.inception_reduce) \
               + "_uR_" + str(args.update_rate) \
               + "_sgd_" + str(args.sgd) \
               + "_newB_" + str(args.new_image_backbone) \
               + "_scale_" + str(args.scale) \
               + "_minSca_" + str(args.min_scale) \
               + "_maxSca_" + str(args.max_scale) \
               + "_spa_" + str(args.spa_loss) \
               + "_spaR_" + str(args.spa_loss_rate) \
               + "_auxType_" + str(args.aux_type)

    return args

if __name__ == '__main__':
    args_parser()

