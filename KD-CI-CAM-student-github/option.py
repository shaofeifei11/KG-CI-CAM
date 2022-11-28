import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--danet", type=int, default=0, choices=[0, 1],
                        help="0: not use danet; 1: danet.")

    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--time", type=str, default=None,
                        help="procedure run time")
    parser.add_argument("--update_rate", type=float, default=0.01,
                        help="update attention rate")
    parser.add_argument("--dataset", type=str, default="cub", choices=["cub", "imagenet"],
                        help="cub: CUB-200-2011; imagenet: ILSVRC 2016")
    parser.add_argument("--backbone", type=str, default="vgg16", choices=["vgg16", "inceptionV3", "inceptionV3_old", "vgg16_baseline", "inceptionV3_baseline"])
    parser.add_argument("--backbone_rate", default=0.1, type=float)
    parser.add_argument("--epoch", type=int, default=100)

    parser.add_argument("--function", default='quadratic', type=str,
                        help='select CCAM function: linear, quadratic，mean')
    parser.add_argument("--mean_num", type=int, default=20,
                        help="mean num")
    parser.add_argument("--decay_epoch", type=int, default=2,
                        help="second decay epoch in imagenet")
    parser.add_argument("--decay_rate", type=float, default=0.5,
                        help="second decay rate in imagenet")

    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument("--seg_thr", type=float, default=0.0,
                        help="segmentation threshold")

    parser.add_argument("--model_path", type=str, default=None,
                        help="None: random init. Else, load model path weight")


    parser.add_argument("--cls_teacher_input_size", type=int, default=256,
                        help="image resize input size")
    parser.add_argument("--cls_teacher_T", type=float, default=15.0,
                        help="cls temperature")

    parser.add_argument("--loc_teacher_input_size", type=int, default=256,
                        help="image resize input size")
    parser.add_argument("--loc_teacher_T", type=float, default=15.0,
                        help="loc temperature")

    # parser.add_argument("--cls_selection", type=str, default="down", choices=["up", "down", "diff", "aux", "comb"],
    #                     help="None: random init. Else, load model path weight")

    parser.add_argument("--max_num", type=int, default=100, help="max number decay")
    parser.add_argument("--inception_reduce", type=float, default=0.1, help="tt")

    parser.add_argument("--kd_mode", type=int, default=3, choices=[0, 1, 2, 3],
                        help="0: not knowledge distillation, 1: cls kd, 2: loc kd, 3 cls+loc kd")

    parser.add_argument("--kd_alpha", type=float, default=0.3, help="kd_alpha")

    parser.add_argument("--kd_fe_loss", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--kd_fe_times", type=float, default=1.0, help="kd_fe_times")
    parser.add_argument("--kd_fe_zero", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cls_min_scale", type=float, default=0.2)
    parser.add_argument("--cls_max_scale", type=float, default=1.0)
    parser.add_argument("--loc_min_scale", type=float, default=0.1)
    parser.add_argument("--loc_max_scale", type=float, default=1.0)
    parser.add_argument("--randAug", type=int, default=0, choices=[0,1,2,3,4,5,6,7], help="0:no; 1:stu; 2:cls; 3:loc; 4:stu+cls; 5:stu+loc; 6:cls+loc; 7:stu+cls+loc")
    parser.add_argument("--clsTea", type=int, default=0)
    parser.add_argument("--locTea", type=int, default=0)

    parser.add_argument("--distillation", type=str, default="random", choices=["random", "none", "cls", "loc"])
    parser.add_argument("--attention", type=int, default=1, choices=[0, 1],
                        help="0: not update attention; 1: update attention")

    parser.add_argument("--ttt", type=int, default=0)

    args = parser.parse_args()
    if args.dataset == "cub":
        args.num_classes = 200
    else:
        args.num_classes = 1000

    args.dir = str(args.dataset) \
               + "_bz_" + str(args.batch_size) \
               + "_bkB_" + str(args.backbone) \
               + "_bkR_" + str(args.backbone_rate) \
               + "_dcE_" + str(args.decay_epoch) \
               + "_dcR_" + str(args.decay_rate) \
               + "_func_" + str(args.function) \
               + "_mN_" + str(args.mean_num) \
               + "_sgT_" + str(args.seg_thr) \
               + "_CipS_" + str(args.cls_teacher_input_size) \
               + "_CT_" + str(args.cls_teacher_T) \
               + "_LipS_" + str(args.loc_teacher_input_size) \
               + "_LT_" + str(args.loc_teacher_T) \
               + "_kd_" + str(args.kd_mode) \
               + "_kdA_" + str(args.kd_alpha) \
               + "_kdFLs_" + str(args.kd_fe_loss) \
               + "_kdFTs_" + str(args.kd_fe_times) \
               + "_kdFZo_" + str(args.kd_fe_zero) \
               + "_lr_" + str(args.lr) \
               + "_dn_" + str(args.danet) \
               + "_uR_" + str(args.update_rate) \
               + "_Cmin_" + str(args.cls_min_scale) \
               + "_Cmax_" + str(args.cls_max_scale) \
               + "_Lmin_" + str(args.loc_min_scale) \
               + "_Lmax_" + str(args.loc_max_scale) \
               + "_rAug_" + str(args.randAug) \
               + "_cTea_" + str(args.clsTea) \
               + "_lTea_" + str(args.locTea) \
               + "_dist_" + str(args.distillation) \
               + "_att_" + str(args.attention) \

    return args

if __name__ == '__main__':
    args_parser()

