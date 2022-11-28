from option import args_parser
args = args_parser()

attention = args.attention # 0: not update attention; 1: update attention

POOLING_SIZE = 7
total_stride = 16
top_k = 5
device = "cuda"

# args 固定参数
update_rate = args.update_rate # update attention rate
attention_size = 14
crop_size = 224
mean_num = args.mean_num # mean num
function = args.function # select CCAM function: linear, quadratic，mean
decay_rate = args.decay_rate # second decay rate in imagenet
non_local = 1 # 0: vgg16; 1: the vgg16 that adds non-local attention
shared_classifier = 1 # 0: not shared; 1: shared classifier
combination = 1 # 0: common cam; 1: use linear function to combination cam
pretrain = 1 # 0: not use pre-train vgg16 conv weight; 1: use pre-train vgg16 conv weight

total_cls_teacher_path = ["../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_304_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_1_dcE_14_dcR_0.5_sgT_0.15_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.1_spa_0_spaR_0.04/2022-05-04_22_09_12/net_train_vgg16_cub_100.pth",  # 无 spa_loss cls 79.62%
                          "../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_304_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_1_dcE_14_dcR_0.5_sgT_0.08_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_spa_2_spaR_0.04/2022-04-29_15_31_38/net_train_vgg16_cub_90.pth",  # cls 79.96% 有 spa_loss
                          "best_weight/vgg_cub/cls/cub_sh_1_bz_6_ipS_304_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_1_dcE_14_dcR_0.5_sgT_0.15_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_spa_0_spaR_0.04/2022-04-29_09_00_14/net_train_vgg16_cub_100.pth",  # cls 79.20% 无 spa_loss
                          "../CI-CAM-KD/save_model/cub_sh_1_bz_6_ipS_304_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_1_dcE_14_dcR_0.5_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_spa_2_spaR_0.04/2022-04-29_12_37_50/net_train_vgg16_cub_75.pth",  # cls 79.48%
                          "best_weight/vgg_cub/cls/dataset_cub_inpSize_304_cropSize_224_attenSize_14_backbone_vgg16_backbone_rate_0.25_func_quadratic_meanNum_20_dEpo_14_dRate_0.5_lr_0.0005_segThr_0.2_epoch_100_danet_1/2022-01-23_12_45_02/net_train_vgg16_cub_75.pth",  # cls 77.58 %
                          "../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_304_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_1_dcE_14_dcR_0.5_sgT_0.08_auxR_1.0_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.2_maxSca_1.0_spa_2_spaR_0.04_auxType_fore/2022-07-03_00_43_26/net_train_vgg16_cub_100.pth"]  # todo 200, cls 79.65% fore

total_loc_teacher_path = ["../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_336_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_0_dcE_14_dcR_0.5_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.1_maxSca_1.0_spa_2_spaR_1e-06/2022-05-07_14_16_15/net_train_vgg16_cub_80.pth",  # 336 cls1 77.87%, loc1 63.55%, gt 80.46% | 304 cls1 77.53%, cls5 93.60%, loc1 62.75%, loc5 75.08%, gt 79.48%
                          "../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_304_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_0_dcE_30_dcR_0.5_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_3_minSca_0.1_maxSca_1.0_spa_2_spaR_1e-06/2022-05-14_15_55_06/net_train_vgg16_cub_100.pth",   # 304 cls1 76.06%, loc1 62.75%, gt 81.36%
                          "../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_336_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_dcR_0.5_lr_0.0005_epc_100_dn_0_dcE_14_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1/2022-03-22_20_56_25/net_train_vgg16_cub_70.pth",  # 336 cls1 74.28%, loc1 61.13%, gt 81.07% | 304 cls1 74.42%, cls5 92.53%, loc1 61.70%, loc5 75.70%, gt 80.89%
                          "best_weight/vgg_cub/loc/dataset_cub_inpSize_304_cropSize_224_attenSize_14_backbone_vgg16_backbone_rate_0.1_func_quadratic_meanNum_20_dEpo_14_dRate_0.5_lr_0.0005_segThr_-0.05_epoch_100_danet_1/2022-01-24_11_47_39/net_train_vgg16_cub_55.pth", # cls 73.20%,  loc 60.17%, gt 80.27% todo 该权重有问题，加载不起来
                          "../CI-CAM-KD/save_model/cub_sh_1_bz_6_ipS_336_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_0_dcE_14_dcR_0.5_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_scale_1_spa_0/2022-04-27_00_19_19/net_train_vgg16_cub_100.pth",  # cls 76.58%, loc 61.29%, gt 79.17%
                          "../CI-CAM-KD/save_model/cub_sh_1_bz_6_ipS_304_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_0_dcE_30_dcR_0.5_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_scale_3_minSca_0.1_maxSca_1.0_spa_2_spaR_1e-06_auxType_diff_aux/2022-07-05_20_53_11/net_train_vgg16_cub_100.pth",  # todo 200 diff_aux 2022-07-05_20_53_11, cls-1 76.34%, cls-5 93.51%, loc-1 62.75%, loc-5 76.25%, gt 81.08%
                          "../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_336_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_0_dcE_14_dcR_0.5_sgT_0.0_auxR_1.0_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_scale_0_minSca_0.2_maxSca_1.0_spa_0_spaR_0.04_auxType_diff_aux/2022-07-07_21_15_08/net_train_vgg16_cub_45.pth",  # cls-1 72.90%, cls-5 92.10%, loc-1 60.22%, loc-5 75.20%, gt 81.15%, ep 45
                          "../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_336_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_0_dcE_14_dcR_0.5_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_scale_1_minSca_0.1_maxSca_1.0_spa_0_spaR_0.04_auxType_diff_aux/2022-07-08_07_15_08/net_train_vgg16_cub_85.pth"]  # cls-1 77.13%, cls-5 93.13%, loc-1 62.22%, loc-5 74.75%, gt 79.93%, ep 85


cls_teacher_path = total_cls_teacher_path[args.clsTea]
loc_teacher_path = total_loc_teacher_path[args.locTea]


if args.backbone == "inceptionV3" or args.backbone == "inceptionV3_old" or args.backbone == "inceptionV3_baseline":
    attention_size = 17
    crop_size = 299

    # 无 spa_loss cls 80.22%
    total_cls_teacher_path = ["../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_350_cpS_299_atS_17_bkB_inceptionV3_bkR_0.6_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_20_dcR_0.5_sgT_0.15_auxR_0.3_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_spa_0_spaR_0.04/2022-05-02_05_07_08/net_train_inceptionV3_cub_70.pth", # 无 spa_loss cls 80.22%
                              "../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_500_cpS_299_atS_17_bkB_inceptionV3_bkR_0.8_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_50_dcR_0.5_sgT_0.25_auxR_0.2_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_3_minSca_0.1_maxSca_1.0_spa_2_spaR_2e-08/2022-05-12_18_59_26/net_train_inceptionV3_cub_95.pth",  # 500 cls 73.59%, loc 61.48%, gt 83.50%, ep95 | 350 cls1 75.56%, cls5 93.11%, loc1 63.29%, loc5 77.94%, gt 83.43%
                              "../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_500_cpS_299_atS_17_bkB_inceptionV3_bkR_0.8_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_50_dcR_0.5_sgT_0.25_auxR_0.2_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.1_maxSca_1.0_spa_2_spaR_2e-08/2022-05-09_23_42_35/net_train_inceptionV3_cub_70.pth",  # 350 cls1 76.41%, loc1 57.70%, gt 75.09%"
                              "../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_350_cpS_299_atS_17_bkB_inceptionV3_bkR_0.6_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_20_dcR_0.5_sgT_0.15_auxR_0.6_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.2_maxSca_1.0_spa_0_spaR_0.04_auxType_fore/2022-07-04_04_57_03/net_train_inceptionV3_cub_90.pth"]  # 106, fore, 350 cls-1 80.55%, cls-5 95.08%, loc-1 40.94%, loc-5 48.14%, gt 50.26%

    total_loc_teacher_path = ["../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_500_cpS_299_atS_17_bkB_inceptionV3_bkR_0.8_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_50_dcR_0.5_sgT_0.25_auxR_0.2_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.1_maxSca_1.0_spa_2_spaR_2e-08/2022-05-09_23_42_35/net_train_inceptionV3_cub_70.pth",  # 350 cls1 76.41%, loc1 57.70%, gt 75.09%
                              "../CI-CAM-KD/new_best_weight/cub_sh_1_bz_6_ipS_500_cpS_299_atS_17_bkB_inceptionV3_bkR_0.8_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_50_dcR_0.5_sgT_0.25_auxR_0.2_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_3_minSca_0.1_maxSca_1.0_spa_2_spaR_2e-08/2022-05-12_18_59_26/net_train_inceptionV3_cub_95.pth",  # 500 cls 73.59%, loc 61.48%, gt 83.50%, ep95 | 350 cls1 75.56%, cls5 93.11%, loc1 63.29%, loc5 77.94%, gt 83.43%
                              "../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_500_cpS_299_atS_17_bkB_inceptionV3_bkR_0.8_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_50_dcR_0.5_sgT_0.25_auxR_0.2_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_scale_3_minSca_0.1_maxSca_1.0_spa_2_spaR_2e-08_auxType_diff_aux/2022-07-05_12_34_45/net_train_inceptionV3_cub_65.pth",  # 350 segthr=0.3 cls-1 73.64%, cls-5 92.59%, loc-1 60.75%, loc-5 76.23%, gt 81.72% | segthr=0.25 500 cls-1 71.57%, cls-5 91.34%, loc-1 61.82%, loc-5 78.34%, gt 85.24% # todo 实验证明 对于 loc teacher 来说，使用 gt高的teacher比使用loc高的teacher效果更好
                              "../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_500_cpS_299_atS_17_bkB_inceptionV3_bkR_0.8_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_50_dcR_0.5_sgT_0.25_auxR_0.2_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_scale_3_minSca_0.1_maxSca_1.0_spa_2_spaR_2e-08_auxType_diff_aux/2022-07-05_12_34_45/net_train_inceptionV3_cub_60.pth",  # 350 segthr=0.3 cls-1 74.87%, cls-5 93.75%, loc-1 61.13%, loc-5 76.33%, gt 81.10% | segthr=0.25 500 cls-1 72.90%, cls-5 92.47%, loc-1 62.50%, loc-5 78.72%, gt 84.73% # todo 实验证明 对于 loc teacher 来说，使用 gt高的teacher比使用loc高的teacher效果更好
                              "save_model/cub_bz_6_bkB_inceptionV3_bkR_0.6_dcE_50_dcR_0.5_func_quadratic_mN_20_sgT_0.3_CipS_350_CT_15.0_LipS_350_LT_15.0_kd_3_kdA_0.0_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0001_dn_0_uR_0.01_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_randAug_5_clsTea_3_locTea_2_dist_none_atten_1/2022-08-23_02_03_50/net_train_inceptionV3_cub_85.pth"]  # 106 也是我们的student， 350, segthr=0.3, cls-1 78.81%, cls-5 94.98%, loc-1 63.77%, loc-5 76.54%, gt 80.19% todo 实验证明 对于 loc teacher 来说，使用 gt高的teacher比使用loc高的teacher效果更好

    cls_teacher_path = total_cls_teacher_path[args.clsTea]
    loc_teacher_path = total_loc_teacher_path[args.locTea]

if args.dataset == "imagenet" and (args.backbone == "vgg16" or args.backbone == "vgg16_baseline"):
    total_cls_teacher_path = ["../KD-CI-CAM-teacher-github/weight_0702/imagenet_sh_1_bz_42_ipS_240_cpS_224_atS_14_bkB_vgg16_bkR_1.0_func_sum_1_mN_20_lr_2.5e-06_epc_10_dn_0_dcE_6_dcR_0.1_sgT_0.2_auxR_1.0_dcMn_100_incR_0.1_uR_0.001_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.4_maxSca_1.0_spa_1_spaR_0.04_auxType_fore/2022-06-26_10_29_18/net_train_vgg16_imagenet_9.pth",  # cls-1 73.03%, cls-5 91.07%, loc-1 45.94%, loc-5 55.08%, gt 58.24%  2022-06-26_10_29_18. 用了 net_train_vgg16_imagenet_0.pth
                              "../KD-CI-CAM-teacher-github/weight_0702/imagenet_sh_1_bz_36_ipS_240_cpS_224_atS_14_bkB_vgg16_bkR_1.0_func_sum_1_mN_20_lr_5.85e-05_epc_20_dn_0_dcE_10_dcR_0.1_sgT_0.1_auxR_1.0_dcMn_100_incR_0.1_uR_0.001_sgd_0_newB_0_scale_1_minSca_0.4_maxSca_1.0_spa_1_spaR_0.04_auxType_fore/2022-08-24_12_20_42/net_train_vgg16_imagenet_20.pth"]  # cls-1 73.46%, cls-5 91.43%, loc-1 34.14%, loc-5 41.32%, gt 43.91%, ep 20  2022-08-24_12_20_42, 不用 net_train_vgg16_imagenet_0.pth.  104与106已同步
    total_loc_teacher_path = ["../CI-CAM-KD/new_best_weight/imagenet_sh_1_bz_96_ipS_240_cpS_224_atS_14_bkB_vgg16_bkR_1.0_func_sum_1_mN_20_lr_5.8e-06_epc_10_dn_0_dcE_3_dcR_0.1_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1_uR_0.001_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.9_maxSca_1.1_spa_2_spaR_1e-07_auxType_back/2022-06-24_21_04_09/net_train_vgg16_imagenet_1.pth",  # gt segthr=0.15 62.76%, segthr=0.0 62.34%
                              "save_model/imagenet_bz_84_bkB_vgg16_bkR_1.0_dcE_3_dcR_0.1_func_sum_1_mN_20_sgT_0.15_CipS_240_CT_5.0_LipS_240_LT_5.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_4e-06_dn_0_uR_0.001_Cmin_0.4_Cmax_1.0_Lmin_0.9_Lmax_1.1_randAug_0_clsTea_0_locTea_0/2022-07-16_00_17_57/net_train_vgg16_imagenet_2.pth",  # gt segthr=0.15 63.12%, segthr=0.0 62.75%
                              "../KD-CI-CAM-teacher-github/weight_0702/imagenet_sh_1_bz_36_ipS_240_cpS_224_atS_14_bkB_vgg16_bkR_1.0_func_sum_1_mN_20_lr_6e-05_epc_20_dn_0_dcE_10_dcR_0.1_sgT_0.0_auxR_1.0_dcMn_100_incR_0.1_uR_0.001_sgd_0_newB_0_scale_0_minSca_0.9_maxSca_1.1_spa_1_spaR_1e-06_auxType_diff_aux/2022-07-10_02_03_44//net_train_vgg16_imagenet_5.pth",  # gt segthr=0.0 62.48%  | segthr=0.0, cls-1 63.05%, cls-5 85.32%, loc-1 42.85%, loc-5 56.06%, gt 62.48%
                              "../KD-CI-CAM-teacher-github/weight_0702/net_train_vgg16_imagenet_0.pth"]  # segthr=0.0, cls-1 72.62%, cls-5 90.93%, loc-1 48.71%, loc-5 58.73%, gt 62.36%
    cls_teacher_path = total_cls_teacher_path[args.clsTea]
    loc_teacher_path = total_loc_teacher_path[args.locTea]

