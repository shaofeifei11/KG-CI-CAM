# todo table 1 CUB VGG16 NL-CCAM 104
python -u baseline_test_center_crop.py --gpu 0 --model_path save_model/cub_bz_6_bkB_vgg16_baseline_bkR_0.1_dcE_100_dcR_0.5_func_quadratic_mN_20_sgT_0.0_CipS_344_CT_15.0_LipS_344_LT_15.0_kd_3_kdA_0.0_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0005_dn_0_uR_0.01_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_rAug_1_cTea_5_lTea_2_dist_none_att_0/2022-11-09_21_35_57/net_train_vgg16_baseline_cub_100.pth --attention 0 --distillation none --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.14 --decay_epoch 100 --backbone vgg16_baseline --backbone_rate 0.1 --dataset cub --lr 0.0005 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/train_baseline_vgg16_cub_1_segThr_0.14.txt 2>&1 &
#classification top 1 accuracy:  0.7474974111149465
#classification top 5 accuracy:  0.9263030721435969
#localization top 1 accuracy:  0.6553331032102174
#localization top 5 accuracy:  0.8111839834311356
#gt localization accuracy:  0.8745253710735243

# todo table 1 vgg16 CI-CAM 104
python -u test_center_crop.py --gpu 0 --model_path save_model/cub_bz_6_bkB_vgg16_bkR_0.1_dcE_40_dcR_0.5_func_quadratic_mN_20_sgT_0.0_CipS_344_CT_15.0_LipS_344_LT_15.0_kd_3_kdA_0.0_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0005_dn_0_uR_0.01_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_rAug_1_cTea_5_lTea_2_dist_none_att_1/2022-11-10_03_36_06/net_train_vgg16_cub_100.pth  --distillation none --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.14 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/train_vgg16_cub_again_3_segThr_0.14.txt 2>&1 &
#classification top 1 accuracy:  0.7514670348636521
#classification top 5 accuracy:  0.9301001035554022
#localization top 1 accuracy:  0.6653434587504314
#localization top 5 accuracy:  0.8244735933724543
#gt localization accuracy:  0.8809112875388333

# todo table 1 vgg16 KD-CI-CAM 104
python -u test_center_crop.py --gpu 1 --model_path save_model/cub_bz_6_bkB_vgg16_bkR_0.1_dcE_40_dcR_0.5_func_quadratic_mN_20_sgT_0.0_CipS_304_CT_15.0_LipS_304_LT_15.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0005_dn_0_uR_0.002_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_randAug_1_clsTea_5_locTea_2_dist_random_atten_1/2022-09-05_13_24_17/net_train_vgg16_cub_100.pth --update_rate 0.002 --distillation random --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.14 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/figure_1_cub_vgg16_kd_14_segThr_0.14.txt 2>&1 &  # todo 定位最高
#classification top 1 accuracy:  0.7923714187090093
#classification top 5 accuracy:  0.9409734207801174
#localization top 1 accuracy:  0.7302381774249224  # todo 这个 VGG16最高
#localization top 5 accuracy:  0.8653779772178115
#gt localization accuracy:  0.9159475319295823  # todo 这个 VGG16最高

# todo table 4 VGG16 CUB stu+cls 106
python -u test_center_crop.py --gpu 1 --model_path save_model/cub_bz_6_bkB_vgg16_bkR_0.1_dcE_40_dcR_0.5_func_quadratic_mN_20_sgT_0.0_CipS_304_CT_15.0_LipS_304_LT_15.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0005_dn_0_uR_0.01_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_randAug_1_clsTea_5_locTea_2_dist_cls/2022-08-07_16_09_20/net_train_vgg16_cub_100.pth --distillation cls --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --seg_thr 0.14 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/table_4_cub_vgg16_stu_cls_segThr_0.14.txt 2>&1 &
#classification top 1 accuracy:  0.774249223334484
#classification top 5 accuracy:  0.9389023127373145
#localization top 1 accuracy:  0.6651708664135313
#localization top 5 accuracy:  0.8016914049016224
#gt localization accuracy:  0.8501898515705902

# todo table 4 VGG16 CUB stu+loc 106
python -u test_center_crop.py --gpu 2 --model_path save_model/cub_bz_6_bkB_vgg16_bkR_0.1_dcE_40_dcR_0.5_func_quadratic_mN_20_sgT_0.0_CipS_304_CT_15.0_LipS_304_LT_15.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0005_dn_0_uR_0.01_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_randAug_1_clsTea_5_locTea_2_dist_loc/2022-08-07_16_09_20/net_train_vgg16_cub_100.pth --distillation loc --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --seg_thr 0.14 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/table_4_cub_vgg16_stu_loc_segThr_0.14.txt 2>&1 &
#classification top 1 accuracy:  0.7808077321366931
#classification top 5 accuracy:  0.9439074905074215
#localization top 1 accuracy:  0.7176389368312047
#localization top 5 accuracy:  0.8655505695547118
#gt localization accuracy:  0.9145667932343804

# todo table 6 VGG16 CUB cls teacher 104
python -u test_center_crop.py --gpu 2 --model_path ../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_304_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_1_dcE_14_dcR_0.5_sgT_0.08_auxR_1.0_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.2_maxSca_1.0_spa_2_spaR_0.04_auxType_fore/2022-07-03_00_43_26/net_train_vgg16_cub_100.pth --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --seg_thr 0.14 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/table_6_cub_vgg16_cls_segThr_0.14.txt 2>&1 &
#classification top 1 accuracy:  0.7928891957197101
#classification top 5 accuracy:  0.942181567138419
#localization top 1 accuracy:  0.6259924059371764
#localization top 5 accuracy:  0.7423196410079392
#gt localization accuracy:  0.7856403175698999

# todo table 6 VGG16 CUB gt loc teacher 104
python -u test_center_crop.py --gpu 3 --model_path ../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_336_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_dcR_0.5_lr_0.0005_epc_100_dn_0_dcE_14_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1/2022-03-22_20_56_25/net_train_vgg16_cub_70.pth --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --seg_thr 0.14 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/table_6_cub_vgg16_gt_loc_segThr_0.14.txt 2>&1 &
#classification top 1 accuracy:  0.7331722471522264
#classification top 5 accuracy:  0.9200897480151882
#localization top 1 accuracy:  0.6584397652744218
#localization top 5 accuracy:  0.8225750776665516
#gt localization accuracy:  0.8876423886779427

# todo table 6 VGG16 CUB top-1 loc teacher 104
python -u test_center_crop.py --gpu 3 --model_path ../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_336_cpS_224_atS_14_bkB_vgg16_bkR_0.1_func_quadratic_mN_20_lr_0.0005_epc_100_dn_0_dcE_14_dcR_0.5_sgT_0.0_auxR_0.5_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_scale_1_minSca_0.1_maxSca_1.0_spa_0_spaR_0.04_auxType_diff_aux/2022-07-08_07_15_08/net_train_vgg16_cub_85.pth --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --seg_thr 0.14 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/table_6_cub_vgg16_top-1_loc_segThr_0.14.txt 2>&1 &
#classification top 1 accuracy:  0.7702795995857784
#classification top 5 accuracy:  0.9326889886089058
#localization top 1 accuracy:  0.6831204694511563
#localization top 5 accuracy:  0.8225750776665516
#gt localization accuracy:  0.8788401794960303

# todo table 6 VGG16 CUB cls + top-1 loc teacher 104
python -u test_center_crop.py --gpu 0 --model_path save_model/cub_bz_6_bkB_vgg16_bkR_0.1_dcE_40_dcR_0.5_func_quadratic_mN_20_sgT_0.0_CipS_304_CT_15.0_LipS_304_LT_15.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0005_dn_0_uR_0.01_CminSca_0.2_LminSca_0.1_randAug_1_clsTea_5_locTea_7/2022-07-08_16_45_44/net_train_vgg16_cub_90.pth --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --seg_thr 0.14 --cls_teacher_input_size 344 --loc_teacher_input_size 344 > log/table_6_cub_vgg16_cls_top-1_loc_segThr_0.14.txt 2>&1 &
#classification top 1 accuracy:  0.7830514325163963
#classification top 5 accuracy:  0.9390749050742146
#localization top 1 accuracy:  0.7148774594408008
#localization top 5 accuracy:  0.8529513289609941
#gt localization accuracy:  0.9047290300310666
