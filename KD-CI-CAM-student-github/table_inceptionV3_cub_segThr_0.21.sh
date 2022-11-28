## todo table 1 inceptionV3 NL-CCAM 104
python -u baseline_test_center_crop.py --gpu 0 --model_path save_model/cub_bz_6_bkB_inceptionV3_baseline_bkR_0.6_dcE_100_dcR_0.5_func_quadratic_mN_20_sgT_0.3_CipS_350_CT_15.0_LipS_350_LT_15.0_kd_3_kdA_0.0_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0001_dn_0_uR_0.01_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_rAug_5_cTea_3_lTea_2_dist_none_att_0/2022-11-09_15_27_41/net_train_inceptionV3_baseline_cub_100.pth --attention 0 --distillation none --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.21 --decay_epoch 100 --backbone inceptionV3_baseline --backbone_rate 0.6 --dataset cub --lr 0.0001 --cls_teacher_input_size 350 --loc_teacher_input_size 350 > log/baseline_table_3_cub_incep_nl-ccam_segThr_0.21.txt 2>&1 &
#classification top 1 accuracy:  0.7676907145322748
#classification top 5 accuracy:  0.9382119433897135
#localization top 1 accuracy:  0.6282361063168795
#localization top 5 accuracy:  0.759233690024163
#gt localization accuracy:  0.8020365895754229

# todo table 1 inceptionV3 CI-CAM 106   6
python -u test_center_crop.py --gpu 0 --model_path save_model/cub_bz_6_bkB_inceptionV3_bkR_0.6_dcE_50_dcR_0.5_func_quadratic_mN_20_sgT_0.3_CipS_350_CT_15.0_LipS_350_LT_15.0_kd_3_kdA_0.0_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0001_dn_0_uR_0.02_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_randAug_5_clsTea_3_locTea_2_dist_none_atten_1/2022-09-21_01_01_29/net_train_inceptionV3_cub_90.pth --attention 1 --distillation none --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 50 --backbone inceptionV3 --backbone_rate 0.6 --dataset cub --lr 0.0001 --seg_thr 0.21 --cls_teacher_input_size 500 --loc_teacher_input_size 500 > log/table_1_cub_incep_ci-cam_segThr_0.21_6.txt 2>&1 &
#classification top 1 accuracy:  0.7865032792544011
#classification top 5 accuracy:  0.9458060062133241
#localization top 1 accuracy:  0.7385226095961339
#localization top 5 accuracy:  0.8878149810148429
#gt localization accuracy:  0.937866758715913

# todo table 1 inceptionV3 KD-CI-CAM 104
python -u test_center_crop.py --gpu 1 --model_path save_model/cub_bz_6_bkB_inceptionV3_bkR_0.6_dcE_50_dcR_0.5_func_quadratic_mN_20_sgT_0.3_CipS_350_CT_15.0_LipS_350_LT_15.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0001_dn_0_uR_0.015_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_rAug_5_cTea_3_lTea_2_dist_random_att_1/2022-09-21_21_25_33/net_train_inceptionV3_cub_100.pth --update_rate 0.015 --attention 1 --distillation random --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.21 --decay_epoch 50 --backbone inceptionV3 --backbone_rate 0.6 --dataset cub --lr 0.0001 --cls_teacher_input_size 500 --loc_teacher_input_size 500 > log/figure_1_cub_incep_kd_5_segThr_0.21_segThr_0.21.txt 2>&1 &  # todo 定位最高
#classification top 1 accuracy:  0.797204004142216
#classification top 5 accuracy:  0.9527096996893338
#localization top 1 accuracy:  0.7626855367621678
#localization top 5 accuracy:  0.9097342078011736
#gt localization accuracy:  0.9534000690369347

# todo table 4 InceptionV3 CUB stu+cls 106
python -u test_center_crop.py --gpu 1 --model_path save_model/cub_bz_6_bkB_inceptionV3_bkR_0.6_dcE_50_dcR_0.5_func_quadratic_mN_20_sgT_0.3_CipS_350_CT_15.0_LipS_350_LT_15.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0001_dn_0_uR_0.01_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_randAug_5_clsTea_3_locTea_2_dist_cls/2022-08-07_06_57_41/net_train_inceptionV3_cub_90.pth --distillation cls --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 50 --backbone inceptionV3 --backbone_rate 0.6 --dataset cub --lr 0.0001 --seg_thr 0.21 --cls_teacher_input_size 500 --loc_teacher_input_size 500 > log/table_4_cub_incep_stu_cls_segThr_0.21.txt 2>&1 &
#classification top 1 accuracy:  0.7984121505005177
#classification top 5 accuracy:  0.9509837763203314
#localization top 1 accuracy:  0.7430100103555403
#localization top 5 accuracy:  0.8835001725923369
#gt localization accuracy:  0.9266482568173973

## todo table 4 InceptionV3 CUB stu+loc 104
python -u test_center_crop.py --gpu 2 --model_path save_model/cub_bz_6_bkB_inceptionV3_bkR_0.6_dcE_50_dcR_0.5_func_quadratic_mN_20_sgT_0.3_CipS_350_CT_15.0_LipS_350_LT_15.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0001_dn_0_uR_0.01_Cmin_0.2_Cmax_1.0_Lmin_0.1_Lmax_1.0_randAug_5_clsTea_3_locTea_2_dist_loc/2022-08-04_01_50_25/net_train_inceptionV3_cub_100.pth --distillation loc --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 50 --backbone inceptionV3 --backbone_rate 0.6 --dataset cub --lr 0.0001 --seg_thr 0.21 --cls_teacher_input_size 500 --loc_teacher_input_size 500 > log/table_4_cub_incep_stu_loc_segThr_0.21.txt 2>&1 &
#classification top 1 accuracy:  0.7856403175698999
#classification top 5 accuracy:  0.9471867449085261
#localization top 1 accuracy:  0.749223334483949
#localization top 5 accuracy:  0.9005868139454608
#gt localization accuracy:  0.9490852606144288

# todo table 6 InceptionV3 CUB cls teacher 106
python -u test_center_crop.py --gpu 2 --model_path ../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_350_cpS_299_atS_17_bkB_inceptionV3_bkR_0.6_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_20_dcR_0.5_sgT_0.15_auxR_0.6_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.2_maxSca_1.0_spa_0_spaR_0.04_auxType_fore/2022-07-04_04_57_03/net_train_inceptionV3_cub_90.pth --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 50 --backbone inceptionV3 --backbone_rate 0.6 --dataset cub --lr 0.0001 --seg_thr 0.21 --cls_teacher_input_size 500 --loc_teacher_input_size 500 > log/table_6_cub_incep_cls_segThr_0.21.txt 2>&1 &
#classification top 1 accuracy:  0.7940973420780117
#classification top 5 accuracy:  0.9480497065930272
#localization top 1 accuracy:  0.6532619951674146
#localization top 5 accuracy:  0.7799447704521919
#gt localization accuracy:  0.8205039696237487

# todo table 6 InceptionV3 CUB gt loc teacher 106
python -u test_center_crop.py --gpu 3 --model_path ../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_500_cpS_299_atS_17_bkB_inceptionV3_bkR_0.8_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_50_dcR_0.5_sgT_0.25_auxR_0.2_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_scale_3_minSca_0.1_maxSca_1.0_spa_2_spaR_2e-08_auxType_diff_aux/2022-07-05_12_34_45/net_train_inceptionV3_cub_65.pth --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 50 --backbone inceptionV3 --backbone_rate 0.6 --dataset cub --lr 0.0001 --seg_thr 0.21 --cls_teacher_input_size 500 --loc_teacher_input_size 500 > log/table_6_cub_incep_gt_loc_segThr_0.21.txt 2>&1 &
#classification top 1 accuracy:  0.7210907835692095
#classification top 5 accuracy:  0.9219882637210908
#localization top 1 accuracy:  0.6819123230928547
#localization top 5 accuracy:  0.870728339661719
#gt localization accuracy:  0.9414911977908181

# todo table 6 InceptionV3 CUB top-1 loc teacher 3 106
python -u test_center_crop.py --gpu 3 --model_path ../KD-CI-CAM-teacher-github/weight_0702/cub_sh_1_bz_6_ipS_500_cpS_299_atS_17_bkB_inceptionV3_bkR_0.8_func_quadratic_mN_20_lr_0.0001_epc_100_dn_0_dcE_50_dcR_0.5_sgT_0.25_auxR_0.2_dcMn_100_incR_0.1_uR_0.01_sgd_0_newB_0_scale_3_minSca_0.1_maxSca_1.0_spa_2_spaR_2e-08_auxType_diff_aux/2022-07-05_12_34_45/net_train_inceptionV3_cub_60.pth --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 50 --backbone inceptionV3 --backbone_rate 0.6 --dataset cub --lr 0.0001 --seg_thr 0.21 --cls_teacher_input_size 500 --loc_teacher_input_size 500 > log/table_6_cub_incep_top-1_loc_segThr_0.21.txt 2>&1 &
#classification top 1 accuracy:  0.7347255781843286
#classification top 5 accuracy:  0.9269934414911978
#localization top 1 accuracy:  0.6858819468415602
#localization top 5 accuracy:  0.8631342768381084
#gt localization accuracy:  0.9292371418709009

# todo InceptionV3 CUB cls + top-1 loc teacher 3 106
python -u test_center_crop.py --gpu 0 --model_path save_model/cub_bz_6_bkB_inceptionV3_bkR_0.6_dcE_50_dcR_0.5_func_quadratic_mN_20_sgT_0.3_CipS_350_CT_15.0_LipS_350_LT_15.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.0001_dn_0_uR_0.01_CminSca_0.2_LminSca_0.1_randAug_5_clsTea_3_locTea_3/2022-07-06_00_43_30/net_train_inceptionV3_cub_90.pth --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --decay_epoch 50 --backbone inceptionV3 --backbone_rate 0.6 --dataset cub --lr 0.0001 --seg_thr 0.21 --cls_teacher_input_size 500 --loc_teacher_input_size 500 > log/table_6_cub_incep_cls_top-1_lo_segThr_0.21c.txt 2>&1 &
#classification top 1 accuracy:  0.7911632723507076
#classification top 5 accuracy:  0.9468415602347255
#localization top 1 accuracy:  0.7504314808422506
#localization top 5 accuracy:  0.8955816361753538
#gt localization accuracy:  0.9447704521919227
