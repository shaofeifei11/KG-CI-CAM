# todo table 2 Imagenet VGG16 NL-CCAM 106  --decay_epoch 10
#python -u baseline_train_distri.py --gpu 0,1,2 --attention 0 --distillation none --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 100 --lr 0.0001625 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16_baseline --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/train_baseline_vgg16_imagenet.txt 2>&1
#classification top 1 accuracy:  0.7227
#classification top 5 accuracy:  0.90448
#localization top 1 accuracy:  0.48586
#localization top 5 accuracy:  0.58756
#gt localization accuracy:  0.62882

# todo table 2 vgg16 CI-CAM 106
python -u test_center_crop.py --gpu 0 --model_path save_model/imagenet_bz_80_bkB_vgg16_bkR_1.0_dcE_10_dcR_0.1_func_sum_1_mN_20_sgT_0.1_CipS_240_CT_2.0_LipS_240_LT_2.0_kd_3_kdA_0.0_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.00013_dn_0_uR_0.001_Cmin_0.4_Cmax_1.0_Lmin_0.9_Lmax_1.1_randAug_0_clsTea_0_locTea_2_dist_none_atten_1/2022-08-23_06_06_14/net_train_vgg16_imagenet_19.pth --attention 1 --distillation none --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 80 --lr 0.00013 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/table_2_imagenet_vgg16_ci-cam.txt 2>&1 &
#classification top 1 accuracy:  0.72172
#classification top 5 accuracy:  0.90428
#localization top 1 accuracy:  0.49038
#localization top 5 accuracy:  0.59404
#gt localization accuracy:  0.63636

# todo table 2 vgg16 KD-CI-CAM 102
python -u test_center_crop.py --gpu 1 --model_path save_model/imagenet_78_vgg16_1.0_EP_20_dcE_10_dcR_0.1_func_sum_1_mN_20_sgT_0.11_CipS_288_CT_1.1_LipS_288_LT_1.1_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_8e-05_dn_0_uR_0.001_Cmin_0.4_Cmax_1.0_Lmin_0.9_Lmax_1.1_rAug_0_cTea_0_lTea_2_dist_random_att_1_LG_1_IPC_0.1/2023-09-25_00_16_42/net_train_vgg16_imagenet_16.pth --danet 0 --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 1.1 --loc_teacher_T 1.1 --danet 0 --batch_size 78 --lr 0.00008 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288  >> log/test_vgg16_imagenet_288.txt 2>&1 &
# seg_thr_i=  0.11 288
Inference Results:
classification top 1 accuracy:  0.72358
classification top 5 accuracy:  0.9032
localization top 1 accuracy:  0.51354
localization top 5 accuracy:  0.62276
gt localization accuracy:  0.67392

# todo table 5 VGG16 Imagenet stu+cls 104
python -u test_center_crop.py --gpu 1 --model_path save_model/imagenet_bz_80_bkB_vgg16_bkR_1.0_dcE_10_dcR_0.1_func_sum_1_mN_20_sgT_0.1_CipS_240_CT_2.0_LipS_240_LT_2.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_0.00013_dn_0_uR_0.001_Cmin_0.4_Cmax_1.0_Lmin_0.9_Lmax_1.1_randAug_0_clsTea_1_locTea_2_dist_cls_atten_1/2022-09-06_20_49_46/net_train_vgg16_imagenet_20.pth --distillation cls --clsTea 1 --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 80 --lr 0.00013 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/table_5_imagenet_vgg16_stu_cls.txt 2>&1 &
#classification top 1 accuracy:  0.722
#classification top 5 accuracy:  0.90492
#localization top 1 accuracy:  0.49822
#localization top 5 accuracy:  0.60374
#gt localization accuracy:  0.64546

# todo table 5 VGG16 Imagenet stu+loc 106
python -u test_center_crop.py --gpu 1 --model_path save_model/imagenet_bz_36_bkB_vgg16_bkR_1.0_dcE_10_dcR_0.1_func_sum_1_mN_20_sgT_0.1_CipS_240_CT_2.0_LipS_240_LT_2.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_5.85e-05_dn_0_uR_0.001_Cmin_0.4_Cmax_1.0_Lmin_0.9_Lmax_1.1_randAug_0_clsTea_0_locTea_2_dist_loc_atten_1/2022-09-01_16_11_51/net_train_vgg16_imagenet_13.pth --distillation loc --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 36 --lr 0.0000585 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/table_5_imagenet_vgg16_stu_loc.txt 2>&1 &
#classification top 1 accuracy:  0.7153
#classification top 5 accuracy:  0.90148
#localization top 1 accuracy:  0.49938
#localization top 5 accuracy:  0.61124
#gt localization accuracy:  0.65702

# todo table 6 VGG16 Imagenet cls teacher 104  (old cls)
python -u test_center_crop.py --gpu 2 --model_path ../KD-CI-CAM-teacher-github/weight_0702/imagenet_sh_1_bz_42_ipS_240_cpS_224_atS_14_bkB_vgg16_bkR_1.0_func_sum_1_mN_20_lr_2.5e-06_epc_10_dn_0_dcE_6_dcR_0.1_sgT_0.2_auxR_1.0_dcMn_100_incR_0.1_uR_0.001_sgd_0_newB_0_CMixP_0.0_CMix_1_scale_1_minSca_0.4_maxSca_1.0_spa_1_spaR_0.04_auxType_fore/2022-06-26_10_29_18/net_train_vgg16_imagenet_9.pth --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 48 --lr 0.000078 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/table_6_imagenet_vgg16_cls.txt 2>&1 &
#classification top 1 accuracy:  0.717
#classification top 5 accuracy:  0.90234
#localization top 1 accuracy:  0.4639
#localization top 5 accuracy:  0.56254
#gt localization accuracy:  0.6018

# todo table 6 VGG16 Imagenet gt loc teacher 104
python -u test_center_crop.py --gpu 2 --model_path ../KD-CI-CAM-teacher-github/weight_0702/imagenet_sh_1_bz_36_ipS_240_cpS_224_atS_14_bkB_vgg16_bkR_1.0_func_sum_1_mN_20_lr_6e-05_epc_20_dn_0_dcE_10_dcR_0.1_sgT_0.0_auxR_1.0_dcMn_100_incR_0.1_uR_0.001_sgd_0_newB_0_scale_0_minSca_0.9_maxSca_1.1_spa_1_spaR_1e-06_auxType_diff_aux/2022-07-10_02_03_44/net_train_vgg16_imagenet_5.pth --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 48 --lr 0.000078 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/table_6_imagenet_vgg16_top-1_loc.txt 2>&1 &
#classification top 1 accuracy:  0.60542
#classification top 5 accuracy:  0.83352
#localization top 1 accuracy:  0.42228
#localization top 5 accuracy:  0.56474
#gt localization accuracy:  0.64676

# todo table 6 VGG16 Imagenet top-1 loc teacher 104
python -u test_center_crop.py --gpu 3 --model_path ../KD-CI-CAM-teacher-github/weight_0702/net_train_vgg16_imagenet_0.pth --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 48 --lr 0.000078 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/table_6_imagenet_vgg16_gt_loc.txt 2>&1 &
#classification top 1 accuracy:  0.7144
#classification top 5 accuracy:  0.90086
#localization top 1 accuracy:  0.49212
#localization top 5 accuracy:  0.60178
#gt localization accuracy:  0.64634

# todo table 6 VGG16 Imagenet cls + top-1 loc teacher 104
python -u test_center_crop.py --gpu 3 --model_path save_model/imagenet_bz_48_bkB_vgg16_bkR_1.0_dcE_10_dcR_0.1_func_sum_1_mN_20_sgT_0.1_CipS_240_CT_2.0_LipS_240_LT_2.0_kd_3_kdA_0.8_kdFLs_1_kdFTs_1.0_kdFZo_1_lr_7.8e-05_dn_0_uR_0.001_Cmin_0.4_Cmax_1.0_Lmin_0.9_Lmax_1.1_randAug_0_clsTea_0_locTea_3_dist_random/2022-08-14_08_25_55/net_train_vgg16_imagenet_20.pth --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 48 --lr 0.000078 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/table_6_imagenet_vgg16_cls_gt_loc.txt 2>&1 &
#classification top 1 accuracy:  0.72644
#classification top 5 accuracy:  0.90732
#localization top 1 accuracy:  0.50386
#localization top 5 accuracy:  0.61164
#gt localization accuracy:  0.65348


# todo table 6 VGG16 Imagenet cls teacher 104  (new cls)
python -u test_center_crop.py --gpu 0 --model_path ../KD-CI-CAM-teacher-github/weight_0702/imagenet_sh_1_bz_36_ipS_240_cpS_224_atS_14_bkB_vgg16_bkR_1.0_func_sum_1_mN_20_lr_5.85e-05_epc_20_dn_0_dcE_10_dcR_0.1_sgT_0.1_auxR_1.0_dcMn_100_incR_0.1_uR_0.001_sgd_0_newB_0_scale_1_minSca_0.4_maxSca_1.0_spa_1_spaR_0.04_auxType_fore/2022-08-24_12_20_42/net_train_vgg16_imagenet_20.pth --clsTea 1 --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 48 --lr 0.000078 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16 --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/table_6_imagenet_vgg16_cls.txt 2>&1 &
#classification top 1 accuracy:  0.72178
#classification top 5 accuracy:  0.90514
#localization top 1 accuracy:  0.38768
#localization top 5 accuracy:  0.47544
#gt localization accuracy:  0.51088
