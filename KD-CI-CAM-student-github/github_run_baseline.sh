## cub inceptionV3 104
python -u baseline_train.py --gpu 0 --attention 0 --distillation none --clsTea 3 --locTea 2 --randAug 5 --cls_min_scale 0.2 --loc_min_scale 0.1 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.3 --decay_epoch 100 --backbone inceptionV3_baseline --backbone_rate 0.6 --dataset cub --lr 0.0001 --cls_teacher_input_size 350 --loc_teacher_input_size 350 >> log/train_baseline_incep_cub.txt 2>&1 &

## cub vgg16 104
python -u baseline_train.py --gpu 0 --attention 0 --distillation none --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.0 --decay_epoch 100 --backbone vgg16_baseline --backbone_rate 0.1 --dataset cub --lr 0.0005 --cls_teacher_input_size 344 --loc_teacher_input_size 344 >> log/train_baseline_vgg16_cub.txt 2>&1 &

## imagenet vgg16 106  --decay_epoch 10
python -u baseline_train_distri.py --gpu 1,2,3 --attention 0 --distillation none --locTea 2 --cls_min_scale 0.4 --cls_max_scale 1.0 --loc_min_scale 0.9 --loc_max_scale 1.1 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.0 --cls_teacher_T 2 --loc_teacher_T 2 --danet 0 --batch_size 100 --lr 0.0001625 --epoch 20 --decay_epoch 10 --decay_rate 0.1 --backbone_rate 1.0 --update_rate 0.001 --backbone vgg16_baseline --dataset imagenet --function sum_1 --mean_num 20 --seg_thr 0.11 --cls_teacher_input_size 288 --loc_teacher_input_size 288 >> log/train_baseline_vgg16_imagenet.txt 2>&1

