# vgg16 cub
# using cls_teacher + gt_known loc_teacher
python -u train_student_randomresizecrop.py --gpu 1 --update_rate 0.002 --distillation random --randAug 1 --clsTea 5 --locTea 2 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.0 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --cls_teacher_input_size 304 --loc_teacher_input_size 304 >> log/train_kd_vgg_kd_1.txt 2>&1 &
# using cls_teacher + top-1 loc_teacher
python -u train_student_randomresizecrop.py --gpu 2 --randAug 1 --clsTea 5 --locTea 7 --kd_mode 3 --kd_fe_loss 1 --kd_fe_zero 1 --kd_fe_times 1.0 --kd_alpha 0.8 --cls_teacher_T 15 --loc_teacher_T 15 --seg_thr 0.0 --decay_epoch 40 --backbone vgg16 --backbone_rate 0.1 --dataset cub --lr 0.0005 --cls_teacher_input_size 304 --loc_teacher_input_size 304 >> log/train_kd_vgg_kd_2.txt 2>&1 &
