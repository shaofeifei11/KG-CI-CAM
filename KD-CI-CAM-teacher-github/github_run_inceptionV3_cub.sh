################################ train  InceptionV3 CUB
# todo table 5 InceptionV3 CUB cls teacher 106
python -u train_0702.py --gpu 2 --dataset cub --shared_classifier 1 --batch_size 6 --input_size 500 --crop_size 299 --attention_size 17 --backbone inceptionV3 --backbone_rate 0.6 --function quadratic --mean_num 20 --lr 0.0001 --epoch 100 --danet 0 --decay_epoch 20 --decay_rate 0.5 --seg_thr 0.15 --aux_rate 0.6 --max_num 100 --inception_reduce 0.1 --update_rate 0.01 --scale 1 --min_scale 0.2 --max_scale 1.0 --spa_loss 0 --spa_loss_rate 0.04 --aux_type fore > log/train_inceptionV3_1.txt 2>&1 &

# todo table 5 InceptionV3 CUB gt loc teacher 106
python -u train_0702.py --gpu 0 --dataset cub --shared_classifier 1 --batch_size 6 --input_size 500 --crop_size 299 --attention_size 17 --backbone inceptionV3 --backbone_rate 0.8 --function quadratic --mean_num 20 --lr 0.0001 --epoch 100 --danet 0 --decay_epoch 50 --decay_rate 0.5 --seg_thr 0.25 --aux_rate 0.2 --max_num 100 --inception_reduce 0.1 --update_rate 0.01 --scale 3 --min_scale 0.1 --max_scale 1.0 --spa_loss 2 --spa_loss_rate 2e-08 --aux_type diff_aux > log/train_inceptionV3_2.txt 2>&1 &

# todo table 5 InceptionV3 CUB top-1 loc teacher 3 106
python -u train_0702.py --gpu 1 --dataset cub --shared_classifier 1 --batch_size 6 --input_size 500 --crop_size 299 --attention_size 17 --backbone inceptionV3 --backbone_rate 0.8 --function quadratic --mean_num 20 --lr 0.0001 --epoch 100 --danet 0 --decay_epoch 50 --decay_rate 0.5 --seg_thr 0.25 --aux_rate 0.2 --max_num 100 --inception_reduce 0.1 --update_rate 0.01 --scale 3 --min_scale 0.1 --max_scale 1.0 --spa_loss 2 --spa_loss_rate 2e-08 --aux_type diff_aux > log/train_inceptionV3_3.txt 2>&1 &

