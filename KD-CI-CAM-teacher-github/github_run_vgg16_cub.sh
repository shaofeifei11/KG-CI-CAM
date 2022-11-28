################################ train  VGG16 CUB
# todo table 5 VGG16 CUB cls teacher 104
python -u train_0702.py --gpu 0 --dataset cub --shared_classifier 1 --batch_size 6 --input_size 344 --crop_size 224 --attention_size 14 --backbone vgg16 --backbone_rate 0.1 --function quadratic --mean_num 20 --lr 0.0005 --epoch 100 --danet 1 --decay_epoch 14 --decay_rate 0.5 --seg_thr 0.08 --aux_rate 1.0 --max_num 100 --inception_reduce 0.1 --update_rate 0.01 --scale 1 --min_scale 0.2 --max_scale 1.0 --spa_loss 2 --spa_loss_rate 0.04 --aux_type fore > log/train_vgg16_1.txt 2>&1 &

# todo table 5 VGG16 CUB gt loc teacher 104
python -u train_0702.py --gpu 1 --dataset cub --shared_classifier 1 --batch_size 6 --input_size 344 --crop_size 224 --attention_size 14 --backbone vgg16 --backbone_rate 0.1 --function quadratic --mean_num 20 --decay_rate 0.5 --lr 0.0005 --epoch 100 --danet 0 --decay_epoch 14 --seg_thr 0.0 --aux_rate 0.5 --max_num 100 --inception_reduce 0.1 > log/train_vgg16_2.txt 2>&1 &

# todo table 5 VGG16 CUB top-1 loc teacher 104
python -u train_0702.py --gpu 2 --dataset cub --shared_classifier 1 --batch_size 6 --input_size 344 --crop_size 224 --attention_size 14 --backbone vgg16 --backbone_rate 0.1 --function quadratic --mean_num 20 --lr 0.0005 --epoch 100 --danet 0 --decay_epoch 14 --decay_rate 0.5 --seg_thr 0.0 --aux_rate 0.5 --max_num 100 --inception_reduce 0.1 --update_rate 0.01 --scale 1 --min_scale 0.1 --max_scale 1.0 --spa_loss 0 --spa_loss_rate 0.04 --aux_type diff_aux > log/train_vgg16_3.txt 2>&1 &



