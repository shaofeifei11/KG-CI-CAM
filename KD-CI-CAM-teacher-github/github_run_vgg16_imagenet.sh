
################################ train VGG16 Imagenet
# todo table 5 VGG16 Imagenet cls teacher 104  (old cls)
python -u train_distri.py --gpu 3,2,1 --dataset imagenet --shared_classifier 1 --batch_size 42 --input_size 240 --crop_size 224 --attention_size 14 --backbone vgg16 --backbone_rate 1.0 --function sum_1 --mean_num 20 --lr 2.5e-06 --epoch 10 --danet 0 --decay_epoch 6 --decay_rate 0.1 --seg_thr 0.2 --aux_rate 1.0 --max_num 100 --inception_reduce 0.1 --update_rate 0.001 --scale 1 --min_scale 0.4 --max_scale 1.0 --spa_loss 1 --spa_loss_rate 0.04 --aux_type fore > log/train_vgg16_imagenet_1.txt 2>&1

# todo table 5 VGG16 Imagenet gt loc teacher 104
python -u train_distri.py --gpu 3,2,1 --dataset imagenet --shared_classifier 1 --batch_size 36 --input_size 240 --crop_size 224 --attention_size 14 --backbone vgg16 --backbone_rate 1.0 --function sum_1 --mean_num 20 --lr 6e-05 --epoch 20 --danet 0 --decay_epoch 10 --decay_rate 0.1 --seg_thr 0.0 --aux_rate 1.0 --max_num 100 --inception_reduce 0.1 --update_rate 0.001 --scale 0 --min_scale 0.9 --max_scale 1.1 --spa_loss 1 --spa_loss_rate 1e-06 --aux_type diff_aux > log/train_vgg16_imagenet_2.txt 2>&1

# todo table 6 VGG16 Imagenet cls teacher 104  (new cls)
python -u train_distri.py --gpu 3,2,1 --dataset imagenet --shared_classifier 1 --batch_size 36 --input_size 240 --crop_size 224 --attention_size 14 --backbone vgg16 --backbone_rate 1.0 --function sum_1 --mean_num 20 --lr 5.85e-05 --epoch 20 --danet 0 --decay_epoch 10 --decay_rate 0.1 --seg_thr 0.1 --aux_rate 1.0 --max_num 100 --inception_reduce 0.1 --update_rate 0.001 --scale 1 --min_scale 0.4 --max_scale 1.0 --spa_loss 1 --spa_loss_rate 0.04 --aux_type fore > log/train_vgg16_imagenet_3.txt 2>&1
