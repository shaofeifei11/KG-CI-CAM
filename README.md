# KG-CI-CAM
This project is the code of the paper "Knowledge-guided Causal Intervention for Weakly-supervised Object Localization".
![1696916298299](https://github.com/shaofeifei11/KG-CI-CAM/assets/50863459/58533b17-520a-4e63-a8df-d76c354cd853)


## KD-CI-CAM-teacher
### training
`./github_run_inceptionV3_cub.sh`

`./github_run_vgg16_cub.sh`

`./github_run_vgg16_imagenet.sh`

### test
please see KD-CI-CAM-student/table*.sh


## KD-CI-CAM-student
### training
`./github_run_baseline.sh`

`./github_run_student_inceptionV3_cub.sh`

`./github_run_student_vgg16_cub.sh`

`./github_run_student_vgg16_imagenet.sh`

### test
table*.sh

# Note
Because the training code does not use the best seg_thr and resize values, we unified our test code in the KD-CI-CAM-student/table*.sh using the best seg_thr and resize values.  

# Install
## 1. randaugment
`pip install git+https://github.com/ildoonet/pytorch-randaugment`

# Model weights
https://drive.google.com/file/d/1pZwzuCsn6JerqGazG3teDDqXDVPlwyvy/view?usp=share_link

# Datasets
## CUB-200-2011
http://www.vision.caltech.edu/datasets/cub_200_2011/
## ILSVRC 2016 val set
https://image-net.org/challenges/LSVRC/2016/#loc
