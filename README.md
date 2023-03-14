# KD-CI-CAM
This project is the code of the paper "Further Improving Weakly-supervised Object Localization via Causal Knowledge Distillation".
![image](https://user-images.githubusercontent.com/50863459/225069343-8f194e35-664f-4288-881a-38c7c1aebfae.png)

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
