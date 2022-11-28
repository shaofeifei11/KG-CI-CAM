# KD-CI-CAM

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
