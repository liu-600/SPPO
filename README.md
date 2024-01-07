## Semantic Prompt for Few-Shot Learning

This is the PyTorch implementation of the SPPO.

### Requirements
* Python >= 3.8
* PyTorch >= 1.7.1
* clip (https://github.com/openai/CLIP)
* sentence_transformers (https://github.com/UKPLab/sentence-transformers)

### Datasets
* miniImageNet: https://rec.ustc.edu.cn/share/1341cd00-ffa6-11ed-8581-c30591dc01d6
* tieredImageNet: https://rec.ustc.edu.cn/share/3df5e760-ffa6-11ed-accd-c197f7deb7f7
* CIFAR-FS: https://rec.ustc.edu.cn/share/58e3c480-ffa6-11ed-bdc6-31dddcd9f8de
* FC100: https://rec.ustc.edu.cn/share/72752780-ffa6-11ed-86c2-435b9749436f


Download the dataset you need and put the xxx.tar.gz in ./dataset
```
cd ./dataset
tar -xvzf xxx.tar.gz
```

### Scripts
#### Pre-train the feature extractor
* miniImageNet
```

python pretrain_resnet12.py --gpu 0 --dataset miniImageNet --exp pre-train --rand_aug --repeat_aug
```

* tieredImageNet
```

python pretrain_resnet12.py --gpu 0 --dataset tieredImageNet --exp pre-train --rand_aug --repeat_aug --epochs 300
```
* CIFAR-FS
```

python pretrain_resnet12.py --gpu 0 --dataset CIFAR-FS --exp pre-train --rand_aug --repeat_aug
```
* FC100
```

python pretrain_resnet12.py --gpu 0 --dataset FC100 --exp pre-train --rand_aug --repeat_aug
```
* CUB
```

python pretrain_resnet12.py --gpu 0 --dataset CUB --exp pre-train --rand_aug --repeat_aug
```

#### Fine-tune the model with SP
* miniImageNet
```
1-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp sp --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_300.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp sp_5shot --shot 5 --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth
```
1-shot: python train_resnet12.py --gpu 0 --dataset miniImageNet --exp sp --init checkpoint/miniImageNet/resnet12/pre-train/checkpoint_epoch_080.pth
5-shot: python train_resnet12.py --gpu 0 --dataset miniImageNet --exp sp_5shot --shot 5 --init checkpoint/miniImageNet/resnet12/pre-train/checkpoint_epoch_080.pth
#### Fine-tune +CAM 
1-shot: python train_resnet12_CAM.py --gpu 0 --dataset miniImageNet --exp sp --init checkpoint/miniImageNet/resnet12/pre-traincon50/checkpoint_epoch_072.pth
checkpoint/miniImageNet/resnet12/spnosp/checkpoint_epoch_best.pth 这是预训练后cam调
5-shot: python train_resnet12_CAM.py --gpu 0 --dataset miniImageNet --exp sp_5shot --shot 5 --init checkpoint/miniImageNet/resnet12/pre-traincon50/checkpoint_epoch_072.pth
checkpoint/miniImageNet/resnet12/sp_5shotnosp/checkpoint_epoch_best.pth

1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset miniImageNet --exp sp --init checkpoint/miniImageNet/resnet12/spnosp/checkpoint_epoch_best.pth
5-shot: 
python train_resnet12_CAM_marginloss.py --gpu 0 --dataset miniImageNet --exp sp_5shot --shot 5 --init checkpoint/miniImageNet/resnet12/sp_5shotnosp/checkpoint_epoch_best.pth
python train_resnet12_CAM_marginloss.py --gpu 0 --dataset miniImageNet --exp sp_5shot --shot 5 --init checkpoint/miniImageNet/resnet12/pre-traincon50/checkpoint_epoch_072.pth

* CUB:
1-shot: python train_resnet12_CAM.py --gpu 0 --dataset CUB --exp sp --init checkpoint/CUB/resnet12/pre-train2/checkpoint_epoch_098.pth
5-shot: python train_resnet12_CAM.py --gpu 0 --dataset CUB --exp sp_5shot --shot 5 --init checkpoint/CUB/resnet12/pre-train2/checkpoint_epoch_098.pth

1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset CUB --exp sp --init checkpoint/CUB/resnet12/sppre98_nocsp/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset CUB --exp sp_5shot --shot 5 --init checkpoint/CUB/resnet12/sppre98_nocsp/checkpoint_epoch_best.pth
# nohup python train_resnet12_CAM_stage234.py --nlp_model clip --gpu 0 --dataset CUB --exp sp --init checkpoint/CUB/resnet12/sppre98_nocsp/checkpoint_epoch_best.pth >cub_best_clip.log 2>&1 &

* tieredImageNet
```
1-shot: python train_vit_sp.py --gpu 0 --dataset tieredImageNet --exp sp  --rand_aug --train_episodes 600 --init checkpoint/tieredImageNet/visformer-t/pre-train/checkpoint_epoch_300.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset tieredImageNet --exp sp_5shot --shot 5  --rand_aug --train_episodes 600 --init checkpoint/tieredImageNet/visformer-t/pre-train/checkpoint_epoch_300.pth
```
1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset tieredImageNet --exp sp --rand_aug --train_episodes 600 --init checkpoint/tieredImageNet/resnet12/spmarginloss_spcam1/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset tieredImageNet --exp sp_5shot --shot 5 --train_episodes 600 --init checkpoint/tieredImageNet/resnet12/pre-train/checkpoint_epoch_065.pth
* CIFAR-FS
```
1-shot: python train_vit_sp.py --gpu 0 --dataset CIFAR-FS --exp sp --init checkpoint/CIFAR-FS/visformer-t/pre-train/checkpoint_epoch_800.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset CIFAR-FS --exp sp_5shot --shot 5 --init checkpoint/CIFAR-FS/visformer-t/pre-train/checkpoint_epoch_800.pth
```
1-shot: python train_resnet12_CAM.py --gpu 0 --dataset CIFAR-FS --exp sp --init checkpoint/CIFAR-FS/resnet12/pre-train/checkpoint_epoch_055.pth
5-shot: python train_resnet12_CAM.py --gpu 0 --dataset CIFAR-FS --exp sp_5shot --shot 5 --init checkpoint/CIFAR-FS/resnet12/pre-train/checkpoint_epoch_055.pth

1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset CIFAR-FS --exp sp --init checkpoint/CIFAR-FS/resnet12/sp_5shotnosp/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset CIFAR-FS --exp sp_5shot --shot 5 --init checkpoint/CIFAR-FS/resnet12/sp_5shotnosp/checkpoint_epoch_best.pth
* FC100
```
1-shot: python train_vit_sp.py --gpu 0 --dataset FC100 --exp sp --init checkpoint/FC100/visformer-t/pre-train/checkpoint_epoch_800.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset FC100 --exp sp_5shot --shot 5 --init checkpoint/FC100/visformer-t/pre-train/checkpoint_epoch_800.pth
```
1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset FC100 --exp sp --init checkpoint/FC100/resnet12/sp_5shotnosp/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset FC100 --exp sp_5shot --shot 5 --init checkpoint/FC100/resnet12/sp_5shotnosp/checkpoint_epoch_best.pth
#### Test
* miniImageNet
```
1-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp test --test --episodes 2000 --resume checkpoint/miniImageNet/visformer-t/sp/checkpoint_epoch_best.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp test --shot 5 --test_classifier fc --aug_support 10 --test --episodes 2000 --resume checkpoint/miniImageNet/visformer-t/sp_5shot/checkpoint_epoch_best.pth
```
1-shot: python train_resnet12.py --gpu 0 --dataset miniImageNet --exp test --test --episodes 2000 --resume checkpoint/miniImageNet/resnet12/sp/checkpoint_epoch_best.pth
5-shot: python train_resnet12.py --gpu 0 --dataset miniImageNet --exp test --shot 5 --aug_support 10 --test --episodes 2000 --resume checkpoint/miniImageNet/resnet12/sp_5shot/checkpoint_epoch_best.pth
1-shot:python train_resnet12_CAM_marginloss.py --gpu 0 --dataset miniImageNet --exp test --test --episodes 2000 --resume checkpoint/miniImageNet/resnet12/sppre72_marginloss_spcam/checkpoint_epoch_best.pth
5-shot:(80.00+0.31)python train_resnet12_CAM_nocam.py --gpu 0 --dataset miniImageNet --exp test --shot 5 --aug_support 10 --test --episodes 2000 --resume checkpoint/miniImageNet/resnet12/sppre72_marginloss_spcam/checkpoint_epoch_best.pth
* ***Test+CAM
1-shot: python train_resnet12_CAM.py --gpu 0 --dataset miniImageNet --exp test --test --episodes 2000 --resume checkpoint/miniImageNet/resnet12/sppre72/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM.py --gpu 0 --dataset miniImageNet --exp test --shot 5  --test --episodes 2000 --resume checkpoint/miniImageNet/resnet12/sp_5shot/checkpoint_epoch_best.pth
*CUB
1-shot: python train_resnet12_CAM.py --gpu 0 --dataset CUB --exp test --test --episodes 2000 --resume checkpoint/CUB/resnet12/sppre98/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM.py --gpu 0 --dataset CUB --exp test --shot 5 --aug_support 10 --test --episodes 2000 --resume checkpoint/CUB/resnet12/sp_5shotpre98/checkpoint_epoch_best.pth

1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset CUB --exp test --test --episodes 2000 --resume checkpoint/CUB/resnet12/spmarginloss_spcam1/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset CUB --exp test --shot 5 --aug_support 10 --test --episodes 2000 --resume checkpoint/CUB/resnet12/sp_5shotmarginloss_spcam1/checkpoint_epoch_best.pth
* tieredImageNet
```
1-shot: python train_vit_sp.py --gpu 0 --dataset tieredImageNet --exp test --test --episodes 2000 --resume checkpoint/tieredImageNet/visformer-t/sp/checkpoint_epoch_best.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset miniImageNet --exp test --shot 5 --test_classifier fc --aug_support 10 --test --episodes 2000 --resume checkpoint/tieredImageNet/visformer-t/sp_5shot/checkpoint_epoch_best.pth
```
1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset tieredImageNet --exp test --test --episodes 2000 --resume checkpoint/tieredImageNet/resnet12/spmarginloss_spcam1/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset tieredImageNet --exp test --shot 5 --aug_support 10 --test --episodes 2000 --resume checkpoint/tieredImageNet/resnet12/sp_5shotmarginloss_spcam1/checkpoint_epoch_best.pth
* CIFAR-FS
```
1-shot: python train_vit_sp.py --gpu 0 --dataset CIFAR-FS --exp test --test --episodes 2000 --resume checkpoint/CIFAR-FS/visformer-t/sp/checkpoint_epoch_best.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset CIFAR-FS --exp test --shot 5 --test_classifier fc --aug_support 10 --test --episodes 2000 --resume checkpoint/CIFAR-FS/visformer-t/sp_5shot/checkpoint_epoch_best.pth
```
1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset CIFAR-FS --exp test --test --episodes 2000 --resume checkpoint/CIFAR-FS/resnet12/spmarginloss_spcam/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset CIFAR-FS --exp test --shot 5  --aug_support 10 --test --episodes 2000 --resume checkpoint/CIFAR-FS/resnet12/sp_5shotmarginloss_spcam/checkpoint_epoch_best.pth

* FC100
```
1-shot: python train_vit_sp.py --gpu 0 --dataset FC100 --exp test --test --episodes 2000 --resume checkpoint/FC100/visformer-t/sp/checkpoint_epoch_best.pth
5-shot: python train_vit_sp.py --gpu 0 --dataset FC100 --exp test --shot 5 --test_classifier fc --aug_support 10 --test --episodes 2000 --resume checkpoint/FC100/visformer-t/sp_5shot/checkpoint_epoch_best.pth
```
1-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset FC100 --exp test --test --episodes 2000 --resume checkpoint/FC100/resnet12/spmarginloss_spcam/checkpoint_epoch_best.pth
5-shot: python train_resnet12_CAM_marginloss.py --gpu 0 --dataset FC100 --exp test --shot 5 --aug_support 10 --test --episodes 2000 --resume checkpoint/FC100/resnet12/sp_5shotmarginloss_spcam/checkpoint_epoch_best.pth

t-sne
CUB:
python train_resnet12_CAM_marginloss_sne.py --gpu 0 --dataset CUB --exp test --way 15 --shot 5 --aug_support 10 --test --episodes 1 --sne sne --resume checkpoint/CUB/resnet12/sp_5shotmarginloss_spcam1/checkpoint_epoch_best.pth
python train_resnet12_CAM_marginloss_sne.py --gpu 0 --dataset CUB --exp test --way 15 --shot 5 --aug_support 10 --test --episodes 2 --sne sne --resume checkpoint/CUB/resnet12/sp_5shotmarginloss_spcam1/checkpoint_epoch_best.pth
miniImagnet:
python train_resnet12_CAM_marginloss_sne.py --gpu 0 --dataset miniImageNet --exp test --way 15 --shot 5 --aug_support 10 --test --episodes 2 --sne sne --resume checkpoint/miniImageNet/resnet12/sppre72_marginloss_spcam/checkpoint_epoch_best.pth
