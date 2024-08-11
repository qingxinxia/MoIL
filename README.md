# MoIL
Update: Our previous work has been accepted by PerCom (WorkShop) 2024 and won the "Best WiP Paper Award". [link](https://percom.org/2024/awards/)

The implementation of the SSL baselines is mainly based on the excellent work CL-HAR [link](https://github.com/Tian0426/CL-HAR).

## Quick Start
This is our proposed method MoIL (CNNRNN)

```
python main_SSL.py --framework 'CNNRNN' --backbone 'CNNRNN' --dataset 'openpack' --user_name 'U0101' --n_epoch 1000  --batch_size 2048 
```
Try baseline methods - simCLR:
```
python main_SSL.py --framework 'simclr'  --backbone 'CNNRNN' --dataset 'openpack' --user_name 'U0101' --n_epoch 1000  --batch_size 2048 
```
Baseline method - simsiam:
```
python main_SSL.py --framework 'simsiam' --backbone 'CNNRNN'  --dataset 'openpack' --user_name 'U0101' --n_epoch 1000  --batch_size 2048 --criterion cos_sim 
```
Baseline method - BYOL:
```
python main_SSL.py --framework 'byol' --backbone 'CNNRNN'  --dataset 'openpack' --user_name 'U0101' --n_epoch 1000  --batch_size 2048 --criterion cos_sim 
```
Baseline method - Bert
```
python main_SSL.py --framework 'SSL' --backbone 'Transformer'  --dataset 'openpack' --user_name 'U0101' --n_epoch 1000  --batch_size 2048 --criterion mse 
```
Baseline method - Multi task
```
python main_SSL.py --framework 'multi' --backbone 'CNN'  --dataset 'openpack' --user_name 'U0101' --n_epoch 1000  --batch_size 2048 --criterion binary 
```


## Supported Datasets
- OpenPack [link](https://open-pack.github.io/)
- Skoda [link](http://har-dataset.org/doku.php?id=wiki:dataset)

## Download training data
```
|data
├──OpenPackDataset
     └── v_3.1
├──omeData
     └── raw
├──skodaData
     └── skoda_wd
├──LogiData
     └── LogiData_wd
```


## SSL Models
```
Refer to ```models/frameworks.py```. For sub-modules (projectors, predictors) in the frameworks, refer to ```models/backbones.py```
- SimSiam
- BYOL
- SimCLR
- Multi-task
- Transformer (Masked reconstruction)
- MoIL (CNNRNN, Proposed)
```

## Reference
If you find any of the codes helpful, kindly cite our paper.

```
>@inproceeding{xia2024preliminary,
>  title={Preliminary Investigation of SSL for Complex Work Activity Recognition in Industrial Domain via MoIL},
 > author={Xia, Qingxin and Maekawa, Takuya and Morales, Jaime and Hara, Takahiro and Oshima, Hirotomo and Fukuda, Masamitsu and Namioka, Yasuo},
 > booktitle={2024 IEEE International Conference on Pervasive Computing and Communications Workshops and other Affiliated Events (PerCom Workshops)},
>  pages={465--468},
 > year={2024},
>  organization={IEEE}
>}
```


## Related Links
Part of the augmentation transformation functions are adapted from
- https://github.com/emadeldeen24/TS-TCC
- https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
- https://github.com/LijieFan/AdvCL/blob/main/fr_util.py

Part of the contrastive models are adapted from 
- https://github.com/Tian0426/CL-HAR
- https://github.com/lucidrains/byol-pytorch
- https://github.com/lightly-ai/lightly
- https://github.com/emadeldeen24/TS-TCC
- https://github.com/harkash/contrastive-predictive-coding-for-har

Loggers used in the repo are adapted from 
- https://github.com/emadeldeen24/TS-TCC
- https://github.com/fastnlp/fitlog
