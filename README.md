# ErcRetinex

Exploring multi-feature relationship in retinex decomposition for low-light image enhancement TMM 2025

# Update

* 09/2025: We update the training code. 
* 08/2025: We release pretrain models and evaluation code. We also provide the code to compute $\sigma$ (noise level esitmation) in ```noise_est.py```

## Environment

We use ```torch 1.12.1```. For other libraries, the latest versions should work fine. This code should be compatible with most versions.

## Pretrain models
Please download pretrained weights and put them under ```/pretrain/```.

[Google drive](https://drive.google.com/drive/folders/18-0KjvZ5V-nBQ5eDfKFdfmu2dPDUqWtl?usp=sharing) 

## Evaluation datasets

[LOLv1](https://daooshee.github.io/BMVC2018website/)  [LOLv2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view)  [LOLv2Syn](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view)  [SICE](https://drive.google.com/file/d/1gM3QeNDOCzx0m1gpOoQD1TnGv1BELy08/view)

Please ensure your structure follows this format for both existing dataset evaluation and custom dataset evaluation.

```text
DATASETPATH/
├── lowlight_image1.png
├── lowlight_image2.png
├── ...
└── lowlight_imageN.png
```

## Evaluating on existing datsets (LOLv1, LOLv2, LOLv2Syn, SICE)

```
python evaluation.py --model ./pretrain/LOLv1.pth --data_test LOLv1 --data_path PATH/TO/LOLV1 --output_path ./results/LOLv1/ 

python evaluation.py --model ./pretrain/LOLv2.pth --data_test LOLv2 --data_path PATH/TO/LOLV2 --output_path ./results/LOLv2/ 

python evaluation.py --model ./pretrain/LOLv2Syn.pth --data_test LOLv2Syn --data_path PATH/TO/LOLV2SYN --output_path ./results/LOLv2Syn/ 

python evaluation.py --model ./pretrain/SICE.pth --data_test SICE --data_path PATH/TO/SICE --output_path /results/SICE/ 
```

## Inference on custom datasets

You are free to try different weights and values of ```--alpha``` to improve enhancement performance. For instance, ```--alpha``` can be set to 0.1, 0.2, and so on.

```
python inference.py --model ./pretrain/ANYWEIGHTSYOULIKE.pth --data_path PATH/TO/DATASET --output_path PATH/TO/OUTPUT --alpha 0.08

# Example
python inference.py --model ./pretrain/LOLv1.pth --data_path ./dataset/ --output_path /results/LOLv1/ --alpha 0.08
```


## Training dataset

Please download the training set (PairLIE-training-dataset) from [PairLIE](https://github.com/zhenqifu/PairLIE) and place it in the current working directory.

## Training 

We found that two-stage training yields better results.

```
# Training in a single stage, use Reflectance Enhancement Loss at all times.
python train_singlestage.py --data_train ./PairLIE-training-dataset/

# Training in two stages, use Reflectance Enhancement Loss only in the second stage.
python trainS1.py --data_train ./PairLIE-training-dataset/
python trainS2.py --data_train ./PairLIE-training-dataset/ --resume ./weights/epoch_300.pth
```


## Citation
```
@ARTICLE{ErcRetinex,
  author={Guo, Ruoyu and Pagnucco, Maurice and Song, Yang},
  journal={IEEE Transactions on Multimedia}, 
  title={Exploring Multi-feature Relationship in Retinex Decomposition for Low-light Image Enhancement}, 
  year={2025},
  pages={1-14},
}
```


## Acknowledge
[PairLIE](https://github.com/zhenqifu/PairLIE), [Noise level estimation](https://github.com/zsyOAOA/noise_est_ICCV2015)
