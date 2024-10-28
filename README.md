# Fourier-Basis-Mapping

### This is the offical implementation of FBM-L, FBM-NL and FBM-NP model. 

### Implement the project

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download the ETTh1, ETTh2, ETTm1, ETTm2, Electricity and Traffic data from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and WTH data from [Google Drive](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR?usp=sharing) Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/FBM/file_to_implement.sh```
```
sh ./scripts/FBM/ETTh1.sh
```
You can adjust the hyperparameters based on your needs and achieve better results.

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/yuqinie98/PatchTST

https://github.com/ServiceNow/N-BEATS

https://github.com/aikunyi/FreTS

https://github.com/hqh0728/CrossGNN

https://github.com/thuml/iTransformer

https://github.com/kwuking/TimeMixer

https://github.com/VEWOXIC/FITS

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
Please contact the author: 2689679964@qq.com or runze.y@sjtu.edu.cn
