# Fourier-Basis-Mapping

### This is an offical implementation of VH-NBEATS and FBM model. 

### Implement the project

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download the ETTH1, ETTH2, Electricity and Traffic data from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and WTH data from [Google Drive](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR?usp=sharing) Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/file_to_implement.sh```
```
sh ./scripts/FBP/ETTh1.sh
sh ./scripts/FBP/ETTh2.sh
sh ./scripts/FBP/ETTm1.sh
sh ./scripts/FBP/ETTm2.sh
```
You can adjust the hyperparameters based on your needs (e.g. different look-back windows and prediction lengths, beta and so on). We also provide codes for some baseline models inside the project.


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



## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
Please contact the author: 2689679964@qq.com or runze.y@sjtu.edu.cn
