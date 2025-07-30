# Rethinking Fourier Transform from A Basis Functions Perspective for Long-term Time Series Forecasting. (NeurIPS 2024)

### This is the offical implementation of FBM-L, FBM-NL and FBM-NP model. 
### Please follow our latest work, which is an extension of this paper to a journal: "Fourier Basis Mapping: A Time-Frequency Learning Framework for Time Series Forecasting" on arxiv with the code available at:  https://github.com/runze1223/FBM-S

### Implement the project

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download the ETTh1, ETTh2, ETTm1, ETTm2, Electricity and Traffic data from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and WTH data from [Google Drive](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR?usp=sharing) Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/FBM/file_to_implement.sh```
```
sh ./scripts/FBM/ETTh1.sh
```
You can adjust the hyperparameters based on your needs. Notably, our method requires a smaller learning rate due to the decomposition of values, and the learning rate adjustment strategy 'TST' has been excluded.

![alt text](https://github.com/runze1223/Fourier-Basis-Mapping/blob/main/pic/imag1.png)
![alt text](https://github.com/runze1223/Fourier-Basis-Mapping/blob/main/pic/imag2.png)
## Main Results

![alt text](https://github.com/runze1223/Fourier-Basis-Mapping/blob/main/pic/imag3.png)

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

If you find this repository useful, please consider citing our paper.

If you have any questions, feel free to contact: runze.y@sjtu.edu.cn
