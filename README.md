# MFFNet
code for MFFNet
Our model is provided in the 'models' folder, and the visualization code for the results on two datasets is provided in the 'tools' folder. 
## platform

My platform is like this: 

* ubuntu 22.04
* nvidia 4060Ti gpu
* cuda 10.2/11.3
* cudnn 8
* miniconda python 3.8.8
* pytorch 1.11.0

## Models
The finetuned model parameters are available for download in the [DownLoad](https://drive.google.com/drive/folders/1CA0phChbpck5SqF5xWE-ufXy4bPmcCOG?usp=sharing) section.

## prepare dataset
1.cityscapes  
Register and download the dataset from the official [website](https://www.cityscapes-dataset.com/). Then decompress them into the `data/cityscapes` directory:  
```
$ mv /path/to/leftImg8bit_trainvaltest.zip datasets/cityscapes
$ mv /path/to/gtFine_trainvaltest.zip datasets/cityscapes
$ cd data/cityscapes
$ unzip leftImg8bit_trainvaltest.zip
$ unzip gtFine_trainvaltest.zip
```

2.camvid
Download the dataset from the official [website](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). Then decompress them into the `data/camvid` directory:  


3.Check if the paths contained in lists of data/list are correct for dataset images.

## train

## eval pretrained models
You can evaluate a trained model like this: 
```
$ python tools/eval_city.py --config configs/cityscapes.py --weight-path /path/to/your/weight.pth
```
or 
```
$ python tools/eval_camvid.py --cfg configs/camvid.yaml
```

