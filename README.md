# Balanced and Hierarchical Relation Learning for One-shot Object Detection
This repository is an official implementation of the CVPR 2022 paper "Balanced and Hierarchical Relation Learning for One-shot Object Detection", based on [mmdetection](https://github.com/open-mmlab/mmdetection).
![BHRL](images/BHRL.png)

## 训练集地址
https://github.com/Wangjing1551/LogoDet-3K-Dataset<br/>
https://hangsu0730.github.io/qmul-openlogo/<br/>
https://github.com/hq03/FoodLogoDet-1500-Dataset<br/>
https://github.com/InputBlackBoxOutput/logo-images-dataset<br/>
https://github.com/mubastan/osld<br/>
https://github.com/neouyghur/METU-TRADEMARK-DATASET<br/>
## 类似项目
https://github.com/Heldenkombinat/Logodetect<br/>

## 视频数据标注借鉴
https://github.com/SA-PKU/sports-video-logo-dataset<br/>

## 模型训练
```shell
# e.g.,
./tools/dist_train.sh
```

## 数据组织格式
We expect the directory structure to be the following:
```
BHRL
├── data
│   ├──dataset_name
│   │   ├── voc_annotation
│   │   ├── VOC2007
│   │   ├── VOC2012
...
```

## Installation

1. Create a conda virtual environment and activate it

```shell
conda create -n BHRL python=3.7 -y
conda activate BHRL
```

2. Install PyTorch and torchvision 

```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install mmcv

```shell
pip install mmcv-full==1.3.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html
```

4. Install build requirements and then install MMDetection.

```shell
pip install -r requirements/build.txt
pip install -v -e . 
```

