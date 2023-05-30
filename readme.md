# XSMT Sovler

 The main inspiration for this project comes from the paper REST: Constructing Rectilinear Steiner Minimum Tree via Reinforcement Learning , which is used to solve the RSMT problem. The project reconfigures the problem to XSMT and incorporates an option for precise counting in the online wirelength calculation.

## Introduction

The XSMT  refers to the X-Structure Steiner Tree, which is a tree structure that connects a set of points on a two-dimensional plane in the shape of the letter "X". 

Given two pins, A = (x1, y1) and B = (x2, y2), where x1 ≤ x2, the following four basic connection actions are defined:

1. Horizontal Connection: Connect A and B horizontally by moving from (x1, y1) to (x2, y1).
2. Vertical Connection: Connect A and B vertically by moving from (x1, y1) to (x1, y2).
3. 45-Degree Connection: Connect A and B diagonally at a 45-degree angle by moving from (x1, y1) to (x2, y2) along the diagonal line.
4. 135-Degree Connection: Connect A and B diagonally at a 135-degree angle by moving from (x1, y1) to (x2, y2) along the diagonal line.

<div align="center">

<style>
    .grid-image {
        width: 200px;
        height: 200px;
        object-fit: cover;
    }
</style>

|                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://github.com/fugjgjguhih/image-hosting-service/blob/main/a1.png?raw=true" alt="图片1" class="grid-image"> | <img src="https://github.com/fugjgjguhih/image-hosting-service/blob/main/a2.png?raw=true" alt="图片2" class="grid-image"> |
| <img src="https://github.com/fugjgjguhih/image-hosting-service/blob/main/a3.png?raw=true" alt="图片3" class="grid-image"> | <img src="https://github.com/fugjgjguhih/image-hosting-service/blob/main/a4.png?raw=true" alt="图片4" class="grid-image"> |

</div>

## Installation

My code has been implemented on Python 3.9 and PyTorch 1.8.1+cu116 ,See other required packages in `requirements.txt`.

```
``pip install -r requirements.txt`
```



## Data preparation

You can modify and run the utils/generate_dataset.py  to  generate your test and vaild set. or you can transfer your dataset to a csv file, in which each row represents a set of point position.

## Usage

#### train

```
python train.py --vaild=${YOUR_VAILD_SET_PATH} --n_pin=${DEGREE_OF_POINT_SET}  --epochs=${NUMBER OF EPOCHS} --model_dir=${SAVEPATH_OF_CHECKPOINT}
```

#### test

```
python test.py --vaild=${YOUR_VAILD_SET_PATH} --n_pin=${DEGREE_OF_POINT_SET}  --epochs=${NUMBER OF EPOCHS}
--model_dir=${SAVEPATH_OF_CHECKPOINT}
```

#### result

![](https://github.com/fugjgjguhih/image-hosting-service/blob/main/a5.png?raw=true)
