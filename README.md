# RFCN with PyTorch
**Note:** This project is pytorch implementation of [RFCN](https://arxiv.org/abs/1605.06409), Resnet101_without_dilation.
This project is mainly based on [faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), while psroi_pooling modules
is copied from another pytorch version of RFCN, [pytorch_RFCN](https://github.com/PureDiors/pytorch_RFCN)

**Difference** Since dilation isn't used in resnet, so the space_scale is 1/32.0 in psroi_pooling,
not 1/16.0 in original paper. As result, I set SCALES=800 and MAX_SIZE=1200. 

### Installation and demo
0. Install the requirements (you can use pip or [Anaconda](https://www.continuum.io/downloads)):

    ```
    conda install pip pyyaml sympy h5py cython numpy scipy
    conda install -c menpo opencv3
    pip install easydict
    ```


1. Clone the Faster RFCN repository
    ```bash
    git clone https://github.com/xingmimfl/pytorch_RFCN.git
    ```

2. Build the Cython modules for nms and the roi_pooling layer
    ```bash
    cd pytorch_RFCN/faster_rcnn
    ./make.sh
    ```

### Training on Pascal VOC 2007

This project use ResNet-101 model converted from Caffe, and you can get it following [RuotianLuo-pytorch-ResNet](https://github.com/ruotianluo/pytorch-resnet).

Since the program loading the data in `pytorch_RFCN/data` by default,
you can set the data path as following.
```bash
cd pytorch_RFCN
mkdir data
cd data
ln -s $VOCdevkit VOCdevkit2007
```
Then you can set some hyper-parameters in `train.py` and training parameters in the `.yml` file.

### Evaluation
Set the path of the trained model in `test.py`.
```bash
cd pytorch_RFCN
python demo.py
```

![image](https://github.com/xingmimfl/pytorch_RFCN/blob/master/demo/out.jpg)

License: MIT license (MIT)
