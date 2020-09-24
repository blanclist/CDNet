# CDNet

This is a PyTorch implementation of our CDNet.

In the current stage, the **test code** of our CDNet is publicly available. To make readers easily understand the internal details of the network structure, we keep the variable names to be consistent with the symbols used in the main paper. *The training code will be publicly released upon the acceptance of this paper*. 

## Prerequisites

- PyTorch 1.4.0
- opencv-python 3.4.2

## Usage

### 1. Clone the repository

```
git clone https://github.com/blanclist/CDNet.git
```

### 2. Download the datasets

We evaluate our CDNet on seven commonly used datasets: NJU2K, STERE, DES, NLPR, SSD, LFSD, and DUT. These datasets can be downloaded from the links provided in http://dpfan.net/d3netbenchmark/.

### 3. Download pre-trained models

We provide two pre-trained CDNets:

1. (*CDNet.pth*) CDNet trained on **NJU2K+NLPR** [GoogleDrive](https://drive.google.com/file/d/1zb2suDroE7TY9qVw2Ss2CbYYN3gmsKoG/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1nsU9k37phtrMqnAQXbVLsQ) (fetch code: j4gx). 

   The evaluation results are listed in *Table I* of the main paper.

2. (*CDNet_2.pth*) CDNet trained on **NJU2K+NLPR+DUT** [GoogleDrive](https://drive.google.com/file/d/1CS0EXjGUnl-8iR61KhOeLCSCHu-jygh7/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1w_e1mZj_LqTALipevwar8g) (fetch code: go86). 

   The evaluation results are listed in *Table II* of the main paper.

### 4. Configure config.py

To run the test code, you are required to assign variables in "*CDNet/code/config.py*" with meaningful values:

- "*img_base*": The directory path of RGB images.
- "*depth_base*": The directory path of depth maps.
- "*checkpoint_path*": The path of pre-trained model (*i.e.*, the path of "*CDNet.pth*" or "*CDNet_2.pth*").
- "*save_base*": The directory path to save the saliency maps produced by the model.

### 5. Test

```
cd CDNet/code/
python main.py
```

**Notes** of the **Input Formats**: The input RGB images and depth maps will be resized to 224*224 and properly normalized by the test code. The pre-trained CDNet requires the input depth maps to follow the criterion: the object is closer to the sensor, the corresponding depth values are lower. To utilize CDNet to normally generate saliency maps, please ensure that the input depth maps follow this criterion before the test.
