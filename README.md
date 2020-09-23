# CDNet

This is a PyTorch implementation of our CDNet.

In current stage, the **test code** of our CDNet is publicly available. We keep the variable names to be consistent with the symbols used in the main paper, to help readers to easily understand the internal details of the network structure. <u>The training code will be released until the main paper is accepted.</u> 

## Prerequisites

- PyTorch 1.4.0
- opencv-python 3.4.2

## Usage

### 1. Clone the repository

```
git clone https://github.com/blanclist/CDNet.git
```

### 2. Download the datasets

We evaluate our CDNet on commonly used seven datasets, including: NJU2K, STERE, DES, NLPR, SSD, LFSD, and DUT. These datasets could be downloaded from the links provided in http://dpfan.net/d3netbenchmark/.

### 3. Download pre-trained models

We provide two pre-trained CDNets:

1. (*CDNet.pth*) CDNet trained on **NJU2K+NLPR** [GoogleDrive](https://drive.google.com/file/d/15x2dnzAySxNa8xXd-L6mF3T0M_QCf9HR/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1AY-NSrXI0IQt2dUzA4BLZw) (fetch code: s0mu). The evaluation results are listed in Table I of the main paper.

2. (*CDNet_2.pth*) CDNet trained on **NJU2K+NLPR+DUT** [GoogleDrive](https://drive.google.com/file/d/1MgTHjexvU-y_qNu7xoPUJbvhPnaKqSjt/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1PE7oOxjNbTrzAaKfTWsajg) (fetch code: brof). The evaluation results are listed in Table II of the main paper.

### 4. Configure config.py

To run the test code, you should assign variables with meaningful value:

- "*img_base*": The directory path of RGB images.
- "*depth_base*": The directory path of depth maps.
- "*checkpoint_path*": The path of pre-trained model (*i.e.*, *CDNet.pth* or *CDNet_2.pth*).
- "*save_base*": The directory path to save the saliency maps produced by the model.

### 5. Test

```
cd CDNet/code/
python -main
```

**Note**: The input RGB images and depth maps will be resized to the size of $224\times 224​$ and normalized to proper magnitude by the test code. But the test code requires the input depth maps to follow the criterion: the depth values are lower, the object is more closer to the sensor. To ensure the CDNet to test normally, please adjust the input depth maps to follow this criterion.