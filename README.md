# Image Relighting Network via Contextual Residuals and Multi-scale Attention
### Team: IMC
#### Members: Wei Wang, DU Xiang-cheng and JIN Cheng

This is the implementation of the paper "Image Relighting Network via Contextual Residuals and Multi-scale Attention"

Paper link: ToDo


## Prerequisites
- Linux (Ubuntu 1604)
- Anaconda
- Python 3.7
- NVIDIA RTX2080Ti GPU (11G memory or larger) + CUDA10.2 + cuDNN
- PyTorch1.6.0
- dominate
- kornia 0.2.0
- lpips-pytorch
![](./architecture.png)
## Getting Started
### Installation

- Install PyTorch and dependencies from http://pytorch.org
```bash
pip install git+https://github.com/S-aiueo32/lpips-pytorch.git
```

- Install dominate
```bash
pip install dominate
```
- Install kornia
```bash
pip install kornia==0.2.0
```
- Install lpips-pytorch
```bash 
pip install git+https://github.com/S-aiueo32/lpips-pytorch.git
```


### Testing
-  test images are included in the [here](https://drive.google.com/drive/folders/1QVp7ilkIz_I9VVBVIvkSSiIIoMA8XwaM?usp=sharing) (google drive link)
-  self-regularization images are included in the [here](https://drive.google.com/drive/folders/1E3t4lfsiAgyaXjPFAIebMKlR5DSRMqZd?usp=share_link) (google drive link)
- Please download trained model
  - Generator from [here](https://drive.google.com/file/d/1cgOf-D_pzA_DHbgEXTEB8ZU75l0MUIiq/view?usp=share_link) (google drive link)
  - Discriminator from [here](https://drive.google.com/file/d/1ZzqBNB3SFGkSTMBVN68fljjP78eRHGuX/view?usp=share_link) (google drive link)
  - Put them under `./checkpoints/best_model/`
- Test the model:
```bash
python test.py
```


### Training
```bash
python train.py
```



### Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
TODO
```

## Acknowledgments
This code borrows heavily from [DRN](https://github.com/WangLiwen1994/DeepRelight).
