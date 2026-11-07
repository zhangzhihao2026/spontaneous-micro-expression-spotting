# spontaneous-micro-expression-spotting

GLACNN: Global-Local Attention Fusion with Sliding Interval Classification for Spontaneous Micro-Expression Spotting

This is the official TensorFlow implementation of the paper:
"Global-Local Attention Fusion with Sliding Interval Classification for Spontaneous Micro-Expression Spotting"
Author: Zhihao Zhang, Ying Huang, Tong Chen, Ke Zhao, Xiaolan Fu

Introduction
Micro-expressions are brief, involuntary facial movements. This project proposes GLACNN, a spatiotemporal network that formulates spotting as sliding-interval classification. It utilizes TV-L1 optical flow sequences from the global face and four local regions of interest (ROIs), processed through 3D CNN branches with Squeeze-and-Excitation (SE) attention modules.

Environment & Dependencies
The code was developed and tested on the following environment:
**OS:  Windows 10
**GPU:  NVIDIA GeForce RTX 2080 (12GB)
**Framework: TensorFlow 2.0.0
**Python:  3.7+

You can install the required packages using:
```bash
pip install -r requirements.txt


Dataset Preparation
We evaluate our method on two public spontaneous micro-expression databases. Please download them from their official sources and strictly follow their license agreements:

CASME II

SMIC-E-HS

Preprocessing (Optical Flow):
We use TV-L1 optical flow. Please place the raw videos in the ./data/ directory and run:

python preprocess_optical_flow.py --dataset CASME_II

Training and Testing

To train the GLACNN model with Leave-One-Subject-Out (LOSO) cross-validation:

python train.py --dataset CASME_II --sliding_interval 0.2 --label_boundary 0.6

To evaluate the trained model:
python test.py --model_path ./weights/CASME_II_fold1.h5

Results
Our method achieves state-of-the-art performance on both datasets:
Dataset	         F1-Score	      Accuracy
CASME II	       0.7597	        0.8167
SMIC-E-HS	       0.4018	        0.8676


