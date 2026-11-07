# GLACNN: Global-Local Attention Fusion with Sliding Interval Classification for Spontaneous Micro-Expression Spotting

This is the official implementation of the project:
**"Global-Local Attention Fusion with Sliding Interval Classification for Spontaneous Micro-Expression Spotting"**

## 📖 Introduction
Micro-expressions are brief, involuntary facial movements that leak suppressed emotions. Spotting them in long untrimmed videos is challenging due to their short duration, low intensity, and frequent non-expression motions. 

To address these challenges, we propose **GLACNN**, a spatiotemporal network that formulates spotting as a sliding-interval classification task. Raw videos are divided into short overlapping intervals, and each interval is classified as micro-expression (ME) or non-micro-expression. The model utilizes TV-L1 optical flow sequences from the global face and four local regions of interest (ROIs: left/right eyebrow, left/right mouth corner) to capture holistic facial dynamics and diagnostically relevant regional movements.

## 🛠️ Environment & Dependencies
The code was developed and tested on the following environment:
*   **OS:** Ubuntu / Windows 10
*   **Hardware:** NVIDIA GeForce RTX 2080 (12G)
*   **Deep Learning Framework:** TensorFlow 2.0.0
*   **Python:** 3.7+

Install the required packages using:
```bash
pip install -r requirements.txt
```

## 📂 Dataset Preparation
We evaluate our method on two widely used spontaneous micro-expression datasets. Please download them from their official sources and strictly follow their license agreements:
*   **[CASME II]** (26 subjects, 247 samples, 200 fps)
*   **[SMIC-E-HS]** (16 subjects, 157 samples, 100 fps)

### Preprocessing (Optical Flow Estimation)
The network takes **TV-L1 optical flow sequences** as input. 
*   Global face input size: `32×64×64×3`
*   Mouth corners input size: `16×32×32×3`
*   Eyebrow regions input size: `16×16×32×3`

You can generate the optical flow features using our preprocessing script (using the first frame of the interval as the reference):
```bash
python preprocess_optical_flow.py --dataset CASME_II --data_dir ./raw_videos/ --save_dir ./optical_flow/
```

## 🚀 Training and Evaluation
### Sliding Interval Generation
Long videos are segmented using a **sliding interval of 0.2s** (step = 0.1s). The label boundary \(\alpha\) is set to **0.6** for generating positive and negative intervals based on the overlap ratio with the ground truth.

### Training
To train the GLACNN model with Leave-One-Subject-Out (LOSO) cross-validation:
```bash
python train.py --dataset CASME_II --sliding_interval 0.2 --label_boundary 0.6 --batch_size 8 --epochs 100
```

### Testing
To evaluate the trained model on the test subject:
```bash
python test.py --dataset CASME_II --model_path ./weights/CASME_II_fold1.h5
```

## 📊 Results
Our method achieves state-of-the-art performance on both datasets, outperforming hand-crafted features (LBP-TOP, HOG3D), feature-difference methods, and other deep learning baselines.

| Dataset | F1-Score | Accuracy |
| :--- | :--- | :--- |
| **CASME II** | **0.7597** | 0.8167 |
| **SMIC-E-HS** | **0.4018** | 0.8676 |

## 📝 Citation
If you find this code useful for your research, please cite our paper:
```bibtex
@article{zhang2026glacnn,
  title={Global-Local Attention Fusion with Sliding Interval Classification for Spontaneous Micro-Expression Spotting},
  author={Zhihao Zhang, Ying Huang, Tong Chen, Ke Zhao, Xiaolan Fu},
  journal={Entropy},
  year={2026},
  publisher={MDPI}
}
```

## ✉️ Contact
If you have any questions or encounter issues running the code, please open an issue in this repository or contact:
**Zhihao Zhang** - 102504@aqnu.edu.cn
```

---
