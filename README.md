# spontaneous-micro-expression-spotting

GLACNN: Global-Local Attention Fusion with Sliding Interval Classification for Spontaneous Micro-Expression Spotting

This is the official TensorFlow implementation of the paper:
"Global-Local Attention Fusion with Sliding Interval Classification for Spontaneous Micro-Expression Spotting"

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
