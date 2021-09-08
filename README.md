# Global-Reasoned Multi-Task Model for Surgical Scene Understanding

---


<!---------------------------------------------------------------------------------------------------------------->
## Feature Extractor
TBD
<!---------------------------------------------------------------------------------------------------------------->
## Global-Reasoned Segmentation Unit
TBD
<!---------------------------------------------------------------------------------------------------------------->
## Scene graph
<!---------------------------------------------------------------------------------------------------------------->
TBD
<!---------------------------------------------------------------------------------------------------------------->

## Code Overview
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch and DGL library, the structure is as follows: 

- `datasets/`: Contains the dataset needed to train the network.
- `checkpoints/`: Conatins trained weights
- `models/`: Contains network models.
- `utils/`: Contains utility tools used for training and evaluation.

---

## Library Prerequisities.

### DGL
<a href='https://docs.dgl.ai/en/latest/install/index.html'>DGL</a> is a Python package dedicated to deep learning on graphs, built atop existing tensor DL frameworks (e.g. Pytorch, MXNet) and simplifying the implementation of graph-based neural networks.

### Prerequisites
- Python 3.6
- Pytorch 1.1.0
- DGL 0.3
- CUDA 10.0
- Ubuntu 16.04

---
### Dataset
#### Download feature extracted data for training and evalutation
1. endovis challange 2018
2. Download the pretrain word2vec model on [GoogleNews](https://code.google.com/archive/p/word2vec/) and put it into `datasets/word2vec`.

### Training
TBD

### Evaluation
TBD

---
### Acknowledgement
Code adopted and modified from :
1. Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion
    - Paper [Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion](https://arxiv.org/abs/2001.02302).
    - Official Pytorch implementation [code](https://github.com/birlrobotics/vs-gats).