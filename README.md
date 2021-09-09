# Global-Reasoned Multi-Task Model for Surgical Scene Understanding

---


<!---------------------------------------------------------------------------------------------------------------->
## Model architecture
### Feature Extractor
TBR
<!---------------------------------------------------------------------------------------------------------------->
### Global-Reasoned Segmentation Unit
TBR
<!---------------------------------------------------------------------------------------------------------------->
### Scene graph
<!---------------------------------------------------------------------------------------------------------------->
TBR
<!---------------------------------------------------------------------------------------------------------------->

## Dir setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch and DGL library, the structure is as follows: 

- `dataset/`: Contains the data needed to train the network.
- `checkpoints/`: Conatins trained weights.
- `models/`: Contains network models.
- `utils/`: Contains utility tools used for training and evaluation

---

## Library Prerequisities

### DGL
<a href='https://docs.dgl.ai/en/latest/install/index.html'>DGL</a> is a Python package dedicated to deep learning on graphs, built atop existing tensor DL frameworks (e.g. Pytorch, MXNet) and simplifying the implementation of graph-based neural networks

### Prerequisites
- Python 3.6
- Pytorch 1.1.0
- DGL 0.3
- CUDA 10.0
- Ubuntu 16.04
---
## Training
### Download dataset for training
1. Frames - endovis challange 2018
2. Insrtument label - TBR
3. BBox and Tool-Tissue interaction annotation - TBR
4. Download the pretrain word2vec model on [GoogleNews](https://code.google.com/archive/p/word2vec/) and put it into `datasets/word2vec`

### Process dataset
1. TBR
### Run training
1. TBR
   
---
## Evaluation

### Dataset
1. Dataset - [data](https://drive.google.com/file/d/1OwWfgBZE0W5grXVaQN63VUUaTvufEmW0/view?usp=sharing) and place it inside `dataset/`
2. checkpoints - [weights](https://drive.google.com/file/d/1HTSYta_Dn9-nF1Df4TUym38Nu0VMtl5l/view?usp=sharing) and place it inside `checkpoints/`


### Run evaluation
1. set the following variables in (model_type, ver, seg_mode, checkpoint_dir) `evaluation.py` and run to reproduce the results.
- model_type
- ver
- seg_mode
- checkpoint_dir

---
## Acknowledgement
Code adopted and modified from :
1. Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion
    - Paper [Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion](https://arxiv.org/abs/2001.02302).
    - Official Pytorch implementation [code](https://github.com/birlrobotics/vs-gats).
1. Graph-Based Global Reasoning Networks
    - Paper [Graph-Based Global Reasoning Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Graph-Based_Global_Reasoning_Networks_CVPR_2019_paper.pdf).
    - Official code implementation [code](https://github.com/facebookresearch/GloRe.git).