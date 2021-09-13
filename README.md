<div align="center">

<samp>

<h1> Global-Reasoned Multi-Task Model for Surgical Scene Understanding </h1>

<h3> Seenivasan lalithkumar, Sai Mitheran, Mobarakol Islam, Hongliang Ren </h3>

</samp>   

---
Manuscript submitted to RA-L/ICRA 2022, under review.
---

</div>     
    
---

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

## Directory setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch and DGL library, the structure is as follows: 

- `dataset/`: Contains the data needed to train the network.
- `checkpoints/`: Contains trained weights.
- `models/`: Contains network models.
- `utils/`: Contains utility tools used for training and evaluation

---

## Library Prerequisities

### DGL
<a href='https://docs.dgl.ai/en/latest/install/index.html'>DGL</a> is a Python package dedicated to deep learning on graphs, built atop existing tensor DL frameworks (e.g. Pytorch, MXNet) and simplifying the implementation of graph-based neural networks

### Dependencies
- Python 3.6
- Pytorch 1.1.0
- DGL 0.3
- CUDA 10.0
- Ubuntu 16.04

## Setup (From an Env File)

We have provided environment files for installation using conda

### Using Conda

```bash
conda env create -f environment.yml
```

---
## Data and Training

### Dataset - Train (TBR)
1. Frames - endovis challange 2018
2. Instrument label - TBR
3. BBox and Tool-Tissue interaction annotation - TBR
4. Download the pretrain word2vec model on [GoogleNews](https://code.google.com/archive/p/word2vec/) and put it into `dataset/word2vec`


### Process dataset (For Spatial Features)
1. TBR

### Run training

- Set the model_type, version for the mode to be trained according to the instructions given in the train file

```bash
python3 model_train.py
```
    
---
## Evaluation

For the direct sequence of commands to be followed, refer to [this link](https://github.com/lalithjets/Global-reasoned-multi-task-model/blob/master/eval_instructions.txt)

### Pre-trained Models
Download from **[[`Checkpoints Link`](https://drive.google.com/file/d/1HTSYta_Dn9-nF1Df4TUym38Nu0VMtl5l/view?usp=sharing)]**, place it inside the repository root and unzip  

### Evaluation Data
Download from **[[`Dataset Link`](https://drive.google.com/file/d/1OwWfgBZE0W5grXVaQN63VUUaTvufEmW0/view?usp=sharing)]** and place it inside the repository root and unzip 

### Inference
To reproduce the results, set the model_type, ver, seg_mode and checkpoint_dir based on the table given [here](https://github.com/lalithjets/Global-reasoned-multi-task-model/blob/c6668fcca712d3bd5ca25c66b11d34305103af94/evaluation.py#L195)
- model_type
- ver
- seg_mode
- checkpoint_dir

```bash
python3 evaluation.py
```


---
## Acknowledgement
Code adopted and modified from :
1. Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion
    - Paper [Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion](https://arxiv.org/abs/2001.02302).
    - Official Pytorch implementation [code](https://github.com/birlrobotics/vs-gats).
1. Graph-Based Global Reasoning Networks
    - Paper [Graph-Based Global Reasoning Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Graph-Based_Global_Reasoning_Networks_CVPR_2019_paper.pdf).
    - Official code implementation [code](https://github.com/facebookresearch/GloRe.git).

---
## Contact

For any queries, please contact [Lalithkumar](mailto:lalithjets@gmail.com) or [Sai Mitheran](mailto:saimitheran06@gmail.com)
