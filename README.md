# ARTree

Code for [ARTree: A Deep Autoregressive Model for Phylogenetic Inference](https://arxiv.org/abs/2310.09553) (NeurIPS 2023, spotlight)

Please cite our paper if you find this code useful:
```
@inproceedings{
xie2023artree,
title={{ART}ree: A Deep Autoregressive Model for Phylogenetic Inference},
author={Xie, Tianyu and Zhang, Cheng},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=SoLebIqHgZ}
}
```

## Demo

Before your training, please first construct the embeded training set of tree topologies by running:
```
python get_embed_data.py --dataset $DATASET --repo $REPO
```
and construct the embed ground truth of tree topologies by running:
```
python get_embed_data.py --dataset $DATASET --empFreq
```

For TDE, run the following command under the folder ./TDE/:
```
python main.py --dataset $DATASET --repo $REPO --empFreq
```

For VBPI, run the following command under the folder ./VBPI/:
```
python main.py --dataset $DATASET --empFreq
```