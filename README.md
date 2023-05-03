The code base for 'ARTree: A Deep Auto-regressive Model for Phylogenetic Inference'

Before your training, please first construct the embeded training set of tree topologies by running:
```
python get_embed_data.py --dataset $DATASET --repo $REPO
```
and construct the embed ground truth of tree topologies by running
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