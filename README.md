The code base for 'ARTree: A Deep Auto-regressive Model for Phylogenetic Inference'

One should first construct the embeded data by running:
```
python get_embed_data.py --dataset DS1 --repo 1
python get_embed_data.py --dataset DS1 --empFreq
```

For TDE, run the following command under the folder ./TDE/:
```
python main.py --dataset DS1 --repo 1 --empFreq
```

For VBPI, run the following command under the folder ./VBPI/:
```
python main.py --dataset DS1 --empFreq
```