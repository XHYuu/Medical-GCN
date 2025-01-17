# Medical-GCN
### Dataset:
* row data: https://drive.google.com/file/d/1ZI-lO4dZFtyOwyf8lxxAvaWB4Hal59bn/view?usp=drive_link
* post process data: https://drive.google.com/file/d/1GIzX3gAK_VmeHPYLXP83G0LpLL-hPvBJ/view?usp=drive_link

### Model Dictionary
#### use basic GCN
``` bash
python train-5fold.py --epochs 600 --model_name GCN_basic
```

#### use Graph Attention Network
``` bash
python train-5fold.py --epochs 600 --model_name GAT
```