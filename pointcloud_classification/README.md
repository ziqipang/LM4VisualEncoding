# Pointcloud Classification
We describe the procedures to re-produce the experiments for Point-BERT+LLaMA in the paper.  
We build our model based on [PointBERT](https://github.com/lulutang0608/Point-BERT) official implementations.

## 1. Environment
### Install Basic Packages
```
conda create -n pointllama python=3.8
conda activate pointllama
```

```
conda install pytorch==1.21.1 torchvision==0.13.1 -c pytorch
pip install -r requirements.txt
```

### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN
```
# Chamfer Distance
bash install.sh
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Experiments Preparation
### 2.1 Data 
Follow the instructions in [PointBERT](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to download and prepare the preprocessed data.

### 2.2 PointBERT Pretrained Checkpoints
Download the ShapeNet pretrained [Point-BERT.pth](https://cloud.tsinghua.edu.cn/f/202b29805eea45d7be92/?dl=1) checkpoints and put it under `checkpoints/pointbert`.

### 2.3 LLaMA Checkpoints
Download the LLaMA-7B checkpoint from [LLaMA-v1](https://github.com/facebookresearch/llama/tree/llama_v1) and put it under `checkpoints/llama`.

The final directory structure should look like this:
```
checkpoints
│   ├── llama 
│   │   ├── checklist.chk
│   │   ├── consolidated.00.pth
│   │   └── params.json
│   └── pointbert
│       └── Point-BERT.pth
```

## 3. Running Experiments
### Training

**Single GPU:** If you are using a single GPU, run the following command:
```shell
bash scripts/train.sh [CONFIG_PATH] [EXP_NAME]
```

**Slurm:**  If you are using slurm for multi-gpus training, run the following command:
```shell
sbatch scripts/train_slurm.sh [CONFIG_PATH] [EXP_NAME]
```

Replace `[CONFIG_PATH]` with the path to the config file, and `[EXP_NAME]` with the name of the experiment.

The training logs and checkpoints will be saved under `experiments/[EXP_NAME]`

### ScanObjectNN-hardest Training Example
We provide an example of how to train our model on ScanObjectNN dataset.

Train our Point-BERT+LLaMA model on ScanObjectNN-hardest split:
```bash
// Single GPU
bash scripts/train.sh configs/ScanObjectNN_models/PointLLaMa_hardest.yaml PointTransformer_LLaMA
// Slurm
sbatch scripts/train_slurm.sh configs/ScanObjectNN_models/PointLLaMa_hardest.yaml PointTransformer_LLaMA
``` 

### Evaluation
You can evaluate the model by checking the log file saved during training or running the following command with the checkpoint:
```shell
bash scripts/eval.sh [CONFIG_PATH] [CKPT_PATH]
```

## 4. Model Zoo

### ScanObjectNN Dataset
| Model | Split | Checkpoint | Config | Acc |
| :-- | :-- | :--: | :--: | :--: |
| PointBERT | hardest | [model](https://cloud.tsinghua.edu.cn/f/2edb5b2810dc4bd9b796/?dl=1) | [config](configs/ScanObjectNN_models/PointTransformer_hardest.yaml) | 83.07 |
| PointBERT | objectbg | [model](https://cloud.tsinghua.edu.cn/f/c66c28c771e24cd588ad/?dl=1) | [config](configs/ScanObjectNN_models/PointTransformer_objectbg.yaml) | 87.43 |
| PointBERT | objectonly | [model](https://cloud.tsinghua.edu.cn/f/60260a3cbd8940f5bf0d/?dl=1) | [config](configs/ScanObjectNN_models/PointTransformer_objectonly.yaml) | 88.12 |
| **PointBERT+LLaMA** | hardest | [log](https://uofi.box.com/s/v68h8moyfrl2zgak60ruyai765fusnah) / [model](https://uofi.box.com/s/5ks3efjdt91itzxoclfrqu6drht3w645) | [config](configs/ScanObjectNN_models/PointLLaMa_hardest.yaml) | **83.87** |
| **PointBERT+LLaMA**  | objectbg | [log](https://uofi.box.com/s/7e1ek3ncerq028u427feacx2qzm6jcdw) / [model](https://uofi.box.com/s/xwkvypnwcn50fbcc0kxw8f5woc09gkoe) | [config](configs/ScanObjectNN_models/PointLLaMa_objectbg.yaml) | **88.64**| 
| **PointBERT+LLaMA** | objectonly | [log](https://uofi.box.com/s/4hnr4abikyhj74p31zwpk4deqrquxuw0) / [model](https://uofi.box.com/s/tcyur9pqm7ohbfka4fkjkqw2gtpbdrv5) | [config](configs/ScanObjectNN_models/PointLLaMa_objectonly.yaml) | **88.81** |

***ModelNet Dataset**: TBD*

## 5. Key Places to Watch
* In [`llama.py`](./models/llama.py), we re-write LLaMA's code by removing positional embedding and auto-regressive attention masks.
* In [`Point_BERT.py`](./models/Point_BERT.py), we add the LLaMA layer to Point-BERT according to user's config.

### Initialize LLaMa Model
```python
# Line 154:
if config.use_llama:
    llama_default_config = dict(config.llama_cfg)
    self.llama = LLaMATransformer(llama_default_config)
    for param in self.llama.parameters():
        param.requires_grad = False
    self.llama_dim_mapper1 = nn.Linear(config.trans_dim, 4096, bias=False)
    self.llama_dim_mapper2 = nn.Linear(4096, config.trans_dim, bias=False)
```

### Load LLaMa Checkpoint
```python
# Line 215:
if self.config.use_llama:
    print_log("Loading LLaMA checkpoints", logger = 'LLaMA')
    checkpoints = sorted(Path(self.config.llama_path).glob("*.pth"))
    ckpt_path = checkpoints[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    self.llama.custom_load_state_dict(checkpoint, tail=True, strict=False)
    print_log("Loading LLaMA Done", logger = 'LLaMA')
```

### Forward LLaMa Model
```python
# Line 241:
# ...
# x = self.blocks(x, pos)
if self.config.use_llama:
    x = self.llama_dim_mapper1(x)
    x = self.llama(x)
    x = self.llama_dim_mapper2(x)
# x = self.norm(x)
```