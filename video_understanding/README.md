# Video Understanding

We mainly focus on training ViT models for action recognition on ["something-something-v2" (SSv2)](https://developer.qualcomm.com/software/ai-datasets/something-something) because it emphasizes undertsanding multi-frame actions instead of single-frame semantics. Our code-base is built upon [VideoMAE](https://github.com/MCG-NJU/VideoMAE/). Great appreciation for the authors of datasets and code-bases! If you have any question on our action recognition part, checking their original repositories will also be helpful.

## 1. Data Preparation

* Step 1: download the dataset from SSv2's [official website](https://developer.qualcomm.com/software/ai-datasets/something-something).

* Step 2: As mentioned in VideoMAE, you should (1) preprocess SSv2 into `mp4` formats with height dimension aligned to 240 pixels, and (2) dowload their `train.csv` and `val.csv` from VideoMAE's [google drive](https://drive.google.com/drive/folders/1cfA-SrPhDB9B8ZckPvnh8D5ysCjD-S_I). **If you are confused about the above steps, you are just like me. Please read the following sentence:** checkout the solution provided in [this issue](https://github.com/MCG-NJU/VideoMAE/issues/62#issuecomment-1317957373), which provides a more detailed guide for data preprocessing. I was following the same procedure.

## 2. Environment Setup

We basically follow VideoMAE's [guide](https://github.com/MCG-NJU/VideoMAE/blob/main/INSTALL.md) for installation. We summarize the most important ones below for your convenience:
* `Pytorch >= 1.8.0`
* `timm == 0.4.12`
* `decord` (for decoding videos on-the-fly in the dataloaders)
* `einops`
* We were unable to use `deepspeed` on our server and also made corresponding changes in our code. **If you want to scale up our method, please check out original VideoMAE for integration with deepspeed.**

## 3. Running Experiments


* **Downloading checkpoints.** We finetune the models from the checkpoints pretrained by MAE in a self-supervised way by VideoMAE. Follow the [instructions](./checkpoints/instructions.md) to download the checkpoints and put them in `./checkpoints/`.

* **Training.** Then use the scripts in `./scripts/` to run the training of models, e.g., [ssv2_vitb_llama.sh](./scripts/ssv2_vitb.sh).

* **Evaluation.** The training script will automatically conduct evaluation, displayed at the end of logs. If you want to evaluate a separate checkpoint, please add `--eval` to the training scipt and use `--resume` to point to the checkpoint you would like to evaluate.

## 4. Model Zoo

| Model | Checkpoint | Acc1 | Acc5 |
|---|---|---|---|
| ViT-S | [[log]](https://uofi.box.com/s/notqyv201hz1j3n36yctqfb55xxsfdbz) / [[model]](https://uofi.box.com/s/lyy5xhcnzho2vrfld1fxx9o5jl81g4t2) | 64.71 | 89.15 |
| ViT-S-LLaMA | [[log]](https://uofi.box.com/s/w6y43wawx716oztt2fen2c5ssoa12ytq) / [[model]](https://uofi.box.com/s/9cy9fr5auahk7l8jlmklaoypubduypen) | 65.88 | 89.93 |
ViT-B | [[log]](https://uofi.box.com/s/y4ay6ni8k3jals7e1zw0gkfur3xwkpca) / [[model]](https://uofi.box.com/s/hao0dypy6s353a994u1kzo0owwc8tgc6) | 64.97 | 89.50 |
ViT-B-LLaMA | [[log]](https://uofi.box.com/s/jttcx5q2s6fes8xz38xbdtvx8id90h72) / [[model]](https://uofi.box.com/s/73n9250bi3kah4lf3i58cx06i7b1324r) | 66.03 | 90.25 |

## 5. Key Places to Watch

The modification to video models are quite similar to image classification.

* In [`llama.py`](llama.py), we re-write LLaMA's code by removing positional embedding and auto-regressive attention masks.
* The major modeling of ViT-LLaMA is in [`vit_llama.py`](./modeling_finetune_llama.py). The initialization and forward are straighforward:
```python
# initialization
...
self.llama = LLaMATransformer(llama_configs)
for param in self.llama.parameters():
    param.requires_grad = False
self.llama_dim_mapper1 = nn.Linear(embed_dim, 4096, bias=False)
self.llama_dim_mapper2 = nn.Linear(4096, embed_dim, bias=False)
...
```