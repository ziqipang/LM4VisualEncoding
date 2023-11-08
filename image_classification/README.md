# Image Classification

We describe the procedures to re-produce the experiments for ViT and ViT-LLaMA in the paper. Before proceeding, **please make sure you have downloaded the checkpoint for LLaMA-7B from LLaMA-v1** ([link](https://github.com/facebookresearch/llama/tree/llama_v1)).

Our code-base is built from [DeiT](https://github.com/facebookresearch/deit) and [AbsViT](https://github.com/bfshi/AbSViT/). Great appreciation for their authors and engineers. If you have any questions on our implementation, checking their repository will also help a lot.

## 1. Environment

Install PyTorch 1.7.0+ and torchvision 0.8.1+ from the official website, then install the packages from the `requirements.txt`.

Then prepare the ILSVRC data for ImageNet, including the training and validation set. I found [this script](https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643) very helpful if you didn't have a copy of ImageNet before. Optionally, you can `tar` the training and validation set into `train.tar` and `val.tar` if you need to move these files a lot on your server. Our script support reading images from `.tar` files.

## 2. Running Experiments

### 2.1 Training

#### 2.1.1 Regular Models

Suppose you have the ImageNet images prepared, you can train a `ViT-Small` from our paper by:

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py --exp_name YOUR_EXP_NAME --model vit_small_patch16_224 \
    --data-path YOUR_IMAGENET_PATH --output_dir YOUR_DIR_SAVING_CKPT \
    --num_workers 32 --batch-size 256 --epochs 300 --warmup-epochs 20
```
Then the training will start and write logs into the directory `YOUR_DIR_SAVING_CKPT/YOUR_EXP_NAME/`. I recommend keeping the total batch size (1024), epochs (300), and warm-up epochs (20) the same as our setup. 

To train other models, you can switch `vit_small_patch16_224` to `vit_tiny_patch16_224`, `vit_llama_tiny_patch16_224`, `vit_small_patch16_224`, and `vit_llama_small_patch16_224`.

#### 2.1.2 LLaMA Models

When you train the models with `llama`, please add an argument `--llama_path` pointing to the directory of your LLaMA-7B checkpoints. The contents in the directory should contains things like: `checklist.chk`, `consolidated.00.pth`, and `params.json`.

#### 2.1.3 Training with `Tar` Files

If your server needs to copy the data to some SSD for training, I recommend you use our `tar` option:
```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py --exp_name YOUR_EXP_NAME --model vit_small_patch16_224 \
    --data-path YOUR_IMAGENET_PATH --output_dir YOUR_DIR_SAVING_CKPT \
    --num_workers 32 --batch-size 256 --epochs 300 --warmup-epochs 20 \
    --data_type tar
```

### 2.2 Evaluation

You can always directly read the accuracy for the validation set from the training logs. If you want to conduct separate evaluation:

```bash
python main.py --model vit_small_patch16_224 --data-path YOUR_IMAGENET_PATH --eval --resume CHECKPOINT_PATH 
```
Please remember to switch the `--model` and `--resume` to your desired model and checkpoint path.

## 3. Model Zoo

| Model | Checkpoint | Acc1 | Acc5 |
|---|---|---|---|
| ViT-Tiny | TBD | TBD | TBD |
| ViT-Tiny-LLaMA | TBD | TBD | TBD |
ViT-Small | [[log]](https://uofi.box.com/s/3tysel3ss26ujfvp6wasfacz3gby8udy) / [[model]](https://uofi.box.com/s/qbghai9lm4ol1a94vqcrrcwzi97iauf4) | 80.1 | 95.1 |
ViT-Small-LLaMA | [[log]](https://uofi.box.com/s/i3043j40w24ww2qp9t9aaslb597xokld) / [[model]](https://uofi.box.com/s/qccs3qeny9ea52mu6pu8jdiiksa49m1s) | 80.7 | 95.4 |

We will also upload the checkpoints and logs for our ablation study. Please stay tuned.

## 4. Key Places to Watch

* In [`llama.py`](./models/llama.py), we re-write LLaMA's code by removing positional embedding and auto-regressive attention masks.
* The major modeling of ViT-LLaMA is in [`vit_llama.py`](./models/vit_llama.py). The initialization and forward are straightforward:
```python
# initialization
...
self.llama = LLaMATransformer(llama_configs)
for param in self.llama.parameters():
    param.requires_grad = False
self.llama_dim_mapper1 = nn.Linear(embed_dim, 4096, bias=False)
self.llama_dim_mapper2 = nn.Linear(4096, embed_dim, bias=False)
...

# forward
...
x = self.llama_dim_mapper1(x)
x = self.llama(x)
x = self.llama_dim_mapper2(x)
...
``` 
* In the `main.py`, we use the following lines to load the LLaMA checkpoint:
```python
# load llama checkpoint for the encoder layer
if 'llama' in args.model:
    print("Loading LLaMA checkpoints")
    start_time = time.time()
    checkpoints = sorted(Path(args.llama_path).glob("*.pth"))
    ckpt_path = checkpoints[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.llama.custom_load_state_dict(checkpoint, tail=True, strict=False)
    print(f"Loaded in {time.time() - start_time:.2f} seconds") 
```
