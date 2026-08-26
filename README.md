# IdentiFace

**IdentiFace: Multi-Modal Iterative Diffusion Framework for Identifiable Suspect Face Generation in Crime Investigations**

Paper: [IdentiFace.pdf](IdentiFace.pdf). 

This repository provides the official IdentiFace implementation on PixArt-alpha, a **preprocess → train → inference → offline metrics calculation** pipeline.
Note that "long prompt" in the repo refers to "long text" in our paper.

## Setup

Python 3.10. Install a CUDA-enabled PyTorch build first (the versions below were tested with PyTorch 2.11 nightly + CUDA 12.8), then the remaining packages:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install xformers==0.0.34
pip install -r requirements.txt
```

`requirements.txt` lists packages used by preprocess, train, inference, and metrics. Gradio / FastAPI and other unused extras are omitted.

Place data under `dataset/ID-FFHQ/` (images, `data_info_FFHQ.json`, and after preprocessing `InternData/`). Place weights under `models/` as listed below.

## Weights (`models/`)

| File | Used by | Download |
| --- | --- | --- |
| `PixArt-XL-2-1024-MS.pth` | training (`load_from`) | [Hugging Face](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth) |
| `sd-vae-ft-ema/` | preprocess, train, infer | [Hugging Face](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema) |
| `t5-v1_1-xxl/` | preprocess, infer | [Hugging Face](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) |
| `adaface_ir101_webface12m.ckpt` | train AdaFace loss, infer 1:N | [Google Drive](https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view) |
| `SFace_backbone.pth` | infer 1:N | [Google Drive](https://drive.google.com/drive/folders/1cxhzIvyXYRr8ZMnL1AKtslqpn6lYKpwC) |
| SegFace: `SegFace/weights/convnext_celeba_512/model_299.pt` | infer facial masks | [Hugging Face](https://huggingface.co/kartiknarayan/SegFace/resolve/main/convnext_celeba_512/model_299.pt) |
| `fg-clip2-large/` | offline FG-CLIP score (optional) | [Hugging Face](https://huggingface.co/qihoo360/fg-clip2-large) |
| Checkpoint (on ID-FFHQ) | inference model | Coming soon |

## Datasets (`dataset/`)
ID-FFHQ [BaiduNetDisk](https://pan.baidu.com/s/1-MhEnl5IFg11TQJZusw94Q?pwd=n28d) Extraction Code: n28d

After training, copy the run directory (must contain `config.py` and `checkpoints/latest.pth`) to `models/controlnet_checkpoint/`, or set `MODEL_DIR` / `CKPT`.

## 1. Preprocess

```bash
./preprocessing.sh          # GPU 0
./preprocessing.sh 1        # GPU 1
```

Defaults: `dataset/ID-FFHQ`, `data_info_FFHQ.json`. Writes VAE / T5 / edges / LQ / hard-LQ features under `$DATASET_ROOT/InternData/`.

## 2. Train

```bash
./start_training.sh
NUM_GPUS=4 ./start_training.sh
```

Defaults: `configs/train_controlnet_ffhq_1024.py`, `dataset/ID-FFHQ`, `work_dirs/controlnet_ffhq`.

## 3. Inference

```bash
MODEL_DIR=models/controlnet_checkpoint CKPT=models/controlnet_checkpoint/checkpoints/latest.pth \
  ./run_inference.sh 0
```

Optional env vars: `DATA_ROOT` (`dataset/ID-FFHQ`), `OUTPUT_PATH`, `INFER_RANGE_START`, `INFER_RANGE_END`.

## 4. Offline metrics

Evaluates a finished batch folder. Each `result_*` subdirectory should contain `ref_img_*`, `one_shot_*` (and `best_match_*` unless you only score one-shot), `results_best.json`, and `result_prompts.json`.

```bash
./other_tools/compute.sh /path/to/batch_output
```

Default flags include `--only_one_shot --write-csv` and the image/identity metric switches. AdaFace / SFace numbers are **read from `results_best.json`**; they are not re-extracted. Image metrics (LPIPS, FID, IS, MS-SSIM, SR-SIM, FSIM, VIF) are computed from GT vs generated images.

FG-CLIP needs weights at `models/fg-clip2-large/` (or pass `--fgclip-model`). To skip FG-CLIP, set `--FG-CLIP_Score 0` in `compute.sh`.

Writes `batch_metrics_one_shot_summary.json` (and `batch_metrics_best_match_summary.json` if you drop `--only_one_shot`) under the batch folder. `--write-csv` also writes per-sample CSVs.

## Layout

- `preprocess_dataset.py`, `inference_controlnet.py`, `AdaFace.py`, `long_prompt_segmentation.py`
- `configs/train_controlnet_ffhq_1024.py`
- `preprocessing.sh`, `start_training.sh`
- `other_tools/camera_lq_pipeline.py`
- `other_tools/compute.sh`, `other_tools/compute_batch_metrics_offline.py`, `other_tools/other_impls.py`
- `other_tools/FG-CLIP/fgclip_similarity.py`
- `PixArt-alpha/`, `AdaFace/`, `SFace/`, `SegFace/`

## Acknowledgements

We thank [RelaCtrl](https://github.com/360CVGroup/RelaCtrl) and [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) and other related open source projects for providing source code and model weights that this project builds upon.
