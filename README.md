# InstructCLIP: Improving Instruction-Guided Image Editing with Automated Data Refinement Using Contrastive Learning (CVPR 2025)

### [Arxiv](http://arxiv.org/abs/2503.18406) | [Image Editing Model](https://huggingface.co/SherryXTChen/InstructCLIP-InstructPix2Pix) | [Data Refinement Model](https://www.dropbox.com/scl/fo/nn9pykrkxuvykdmt6gmok/ALPoT3AqY_sbnD7d_dq3gNE?rlkey=dx6jjujqhnx3l9g9vsz4cibsj&st=892mjrq2&dl=0) | [Data](https://huggingface.co/datasets/SherryXTChen/InstructCLIP-InstructPix2Pix-Data)

## Table of Contents
- [Capabilities](#capabilities)
- [Installation](#installation)
- [Image Editing Instruction Refinement](#image-editing-instruction-refinement)
    - [Data Preparation](#data-preparation)
    - [LD-DINOv2 Training](#ld-dinov2-training)
    - [Instruct-CLIP Training](instruct-clip-training)
    - [Edit Instruction Refinement](#edit-instruction-refinement)
- [Image Editing](#image-editing)
    - [Data Preparation with Refined Instructions](#data-preparation-with-refined-instructions)
    - [Training](#training)
    - [Inference](#inference)
- [Citation](#citation)

## Capabilities

<p align="center">
  <img src="https://github.com/SherryXTChen/Instruct-CLIP/blob/main/assets/teaser_1.png" alt="Figure 1" width="42%">
  <img src="https://github.com/SherryXTChen/Instruct-CLIP/blob/main/assets/teaser_2.png" alt="Figure 2" width="50%">
</p>

## Installation
```
pip install -r requirements.txt
```

## Image Editing Instruction Refinement

### Data Preparation
Download [timbrooks/instructpix2pix-clip-filtered](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) to a new folder `instructclip_datasets`. This is the training data for both LD-DINOv2 and InstructCLIP. Also, download [ip2p_clip_feat.npy](https://www.dropbox.com/scl/fo/id2ow98wqhc38x6csjmxe/AC3SmN-0klY-C6yuZMSd_ic?rlkey=o7mmbh1x60br2y8l3fas2eag2&st=22mqu2si&dl=0) to the same folder. It contains the CLIP text features of all edit instruction in the InstructPix2Pix dataset, which we will use to refine edit instructions later. The dataset folder should look like this:
```
instructclip_datasets
├── instructpix2pix-clip-filtered
│   ├── dataset_dict.json
│   └── train
└── ip2p_clip_feat.npy
```

### LD-DINOv2 Training 

To train LD-DINOv2, run the following command, which save checkpoints in `ckpts/lddinov2` by default:
```bash
bash scripts/train_lddinov2.sh
```
We have also provided the checkpoint [here](https://www.dropbox.com/scl/fo/nn9pykrkxuvykdmt6gmok/ALPoT3AqY_sbnD7d_dq3gNE?rlkey=dx6jjujqhnx3l9g9vsz4cibsj&st=892mjrq2&dl=0).

### Instruct-CLIP Training

To train Instuct-CLIP, run the following command, which load the latest LD-DINOv2 checkpoint from `ckpts/lddinov2/final.ckpt` and save its checkpoints in `ckpts/instructclip` by default:
```bash
bash scripts/train_iclip.sh
```
We have also provided the checkpoint [here](https://www.dropbox.com/scl/fo/nn9pykrkxuvykdmt6gmok/ALPoT3AqY_sbnD7d_dq3gNE?rlkey=dx6jjujqhnx3l9g9vsz4cibsj&st=892mjrq2&dl=0).

### Edit Instruction Refinement

After training, to get the edit instruction from an image pair, run:
```bash
python get_edit_instruction.py --input_path <input_path> --output_path <output_path>
```

We provide two image pairs as examples. When executing the following command
```bash
python get_edit_instruction.py --input_path assets/1_input.jpg --output_path assets/1_output.jpg
```
which outputs `as a 3 d sculpture` for the image pair below, as oppose to `make it a video game character` in the original dataset.

<p align="left">
  <img src="https://github.com/SherryXTChen/Instruct-CLIP/blob/main/assets/1_input.jpg" width="30%">
  <img src="https://github.com/SherryXTChen/Instruct-CLIP/blob/main/assets/1_output.jpg" width="30%">
</p>

When running the script with the other image pairs:
```bash
python get_edit_instruction.py --input_path assets/2_input.jpg --output_path assets/2_output.jpg
```
which outputs `make it spring` for the image pair below, as oppose to `make it a daydream` in the original dataset. 

<p align="left">
  <img src="https://github.com/SherryXTChen/Instruct-CLIP/blob/main/assets/2_input.jpg" width="30%">
  <img src="https://github.com/SherryXTChen/Instruct-CLIP/blob/main/assets/2_output.jpg" width="30%">
</p>

## Image Editing

### Data Preparation with Refined Instructions
We get over 120K samples with refined editing instructions [here](https://huggingface.co/datasets/SherryXTChen/InstructCLIP-InstructPix2Pix-Data). Download it to `instructclip_datasets` as well. Now the folder should look liks this:
```
├── InstructCLIP-InstructPix2Pix-Data
│   ├── dataset_dict.json
│   └── train
├── instructpix2pix-clip-filtered
│   ├── dataset_dict.json
│   └── train
└── ip2p_clip_feat.npy
```

### Training

LD-DINOv2 and Instruct-CLIP checkpoints are needed for training image editing models. See 
We have also provided checkpoints for LD-DINOv2 and Instruct-CLIP [here](https://www.dropbox.com/scl/fo/nn9pykrkxuvykdmt6gmok/ALPoT3AqY_sbnD7d_dq3gNE?rlkey=dx6jjujqhnx3l9g9vsz4cibsj&st=892mjrq2&dl=0), which are needed for fine-tuning InstructPix2Pix. To fine-tune InstructPixPix on our dataset, run the following command where the checkpoints are stored in `ckpts/ip2p_finetuned` by default:
```bash
bash train_instruct_pix2pix.sh
```

### Inference

We provide the checkpoint in [here](https://huggingface.co/SherryXTChen/InstructCLIP-InstructPix2Pix). To use it for image editing:

```python
import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.load_lora_weights("SherryXTChen/InstructCLIP-InstructPix2Pix")
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/SherryXTChen/Instruct-CLIP/main/assets/1_input.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)

prompt = "as a 3 d sculpture"
images = pipe(prompt, image=image, num_inference_steps=20).images
images[0].save("output.jpg")
```

## Citation
If this work is helpful, please kindly cite as:
```bibtex
@misc{chen2025instructclipimprovinginstructionguidedimage,
      title={Instruct-CLIP: Improving Instruction-Guided Image Editing with Automated Data Refinement Using Contrastive Learning}, 
      author={Sherry X. Chen and Misha Sra and Pradeep Sen},
      year={2025},
      eprint={2503.18406},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18406}, 
}
```
