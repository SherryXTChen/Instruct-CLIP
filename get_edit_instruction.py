import argparse
from PIL import Image
import torch
from torchvision import transforms

from model import InstructCLIP
from utils import get_sd_components, normalize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of estimating edit instruction from image pair")
    parser.add_argument(
        "--instructclip_ckpt",
        type=str,
        default="ckpts/instructclip/final.ckpt",
        help=(
            "Path to Instruct-CLIP checkpoint"
        )
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help=(
            "sd pretrained checkpoints"
        ),
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="assets/1_input.jpg",
        help=(
            "Input image path"
        )
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="assets/1_output.jpg",
        help=(
            "Output image path"
        )
    )
    args = parser.parse_args()
    device = "cuda"
    
    # load model for edit instruction estimation
    model = InstructCLIP()
    model.load_pretrained(args.instructclip_ckpt)
    model = model.to(device).eval()
    
    # load model to preprocess/encode image to latent space
    tokenizer, _, vae, _, _ = get_sd_components(args, device, torch.float32)
    
    # prepare image input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    image_list = [args.input_path, args.output_path]
    image_list = [
        transform(Image.open(f).resize((512, 512))).unsqueeze(0).to(device) 
        for f in image_list
    ]
    
    with torch.no_grad():
        image_list = [vae.encode(x).latent_dist.sample() * vae.config.scaling_factor for x in image_list]
        
        # get image feature
        zero_timesteps = torch.zeros_like(torch.tensor([0])).to(device) 
        img_feat = model.get_image_features(
            inp=image_list[0], out=image_list[1], inp_t=zero_timesteps, out_t=zero_timesteps)
        img_feat = normalize(img_feat)
        
        # get edit instruction
        pred_instruct_input_ids = model.text_decoder.infer(img_feat[:1])[0]
        pred_instruct = tokenizer.decode(pred_instruct_input_ids, skip_special_tokens=True)
        print(pred_instruct)
