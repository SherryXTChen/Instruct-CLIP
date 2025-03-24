import copy
import numpy as np

import torch
import torch.nn as nn

from transformers import CLIPModel, GPT2LMHeadModel
from diffusers.models.embeddings import Timesteps
from utils import normalize


def get_features(features, name):
    """
    Add hook to model layers to get intermediate features
    """
    def hook(model, input, output):
        features[name] = output.detach()  # detach to prevent storing computation graph
    return hook


class Model(nn.Module):
    """
    Base model definition to add checkpoint saving and loading functions
    """
    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def load_pretrained(self, path):
        self.load_state_dict(torch.load(path))


class LDDinov2(Model):
    """
    DINOv2 model that takes VAE encoded latent image that may also be noisied by diffusion noise scheduler
    """
    def __init__(self):
        super().__init__()
        # load DINOv2 backbone
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        
        # change configuration based on latent image input
        self.dino.patch_size = 2
        self.dino.patch_embed.proj = nn.Conv2d(4, 1024, kernel_size=(2, 2), stride=(2, 2))
        self.dino.patch_embed.img_size = (32, 32)
        self.dino.patch_embed.patch_size = (2, 2)
        self.dino.patch_embed.in_chans = 4
        del self.dino.mask_token
        nn.init.xavier_uniform_(self.dino.patch_embed.proj.weight)

        # for timestep conditioning
        del self.dino.cls_token
        self.time_proj = Timesteps(num_channels=320, flip_sin_to_cos=True, downscale_freq_shift=0, scale = 1)
        self.nonlinearity = nn.SiLU()
        self.time_emb_proj = nn.Linear(in_features=320, out_features=1024)
        nn.init.xavier_uniform_(self.time_emb_proj.weight)

    def get_time_embed(self, sample, timestep):
        """
        Get timestep embedding, adapted from 
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py
        """
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb

    def prepare_tokens_with_masks(self, x, emb):
        """
        Preprocess image and timestep embedding, copied from 
        https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
        """
        _, _, w, h = x.shape
        x = self.dino.patch_embed(x)
        x = torch.cat((emb, x), dim=1)
        x = x + self.dino.interpolate_pos_encoding(x, w, h)
        return x

    def forward(self, x, timestep):
        with torch.no_grad():
            emb = self.nonlinearity(self.get_time_embed(sample=x, timestep=timestep))
        emb = self.time_emb_proj(emb).unsqueeze(1)
        x = self.prepare_tokens_with_masks(x, emb)
        feat_dict = {}
        
        # get intermediate features
        for idx, block in enumerate(self.dino.blocks):
            x = block(x)
            feat_dict[f'block_{idx}']  = x
        x_norm = self.dino.norm(x)
        return x_norm, feat_dict # self.dino.head(x_norm[:, 0]), feat_dict


class TextDecoder(Model):
    """
    Text decoder to decoder CLIP feature to captions, adapted from 
    https://github.com/dhg-wei/DeCap/blob/main/train.py
    """
    def __init__(self, in_channels):
        super().__init__()
        self.decoder = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.decoder.lm_head = copy.deepcopy(self.decoder.lm_head) # to avoid accelerator skip saving shared tensor issue
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = nn.Linear(in_channels, self.embedding_size, bias=True)
        self.in_channels = in_channels
        
    def forward(self, clip_features, gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(clip_features).unsqueeze(1)
        embedding_cat = torch.cat([embedding_clip, embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out
    
    @torch.no_grad()
    def infer(self, image_features):
        image_features = normalize(image_features)
        
        # load features of all instructions in the dataset
        text_features = np.load('instructclip_datasets/ip2p_clip_feat.npy')
        text_features = torch.asarray(text_features, dtype=image_features.dtype, device=image_features.device)
        
        # map image features in the text feature space
        sim = image_features@text_features.T.float()
        prefix_embedding = (sim*100).softmax(dim=-1)@text_features.float()
        prefix_embedding /= prefix_embedding.norm(dim=-1,keepdim=True)
        
        # define the maximum output length
        entry_length = 77
        tokens = torch.ones(
            (prefix_embedding.shape[0], 1), 
            dtype=torch.long, device=prefix_embedding.device
        ) * 49406
        end_idx = 49407
        remain_len = entry_length - 2
        
        # autoregressively predict the next token
        embedding_cat = self.clip_project(prefix_embedding).unsqueeze(1)
        for _ in range(remain_len):
            out = self.decoder(inputs_embeds=embedding_cat)
            logits = torch.nn.functional.softmax(out.logits[:, -1, :], dim=-1)
            next_token = torch.argmax(logits, -1).unsqueeze(-1)
            next_token_embed = self.decoder.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            if prefix_embedding.shape[0] == 1 and next_token.item()==end_idx:
                break
            embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)

        end_tokens = torch.ones((
            prefix_embedding.shape[0], 1), 
            dtype=torch.long, device=prefix_embedding.device
        ) * 49407
        tokens = torch.cat((tokens, end_tokens), dim=1)
        return tokens


class InstructCLIP(Model):
    """
    CLIP model for source/target image difference and edit instruction
    """
    def __init__(self):
        super().__init__()
        # get LD-DINO backbone and add hook to get intermediate features
        self.backbone = LDDinov2()
        self.feat_dict = {}
        for idx, block in enumerate(self.backbone.dino.blocks):
            block.register_forward_hook(get_features(self.feat_dict, f"block_{idx}"))

        # initialize text and image encoder from CLIP
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
        )
        self.text_model = model.text_model
        self.text_projection = model.text_projection

        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection
        del self.vision_model.embeddings
        del self.vision_model.pre_layrnorm
        
        self.text_decoder = TextDecoder(in_channels=768)
        self.logit_scale = nn.Parameter(torch.tensor(model.config.logit_scale_init_value))

    @torch.no_grad()
    def get_text_features(self, text):
        return self.text_projection(self.text_model(text)[1])

    def get_image_features(self, inp, out, inp_t, out_t):
        # get dino feature for source and target image based on the diffusion timestep inp_t, out_t
        with torch.no_grad():
            inp_emb, _ = self.backbone(inp, inp_t)
            inp_feat_dict = copy.deepcopy(self.feat_dict)
            out_emb, _ = self.backbone(out, out_t)
            out_feat_dict = copy.deepcopy(self.feat_dict)

        hidden_states = out_emb - inp_emb
        for idx, encoder_layer in enumerate(self.vision_model.encoder.layers):
            # use the source/target feature difference to guide CLIP feature generation
            feature = out_feat_dict[f'block_{idx}'] - inp_feat_dict[f'block_{idx}']
            hidden_states = hidden_states + feature

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
            )
            hidden_states = layer_outputs[0]

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)
        image_features = self.visual_projection(pooled_output)
        return image_features
