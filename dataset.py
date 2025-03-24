import argparse
from transformers import AutoTokenizer
import datasets

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset
from torchvision import transforms
import random


class InstructCLIPDataset(Dataset):
    def __init__(self, args: argparse.Namespace, tokenizer: AutoTokenizer, ds: datasets.Dataset, split: str):
        """Training Data for LD-DINO and Instruct-CLIP
        
        Args:
            args (argparse.Namespace): arguments to initialize datasets
            tokenizer (AutoTokenizer): tokenizer to encode prompts
            ds (datasets.Dataset): dataset with images and prompts
            split (str): training or validation stage
        """
        self.tokenizer = tokenizer
        self.split = split
        self.ds = ds
        self.size = (args.resolution, args.resolution)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        # list of prompts that keep the images the same
        self.maintain_list = [
            'do nothing',
            'keep it the same',
            'keep image the same',
            'make no change',
            'make no edit',
            'do not change',
            'do not edit anything',
        ]
        self.maintain_list = [self.tokenize_captions(p) for p in self.maintain_list]

    def __len__(self):
        return len(self.ds)

    def tokenize_captions(self, caption: str):
        # tokenize caption to input ids
        inputs = self.tokenizer(
            [caption], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids[0]

    def __getitem__(self, index):
        data = self.ds[index]
        input_image_pil = data['original_image'].convert('RGB').resize(self.size)
        input_image = self.transform(input_image_pil)

        output_image_pil = data['edited_image'].convert('RGB').resize(self.size)
        output_image = self.transform(output_image_pil)

        instruction_text = data['edit_prompt'].lower().replace('.', '')
        instruction = self.tokenize_captions(instruction_text)
        
        data_dict = {
            'input': input_image,
            'output': output_image,
            'instruction': instruction,

            'input_path': f'{index}.jpg',
            'input_pil': input_image_pil,
            'output_pil': output_image_pil,
            'instruction_text': instruction_text,
        }
        return data_dict


class InstructPix2PixDataset(Dataset):
    def __init__(self, args, tokenizer, ds, split):
        self.tokenizer = tokenizer
        self.split = split
        
        self.ds = ds
        if self.split == 'train':
            self.resolution_range = (args.resolution, int(1.125 * args.resolution))
            self.center_crop = transforms.Compose([
                transforms.CenterCrop((args.resolution, args.resolution))
            ])
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.resolution_range = (512, 512)
            self.center_crop = transforms.Compose([])
            self.transform = transforms.Compose([])
         
        self.maintain_text_list = [
            'do nothing',
            'keep it the same',
            'keep image the same',
            'make no change',
            'make no edit',
            'do not change',
            'do not edit anything',
        ]
        self.maintain_list = [self.tokenize_captions(p) for p in self.maintain_text_list]     
         
    def __len__(self):
        return len(self.ds)

    def tokenize_captions(self, caption):
        inputs = self.tokenizer(
            [caption], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids[0]

    def __getitem__(self, index):
        data = self.ds[index]
        input_image_pil = data['source_image'].convert('RGB')
        output_image_pil = data['target_image'].convert('RGB')
        
        instruction_text = data['instruction']
        old_instruction_text = data['original_instruction']
            
        # random resize and center crop to desired resolution
        resolution = random.randint(*self.resolution_range)
        input_image_pil = self.center_crop(
            input_image_pil.resize((resolution, resolution))
        )
        output_image_pil = self.center_crop(
            output_image_pil.resize((resolution, resolution))
        )
        # random horizontal flip
        if random.randint(0, 1):
            input_image_pil = input_image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            output_image_pil = output_image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            
        input_image = self.transform(input_image_pil)
        output_image = self.transform(output_image_pil)
        instruction = self.tokenize_captions(instruction_text) if self.split == 'train' else instruction_text        
        old_instruction = self.tokenize_captions(old_instruction_text) if self.split == 'train' else old_instruction_text
        
        return {
            'original_pixel_values': input_image,
            'edited_pixel_values': output_image,
            'input_ids': instruction,
            'old_input_ids': old_instruction,
        }
