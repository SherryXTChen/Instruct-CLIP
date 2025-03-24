from typing import List
import os
from pathlib import Path
import logging
from tqdm import tqdm
import math
import shutil
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
import diffusers
from diffusers import SchedulerMixin
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from dataset import InstructCLIPDataset
from model import InstructCLIP
from utils import *
import wandb

logger = get_logger(__name__)


def contrastive_loss(logits: torch.Tensor):
    """
    constractive loss, adapted from
    https://sachinruk.github.io/blog/2021-03-07-clip.html
    """
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor):
    """
    CLIP loss for image difference and edit instruction matching
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def run_model(
    batch: dict, 
    model: InstructCLIP,
    vae: AutoencoderKL,
    noise_scheduler: SchedulerMixin,
    tokenizer: AutoTokenizer,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    timesteps_input: torch.Tensor =None, 
    maintain_list: List[str] =None,
    log_validation: bool =False, 
):
    """Function to perform one training step

    Args:
        batch (dict): a batch of training data
        model (InstructCLIP): Instruct-CLIP model to be trained
        vae (AutoencoderKL): Stable Diffusion VAE encoder to encode input image with
        noise_scheduler (SchedulerMixin): Stable Diffusion noise scheduler to add noise to image
        tokenizer (AutoTokenizer): Stable Diffusion tokenizer to encode text to input ids
        accelerator (Accelerator): accelerator with the GPU to load data to
        weight_dtype (torch.dtype): data type to convert data to
        timesteps_input (torch.Tensor, optional): diffusion timestep that define the amount of noise added to the image. Defaults to None.
        maintain_list (List[str], optional): List of edit instructions that maintain the images. Defaults to None.
        log_validation (bool, optional): whether to log validation results. Defaults to False.

    Returns:
        _type_: _description_
    """
    cos_simi_func = lambda x, y: nn.CosineSimilarity()(x.float(), y.float()).mean()
    loss_ce = torch.nn.CrossEntropyLoss()

    inp = batch['input'].to(weight_dtype).to(accelerator.device)
    out = batch['output'].to(weight_dtype).to(accelerator.device)
    instruct = batch['instruction'].to(accelerator.device)
    
    random_instruct = None
    if 'random_instruction' in batch:
        random_instruct = batch['random_instruction'].to(accelerator.device)

    if len(inp.shape) < 4:
        inp = inp.unsqueeze(0)
        out = out.unsqueeze(0)
        instruct = instruct.unsqueeze(0)
        if random_instruct is not None:
            random_instruct = random_instruct.unsqueeze(0)
    loss_dict = {}
    
    # mix no change instruct
    if maintain_list is not None:
        ratio = 0.05
         # Create a mask where each entry has a 'ratio' chance of being True
        mask = torch.rand(instruct.size(0)) < ratio

        # Extract corresponding no_change elements based on the mask
        if mask.sum().item() > 0:
            no_change = torch.stack(
                random.choices(maintain_list, k=mask.sum().item()), dim=0
            ).to(accelerator.device)

            # Replace elements in 'instruct' and 'out' using the mask
            instruct[mask] = no_change
            out[mask] = inp[mask]
    
    with torch.no_grad():
        latent_inp = vae.encode(inp).latent_dist.sample() * vae.config.scaling_factor
        latent_out = vae.encode(out).latent_dist.sample() * vae.config.scaling_factor
        if timesteps_input is None:
            timesteps = torch.tensor([random.randint(0, noise_scheduler.config.num_train_timesteps-1)])
        else:
            timesteps = timesteps_input
        timesteps = timesteps.to(accelerator.device).long()
        noise = torch.randn_like(latent_out)
        noisy_out = noise_scheduler.add_noise(latent_out, noise, timesteps)

        text_feat = accelerator.unwrap_model(model).get_text_features(instruct)
        text_feat = normalize(text_feat)
    
    loss_text_decoder = 0
    lst = [(text_feat, instruct)]
    for tf, inst in lst:
        logits = model.text_decoder(tf, inst).logits[:, :-1]
        logits = logits.reshape(-1, logits.shape[-1])
        clip_tokens = inst.flatten()
        loss_text_decoder = loss_text_decoder + loss_ce(logits, clip_tokens)
    loss_dict['loss_text_decoder'] = loss_text_decoder / len(lst)
    
    zero_timesteps = torch.zeros_like(timesteps)
    img_feat = accelerator.unwrap_model(model).get_image_features(
        inp=latent_inp, out=noisy_out, inp_t=zero_timesteps, out_t=timesteps)
    img_feat = normalize(img_feat)
    
    logit_scale = accelerator.unwrap_model(model).logit_scale.exp()
    logits_per_text = torch.matmul(text_feat, img_feat.t()) * logit_scale
    loss_dict['loss_contrastive_text_image'] = clip_loss(logits_per_text)

    # visualize image and predicted instruction
    if log_validation:
        input_pil = latent_to_pil(vae, latent_inp[:1])
        ori_output_pil = latent_to_pil(vae, latent_out[:1])
        output_pil = latent_to_pil(vae, noisy_out[:1])
        cos_simi = cos_simi_func(img_feat, text_feat).detach().item()
        instruct_text = tokenizer.decode(instruct[0].detach(), skip_special_tokens=True)
        
        pred_instruct_input_ids = accelerator.unwrap_model(model).text_decoder.infer(img_feat[:1])[0]
        pred_instruct = tokenizer.decode(pred_instruct_input_ids, skip_special_tokens=True)
        pred_instruct = pred_instruct.replace('\n', '')

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                formatted_images = []
                formatted_images.append(wandb.Image(input_pil, caption=f"edit input, instruct = {instruct_text}"))
                formatted_images.append(wandb.Image(ori_output_pil, caption=f"edit output, pred_inst = {pred_instruct}"))
                formatted_images.append(wandb.Image(output_pil, 
                    caption=f"noisy edit output, t = {timesteps.detach().item()}, cos_simi = {cos_simi}"))
                tracker.log({"validation": formatted_images})
    return loss_dict


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # define our model and dataset
    model = InstructCLIP()
    model.backbone.load_pretrained('ckpts/lddinov2/final.ckpt')
    tokenizer, noise_scheduler, vae, _, _ = get_sd_components(args, accelerator.device, weight_dtype)
    train_dataset, _, train_dataloader, val_dataloader = get_dataloader(args, InstructCLIPDataset, tokenizer)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = \
        list(model.vision_model.parameters()) + \
        list(model.visual_projection.parameters()) + \
        list(model.text_decoder.parameters()) + \
        [model.logit_scale]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    model, optimizer, train_dataloader, lr_scheduler = \
        accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    model.train()
    for _ in range(first_epoch, args.num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # one forward call
                loss_dict = run_model(
                    batch=batch, 
                    model=model,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    maintain_list=train_dataset.maintain_list,
                )
                
                # sum loss together
                loss = 0
                for k, v in loss_dict.items():
                    if k.startswith('loss_'):
                        loss = loss + v

                # back-propagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # logging
            logs = {k:v.detach().item() for k,v in loss_dict.items()}
            logs["lr"] = lr_scheduler.get_last_lr()[0]

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        accelerator.unwrap_model(model).save_pretrained(os.path.join(args.output_dir, 'final.ckpt'))
        
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        val_loss_dict_total = defaultdict(list)
                        is_first = True
                        with torch.no_grad():
                            for val_batch in tqdm(val_dataloader):
                                # for the first validation batch, log visualization
                                val_loss_dict = run_model(
                                    batch=val_batch, 
                                    model=model,
                                    vae=vae,
                                    noise_scheduler=noise_scheduler,
                                    tokenizer=tokenizer,
                                    accelerator=accelerator,
                                    weight_dtype=weight_dtype,
                                    timesteps_input=torch.tensor([0]),
                                    log_validation=is_first, 
                                )
                                is_first = False
                                for k, v in val_loss_dict.items():
                                    val_loss_dict_total[k].append(v.detach().item())
                        for k, v in val_loss_dict_total.items():
                            logs['val_' + k] = sum(v) / len(v)

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # save the final checkpoint
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(os.path.join(args.output_dir, 'final.ckpt'))
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)