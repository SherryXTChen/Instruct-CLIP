import os
from pathlib import Path
import logging
from tqdm import tqdm
import math
from packaging import version
import shutil
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import torch
import transformers
import diffusers
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, SchedulerMixin

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from dataset import InstructCLIPDataset
from model import LDDinov2, get_features
from utils import *
import wandb

logger = get_logger(__name__)


def run_model(
    args: argparse.Namespace,
    batch: dict, 
    model: LDDinov2,
    dino_ori: torch.nn.Module,
    dino_ori_feat_dict: dict,
    vae: AutoencoderKL,
    noise_scheduler: SchedulerMixin,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    global_step: int,
    timesteps_input: torch.Tensor =None, 
    log_validation: bool =False,
):
    """Function to perform one training step

    Args:
        args (argparse.Namespace): training configuration
        batch (dict): a batch of training data
        model (LDDinov2): LD-DINO model to be trained
        dino_ori (torch.nn.Module): DINOv2 model to compute ground-truth
        dino_ori_feat_dict (dict): dictionary to store DINOv2 features from model hooks
        vae (AutoencoderKL): Stable Diffusion VAE encoder to encode input image with
        noise_scheduler (SchedulerMixin): Stable Diffusion noise scheduler to add noise to image
        accelerator (Accelerator): accelerator with the GPU to load data to
        weight_dtype (torch.dtype): data type to convert data to
        global_step (int): current training step
        timesteps_input (torch.tensor, optional): diffusion timestep that define the amount of noise added to the image. Defaults to None.
        log_validation (bool, optional): whether to log validation results. Defaults to False.

    Returns:
        `dict`: dictionary with loss function values
    """
    cos_simi_func = lambda x, y: nn.CosineSimilarity()(x.float(), y.float()).mean()
    
    # the original DINOv2 takes 224 x 224 images with a specific mean/std
    other_size = (224, 224)
    other_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    inp = batch['input'].to(accelerator.device, dtype=weight_dtype)
    if len(inp.shape) < 4:
        inp = inp.unsqueeze(0)
    loss_dict = {}

    # get the original DINOv2 features
    with torch.no_grad():
        dino_ori_inp = F.interpolate(inp, size=other_size, mode='bilinear')
        dino_ori_inp = other_transform(dino_ori_inp)
        dino_ori_out_dict = dino_ori(dino_ori_inp, is_training=True)
        dino_ori_out = torch.cat(
            (
                dino_ori_out_dict['x_norm_clstoken'].unsqueeze(1),
                dino_ori_out_dict['x_norm_regtokens'],
                dino_ori_out_dict['x_norm_patchtokens'],
            ),
            dim=1
        )

    # encode the image to VAE latent space
    latent_inp = vae.encode(inp).latent_dist.sample() * vae.config.scaling_factor
    
    # for training, if the diffusion timestep is not defined
    # gradually increase the upper range of the diffuson timesteps we randomly sample from
    # to noisify the latent image later
    if timesteps_input is None:
        step_min = args.max_train_steps * 0.1
        step_max = args.max_train_steps * 0.9
        timestep_min = 0
        timestep_max = noise_scheduler.config.num_train_timesteps
        alpha = (timestep_max - timestep_min) / (step_max - step_min)
        curr_max_func = lambda x: min(
            max(
                round(alpha * (x - step_min) + timestep_min),
                1,
            ),
            noise_scheduler.config.num_train_timesteps
        )
        curr_max = curr_max_func(global_step)
        loss_dict['curr_max_timestep'] = torch.tensor(curr_max).to(accelerator.device)
        timesteps = torch.tensor([random.randint(0, curr_max-1)])
    else:
        timesteps = timesteps_input
        
    # add noise to the latent image based on the sampled timestep
    timesteps = timesteps.to(accelerator.device).long()
    noise = torch.randn_like(latent_inp)
    noisy_inp = noise_scheduler.add_noise(latent_inp, noise, timesteps)

    # get the intermediate and final features
    out, dino_feat_dict = model(noisy_inp, timesteps)

    # minimize the l1 loss between predicted features and features from the original DINOv2
    out_loss = F.l1_loss(dino_ori_out.float(), out.float())
    feat_loss = 0
    for idx in range(len(dino_feat_dict)):
        k = f'block_{idx}'
        feat_loss = feat_loss + F.l1_loss(dino_feat_dict[k].float(), dino_ori_feat_dict[k].float())
    loss_dict['loss_out'] = out_loss
    loss_dict['loss_feat'] = feat_loss
    loss_dict['loss_cos_dist'] = 1 - cos_simi_func(dino_ori_out[:, 0], out[:, 0])

    # add feature visualization
    if log_validation:
        ori_patch = dino_ori_out[:, dino_ori.num_register_tokens + 1 :][0]
        patch = out[:, dino_ori.num_register_tokens + 1 :][0]
        ori_image = get_vis(ori_patch)
        image = get_vis(patch)
        dino_ori_input_pil = latent_to_pil(vae, latent_inp[:1])
        input_pil = latent_to_pil(vae, noisy_inp[:1])

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                formatted_images = []
                formatted_images.append(wandb.Image(dino_ori_input_pil, caption="ori input"))
                formatted_images.append(wandb.Image(input_pil, caption="our input"))
                formatted_images.append(wandb.Image(ori_image, caption="ori vis"))
                formatted_images.append(wandb.Image(image, caption="our vis"))
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
            
    # load Stable Diffusion components
    tokenizer, noise_scheduler, vae, _, _ = get_sd_components(args, accelerator.device, weight_dtype)
    
    # define our model
    model = LDDinov2()
    
    # load DINOv2 checkpoint to compute ground-truth features later
    dino_ori = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dino_ori.eval().requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    
    # add model hook to extract features from intermediate layers
    dino_ori_feat_dict = {}
    for idx, block in enumerate(dino_ori.blocks):
        block.register_forward_hook(get_features(dino_ori_feat_dict, f"block_{idx}"))
    
    # load InstructCLIPDataset dataset
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
        list(model.dino.patch_embed.proj.parameters()) + \
        list(model.time_emb_proj.parameters()) + \
        list(model.dino.blocks.parameters())
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
                    args=args,
                    batch=batch, 
                    model=model,
                    dino_ori=dino_ori,
                    dino_ori_feat_dict=dino_ori_feat_dict,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    global_step=global_step,
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
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        model.eval()
                        val_loss_dict_total = defaultdict(list)
                        is_first = True
                        with torch.no_grad():
                            for val_batch in tqdm(val_dataloader):
                                # for the first validation batch, log feature visualization
                                val_loss_dict = run_model(
                                    args=args,
                                    batch=val_batch, 
                                    model=model, 
                                    dino_ori=dino_ori,
                                    dino_ori_feat_dict=dino_ori_feat_dict,
                                    vae=vae,
                                    noise_scheduler=noise_scheduler,
                                    accelerator=accelerator,
                                    weight_dtype=weight_dtype,
                                    global_step=global_step,
                                    timesteps_input=torch.tensor([0]),
                                    log_validation=is_first,
                                )
                                is_first = False
                                for k, v in val_loss_dict.items():
                                    val_loss_dict_total[k].append(v.detach().item())
                        
                        # add validation loss to log
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