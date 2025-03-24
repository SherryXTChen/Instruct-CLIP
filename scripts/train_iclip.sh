accelerate launch train_iclip.py \
 --tracker_project_name "instructclip" \
 --output_dir ckpts/instructclip \
 --train_data_dir instructclip_datasets/instructpix2pix-clip-filtered \
 --train_batch_size 32 \
 --dataloader_num_workers 8 \
 --max_train_steps 100000 \
 --validation_steps 10000 \
 --checkpointing_steps 10000 \
 --learning_rate 1e-5 \
 --report_to wandb
