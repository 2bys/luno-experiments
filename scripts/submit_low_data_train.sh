#!/bin/bash

python3 submit/submit.py --mode slurm --script train \
  --data_name diff_lin_1 diff_ks_cons_1 diff_hyp_diff_1 diff_burgers_1 \
  --num_epochs 100 \
  --batch_size 5 \
  --num_train_samples 25 \
  --seed 0 \
  --enable_progress_bar False \
  --check_val_every_n_epoch 10 \
  --wandb True
