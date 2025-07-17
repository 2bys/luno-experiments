#!/bin/bash

python3 submit/submit.py --mode slurm --script generate \
  --data_name diff_lin_1 diff_ks_cons_1 diff_hyp_diff_1 diff_burgers_1 \
  --num_train_samples 25 \
  --num_valid_samples 250 \
  --num_test_samples 250
