#!/bin/bash

declare -a files=(
  'test_grad.py'
  'test_hessian.py'
  'test_binary_classif_mlp_log_lik.py'
  'test_binary_classif_mlp_log_target_deriv.py'
  'test_multitask_classif_mlp_log_lik.py'
)

for file in "${files[@]}"
do
   echo -e "\nRunning tests in tests/$file..."
   python -m unittest tests/$file
done
