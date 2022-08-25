#!/bin/bash

declare -a files=(
  'test_binary_classif_mlp221_log_lik.py'
  'test_binary_classif_mlp221_log_target_deriv.py'
  'test_binary_classif_mlp2321_log_lik.py'
  'test_gibbs_blocking.py'
  'test_grad.py'
  'test_multiclass_classif_mlp4323_log_lik.py'
  'test_multiclass_classif_mlp433_log_lik.py'
)

for file in "${files[@]}"
do
   echo -e "\nRunning tests in tests/$file..."
   python -m unittest tests/$file
done
