#!/bin/bash

gcloud compute instances create \
  preemptible01  \
  --machine-type=e2-medium \
  --zone=us-central1-b  \
  --preemptible \
  --image-project=ubuntu-os-cloud \
  --image-family=ubuntu-2004-lts \
  --no-restart-on-failure \
  --maintenance-policy=terminate \
  --metadata-from-file startup-script=gstartup.sh
