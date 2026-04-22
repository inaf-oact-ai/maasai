#!/bin/bash

source /home/riggi/Software/venvs/maasai/bin/activate

INPUTIMG="test/galaxy0001.fits,test/galaxy0002.fits"
python scripts/run.py --config='config/litellm_config.yaml' --query "Do you see any radio galaxy in the attached images?" --input_imgs $INPUTIMG
