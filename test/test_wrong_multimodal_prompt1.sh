#!/bin/bash

source /home/riggi/Software/venvs/maasai/bin/activate

INPUTIMG="test/lena.png"
python scripts/run.py --config='config/litellm_config.yaml' --query "Check whether this is an astronomy-related image." --input_imgs $INPUTIMG
