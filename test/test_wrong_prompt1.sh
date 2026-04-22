#!/bin/bash

source /home/riggi/Software/venvs/maasai/bin/activate

python scripts/run.py --config='config/litellm_config.yaml' --query="Potresti dirmi che tempo fa a Roma?"
