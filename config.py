from easydict import EasyDict as edict
from os.path import join as osp
import os
import json

ROOT = os.getcwd()

with open("./src/config.json", 'r') as f:
    config = json.loads(f.read()) 

__C = edict(config)
cfg = __C
