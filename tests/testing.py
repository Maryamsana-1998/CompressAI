# %%
import compressai
import importlib
import json
import os
import random
import numpy as np
import torch
from pathlib import Path
import json
import glob 

eval_model = importlib.import_module("compressai.utils.video.eval_model.__main__")

# Derive the directory containing the notebook file
checkpoint = "checkpoints/checkpoint_best_loss_8.pth.tar"
assert os.path.isfile(checkpoint)
model = 'ssf2020'
metric = 'mse'
here = Path('tmp/out.bin')

# %%
net = eval_model.load_checkpoint(model, False, checkpoint)

videos = glob.glob('/data/maryam.sana/UVG/*.yuv')
print(len(videos))

for video in videos:
    try:
        result = eval_model.eval_model(net, Path(video),here)
        json_file_path = str(Path(video).stem) + '.json'
        with open(json_file_path, 'w') as json_file:
            json.dump(result, json_file, indent=4)
        print(video, 'processed')
    except Exception as e:
        print(video,e)


