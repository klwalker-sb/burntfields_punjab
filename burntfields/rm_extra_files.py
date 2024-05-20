#!/usr/bin/env python
# coding: utf-8
# %%
import os
import shutil

keepers = []
for img in os.listdir(in_dir, tmp_dir):
    if img.endswith('AnalyticMS_SR_clip.tif'):
        keepname = str(os.path.basename(f)[:21])
        keepers.append(keepname)
for any in os.listdir(in_dir):
    name = str(os.path.basename(any)[:21])
    file = os.path.join(in_dir,any)
    if name not in keepers:
        shutil move file os.path.join(tmp_dir,any)

