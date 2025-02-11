# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
dir_path = osp.join(this_dir, '..')
print(f"adding {dir_path} to path")
add_path(dir_path)

lib_path = osp.join(this_dir, '..', 'lib')
print(f"adding {lib_path} to path")
add_path(lib_path)

lib_path = osp.join(this_dir, '..', 'ycb_render')
print(f"adding {lib_path} to path")
add_path(lib_path)
