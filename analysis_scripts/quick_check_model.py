#!/usr/bin/env python3
import torch
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

model_path = os.path.join(project_root, "model/open_set_gee/exclude_10/gating_network.pt")
w = torch.load(model_path, map_location='cpu')
print(f"Keys: {list(w.keys())}")
if 'network.0.weight' in w:
    print(f"Weight shape: {w['network.0.weight'].shape}")
