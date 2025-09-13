import torch
import numpy as np


def dataset_collate_function(batch):
    feature = torch.stack([torch.from_numpy(np.array([data["feature"]], dtype=np.float32)) for data in batch])
    label = torch.from_numpy(np.array([data["label"] for data in batch], dtype=np.int64))
    transformed_batch = {"feature": feature, "label": label}
    return transformed_batch
