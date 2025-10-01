import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import click
from ruamel.yaml import YAML
import torch
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import pyarrow.parquet as pq
import numpy as np
from collections import Counter
import os

# Make sure the ml module is in the python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import MixtureOfExperts, ResNet

class MoEDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_path = config['data_path']
        self.minority_classes = config['minority_classes']
        self.batch_size = config.get('batch_size', 16)
        self.validation_split = config.get('validation_split', 0.1)
        self.num_workers = config.get('num_workers', 4)

    def setup(self, stage=None):
        # Load full dataset from the training path
        train_path = os.path.join(self.data_path, 'train.parquet')
        table = pq.read_table(train_path)
        df = table.to_pandas()
        
        features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
        labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))
        
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        full_dataset = TensorDataset(features, labels)
        
        # Split dataset
        val_size = int(len(full_dataset) * self.validation_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create sampler for the gating network training
        train_labels = self.train_dataset.dataset.tensors[1][self.train_dataset.indices]
        
        # Create meta-labels: 0 for majority, 1 for minority
        is_minority = torch.isin(train_labels, torch.tensor(self.minority_classes)).long()
        meta_class_counts = Counter(is_minority.tolist())
        
        if len(meta_class_counts) < 2:
            print("Warning: Only one meta-class (majority/minority) present in the training split. Using random shuffling instead of weighted sampling.")
            self.sampler = None
            return

        # Weight is inverse of meta-class frequency
        meta_class_weights = {i: 1.0 / count for i, count in meta_class_counts.items()}
        
        sample_weights = torch.tensor([meta_class_weights[label.item()] for label in is_minority], dtype=torch.float)
        
        self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    def train_dataloader(self):
        # If sampler is created, shuffle must be False
        shuffle = self.sampler is None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=shuffle,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

@click.command()
@click.option('--config', '-c', 'config_path', required=True, type=click.Path(exists=True), help='Path to the MoE YAML config file.')
def main(config_path):
    # Load config using ruamel.yaml
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    # Load pre-trained experts
    print(f"Loading Generalist expert from: {config['generalist_expert_path']}")
    generalist_expert = ResNet.load_from_checkpoint(config['generalist_expert_path'])
    print(f"Loading Minority expert from: {config['minority_expert_path']}")
    minority_expert = ResNet.load_from_checkpoint(config['minority_expert_path'])

    # Instantiate the main MoE model
    moe_model = MixtureOfExperts(
        generalist_expert=generalist_expert,
        minority_expert=minority_expert,
        num_total_classes=config['num_total_classes'],
        majority_classes=config['majority_classes'],
        minority_classes=config['minority_classes']
    )

    # --- Freeze experts for Phase 2: Gating Network Training ---
    print("Freezing expert weights...")
    for param in moe_model.generalist_expert.parameters():
        param.requires_grad = False
    for param in moe_model.minority_expert.parameters():
        param.requires_grad = False
    print("Expert weights frozen.")
    # Verify that only gating network parameters are trainable
    trainable_params = sum(p.numel() for p in moe_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    # -----------------------------------------------------------

    # Instantiate the DataModule
    data_module = MoEDataModule(config)

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='model',
        filename='moe_gate',
        save_top_k=1,
        mode='min',
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min")

    # Configure and run trainer
    trainer = Trainer(
        max_epochs=config.get('max_epochs', 30),
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices='auto'
    )
    
    print("--- Starting MoE Gating Network Training ---")
    trainer.fit(moe_model, datamodule=data_module)
    print("--- MoE Gating Network Training Finished ---")


if __name__ == '__main__':
    main()
