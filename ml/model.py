import multiprocessing
import numpy as np

import datasets
import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter


from ml.dataset import dataset_collate_function





class CNN(LightningModule):
    def __init__(
        self,
        c1_output_dim,
        c1_kernel_size,
        c1_stride,
        c2_output_dim,
        c2_kernel_size,
        c2_stride,
        output_dim,
        data_path,
        signal_length,
    ):
        super().__init__()
        # save parameters to checkpoint
        self.save_hyperparameters()

        # two convolution, then one max pool
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hparams.c1_output_dim,
                kernel_size=self.hparams.c1_kernel_size,
                stride=self.hparams.c1_stride,
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hparams.c1_output_dim,
                out_channels=self.hparams.c2_output_dim,
                kernel_size=self.hparams.c2_kernel_size,
                stride=self.hparams.c2_stride,
            ),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2)

        # flatten, calculate the output size of max pool
        # use a dummy input to calculate
        dummy_x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        # followed by 5 dense layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=max_pool_out, out_features=200),
            nn.Dropout(p=0.05),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=100), nn.Dropout(p=0.05), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.Dropout(p=0.05), nn.ReLU()
        )

        # finally, output layer
        self.out = nn.Linear(in_features=50, out_features=self.hparams.output_dim)

    def forward(self, x):
        # make sure the input is in [batch_size, channel, signal_length]
        # where channel is 1
        # signal_length is 1500 by default
        batch_size = x.shape[0]

        # 2 conv 1 max
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = x.reshape(batch_size, -1)

        # 3 fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # output
        x = self.out(x)

        return x

    def train_dataloader(self):
        import os
        import torch
        import pyarrow.parquet as pq
        from torch.utils.data import TensorDataset

        train_path = os.path.join(self.hparams.data_path, 'train.parquet')
        table = pq.read_table(train_path)
        df = table.to_pandas()
        
        features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
        labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))
        
        # The CNN model expects a channel dimension, so we add it.
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        dataset = TensorDataset(features, labels)

        try:
            num_workers = multiprocessing.cpu_count()
        except:
            num_workers = 1
        
        # We no longer need a collate_fn because TensorDataset handles it.
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            num_workers=num_workers,
            shuffle=True,
        )

        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        entropy = F.cross_entropy(y_hat, y)
        self.log(
            "training_loss",
            entropy,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        loss = {"loss": entropy}

        return loss


class CustomConv1d(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(CustomConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class CustomMaxPool1d(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(CustomMaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class GatingNetwork(nn.Module):
    """
    A lightweight CNN to act as a router for the Mixture of Experts model.
    It performs a binary classification to decide which expert to use.
    """
    def __init__(self, signal_length=1500):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the output size after conv layers
        dummy_x = torch.rand(1, 1, signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        flattened_size = dummy_x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 100)
        self.out = nn.Linear(100, 2) # Output for 2 experts (Generalist vs. Specialist)

    def forward(self, x):
        # Add channel dimension if not present, which is expected by Conv1d
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        downsample,
        use_bn,
        use_do,
        is_first_block=False,
    ):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = CustomConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = CustomConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups,
        )

        self.max_pool = CustomMaxPool1d(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1d(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        stride,
        groups,
        n_block,
        n_classes,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False,
    ):
        super(ResNet1d, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = CustomConv1d(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=self.kernel_size,
            stride=1,
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(
                    base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap)
                )
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block,
            )
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out = x

        # first conv
        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print(
                    "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                        i_block, net.in_channels, net.out_channels, net.downsample
                    )
                )
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)

        out = out.mean(-1)
        if self.verbose:
            print("final pooling", out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print("dense", out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print("softmax", out.shape)

        return out


class ResNet(LightningModule):
    def __init__(
        self,
        c1_output_dim,
        c1_kernel_size,
        c1_stride,
        c1_groups,
        c1_n_block,
        output_dim,
        data_path,
        signal_length,
        validation_split=0.1,
        loss_type='cross_entropy',
        sampling_strategy='random', # New parameter
    ):
        super().__init__()
        # save parameters to checkpoint
        self.save_hyperparameters()

        # two convolution, then one max pool
        self.conv1 = nn.Sequential(
            ResNet1d(
                in_channels=1,
                base_filters=self.hparams.c1_output_dim,
                kernel_size=self.hparams.c1_kernel_size,
                stride=self.hparams.c1_stride,
                groups=self.hparams.c1_groups,
                n_block=self.hparams.c1_n_block,
                n_classes=self.hparams.c1_output_dim,
            ),
            nn.ReLU(),
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2)

        # flatten, calculate the output size of max pool
        # use a dummy input to calculate
        dummy_x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        # followed by 5 dense layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=max_pool_out, out_features=200),
            nn.Dropout(p=0.05),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=100), nn.Dropout(p=0.05), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.Dropout(p=0.05), nn.ReLU()
        )

        # finally, output layer
        self.out = nn.Linear(in_features=50, out_features=self.hparams.output_dim)

    def forward(self, x):
        # make sure the input is in [batch_size, channel, signal_length]
        # where channel is 1
        # signal_length is 1500 by default
        batch_size = x.shape[0]

        # 1 conv 1 max
        x = self.conv1(x)
        x = self.max_pool(x)

        x = x.reshape(batch_size, -1)

        # 3 fc
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # output
        x = self.out(x)

        return x

    def setup(self, stage=None):
        # This is called by the Trainer before fitting
        import os
        import torch
        import pyarrow.parquet as pq
        from torch.utils.data import TensorDataset, random_split
        from collections import Counter

        train_path = os.path.join(self.hparams.data_path, 'train.parquet')
        table = pq.read_table(train_path)
        df = table.to_pandas()
        
        features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
        labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))
        
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        full_dataset = TensorDataset(features, labels)
        
        # Split dataset
        val_size = int(len(full_dataset) * self.hparams.validation_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        # Calculate weights for class-aware sampling
        if self.hparams.sampling_strategy == 'class_aware':
            train_labels = self.train_dataset.dataset.tensors[1][self.train_dataset.indices]
            class_counts = Counter(train_labels.tolist())
            
            # Compute weight for each sample. The weight is the inverse of its class frequency.
            class_weights = {c: 1.0 / count for c, count in class_counts.items()}
            self.train_sample_weights = torch.tensor([class_weights[label.item()] for label in train_labels], dtype=torch.float)
        else:
            self.train_sample_weights = None

    def train_dataloader(self):
        try:
            num_workers = multiprocessing.cpu_count()
        except:
            num_workers = 1

        sampler = None
        shuffle = True
        if self.hparams.sampling_strategy == 'class_aware' and self.train_sample_weights is not None:
            sampler = WeightedRandomSampler(
                self.train_sample_weights,
                num_samples=len(self.train_sample_weights),
                replacement=True
            )
            shuffle = False # Sampler is mutually exclusive with shuffle
        
        return DataLoader(
            self.train_dataset,
            batch_size=16,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        try:
            num_workers = multiprocessing.cpu_count()
        except:
            num_workers = 1

        return DataLoader(
            self.val_dataset,
            batch_size=16,
            num_workers=num_workers,
            shuffle=False,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / (len(y) * 1.0))
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=3, # Scheduler patience is different from EarlyStopping patience
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        entropy = F.cross_entropy(y_hat, y)
        self.log(
            "training_loss",
            entropy,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return entropy

class MixtureOfExperts(LightningModule):
    def __init__(self, generalist_expert, minority_expert, num_total_classes, majority_classes, minority_classes):
        super().__init__()
        self.save_hyperparameters(ignore=['generalist_expert', 'minority_expert'])

        self.generalist_expert = generalist_expert
        self.minority_expert = minority_expert
        self.gating_network = GatingNetwork()

        # Mappings for labels
        self.num_total_classes = num_total_classes
        self.majority_classes = sorted(majority_classes)
        self.minority_classes = sorted(minority_classes)

        # Create reverse mappings from expert-local output index to global class index
        self.majority_map = {i: global_idx for i, global_idx in enumerate(self.majority_classes)}
        self.minority_map = {i: global_idx for i, global_idx in enumerate(self.minority_classes)}

    def forward(self, x):
        # Gating network decides which expert to use
        gate_logits = self.gating_network(x)
        
        # For inference/validation, we might want to route to the chosen expert
        # For training the gate, we just need the gate's output.
        # The training_step will handle the logic.
        return gate_logits

    def training_step(self, batch, batch_idx):
        x, y_global = batch
        
        # 1. Get gating network's prediction
        gate_logits = self.gating_network(x)

        # 2. Determine the "true" expert for each sample
        # Create a tensor of minority classes for efficient lookup
        minority_classes_tensor = torch.tensor(self.minority_classes, device=y_global.device)
        # y_meta is 1 if the class is in minority_classes, 0 otherwise
        y_meta = torch.isin(y_global, minority_classes_tensor).long()

        # 3. Calculate loss for the gating network
        gate_loss = F.cross_entropy(gate_logits, y_meta)
        self.log('train_gate_loss', gate_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return gate_loss

    def validation_step(self, batch, batch_idx):
        x, y_global = batch
        
        # Gate output
        gate_logits = self.gating_network(x)
        gate_preds = torch.argmax(gate_logits, dim=1)

        # Meta labels
        minority_classes_tensor = torch.tensor(self.minority_classes, device=y_global.device)
        y_meta = torch.isin(y_global, minority_classes_tensor).long()

        # Gate validation loss and accuracy
        val_gate_loss = F.cross_entropy(gate_logits, y_meta)
        val_gate_acc = torch.sum(gate_preds == y_meta).item() / (len(y_meta) * 1.0)
        
        self.log('val_loss', val_gate_loss, prog_bar=True) # val_loss is monitored by callbacks
        self.log('val_gate_acc', val_gate_acc, prog_bar=True)

        # --- Full MoE validation logic (for later phases) ---
        # This part combines expert outputs for a full prediction
        with torch.no_grad():
            # Get expert predictions
            # Unsqueeze(1) is needed if experts expect a channel dimension
            if len(x.shape) == 2:
                 x_exp = x.unsqueeze(1)
            else:
                 x_exp = x
            generalist_logits = self.generalist_expert(x_exp)
            minority_logits = self.minority_expert(x_exp)

            # Combine predictions based on gate
            final_logits = torch.zeros((x.shape[0], self.num_total_classes), device=x.device)
            
            for i in range(x.shape[0]):
                if gate_preds[i] == 0: # Route to generalist
                    for local_idx, global_idx in self.majority_map.items():
                        final_logits[i, global_idx] = generalist_logits[i, local_idx]
                else: # Route to minority
                    for local_idx, global_idx in self.minority_map.items():
                        final_logits[i, global_idx] = minority_logits[i, local_idx]

            moe_preds = torch.argmax(final_logits, dim=1)
            val_moe_acc = torch.sum(moe_preds == y_global).item() / (len(y_global) * 1.0)
            self.log('val_moe_acc', val_moe_acc, prog_bar=True)

        return val_gate_loss

    def configure_optimizers(self):
        # By default, this will grab all parameters in the module
        # We will freeze the experts in the training script
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
