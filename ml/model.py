import multiprocessing
import numpy as np

import datasets
import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import Counter

from ml.dataset import dataset_collate_function


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha can be a float for binary classification, or a tensor for multi-class
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_term = (1-pt)**self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # If alpha is a tensor, gather the weights for each sample
                alpha_t = self.alpha.gather(0, targets.data.view(-1))
                focal_loss = alpha_t * focal_term * ce_loss
            else:
                # If alpha is a scalar
                focal_loss = self.alpha * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // r, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)


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
        use_attention=False,
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
        self.use_attention = use_attention

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
        
        if self.use_attention:
            self.se = SEBlock(out_channels)

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

        if self.use_attention:
            out = self.se(out)

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
        use_attention=False,
        validation_split=0.1,
        loss_type='cross_entropy',  # New parameter
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
                use_attention=self.hparams.use_attention,
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

        # Setup loss function
        if self.hparams.loss_type == 'focal_loss':
            # Calculate class weights for alpha parameter in Focal Loss
            train_labels = self.train_dataset.dataset.tensors[1][self.train_dataset.indices]
            class_counts = Counter(train_labels.tolist())
            
            # Compute weights using inverse frequency, with a safe guard for unseen classes
            weights = torch.tensor([1.0 / class_counts[i] if i in class_counts and class_counts[i] > 0 else 1.0 for i in range(self.hparams.output_dim)], dtype=torch.float32)
            
            # The alpha in FocalLoss needs to be on the same device as the model
            # We register it as a buffer so it's automatically moved to the correct device
            self.register_buffer('alpha_weights', weights)
            
            self.criterion = FocalLoss(alpha=self.alpha_weights, gamma=2, reduction='mean')
        else: # Default to cross_entropy
            self.criterion = nn.CrossEntropyLoss()

    def train_dataloader(self):
        try:
            num_workers = multiprocessing.cpu_count()
        except:
            num_workers = 1
        
        return DataLoader(
            self.train_dataset,
            batch_size=16,
            num_workers=num_workers,
            shuffle=True,
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
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / (len(y) * 1.0))
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        entropy = self.criterion(y_hat, y)
        self.log(
            "training_loss",
            entropy,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return entropy
