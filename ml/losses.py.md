import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMacroF1Loss(nn.Module):
    """
    A differentiable approximation of Macro-F1 Loss.

    This loss computes a soft version of precision, recall, and F1 score
    for each class, then averages them across all classes. This forces the
    model to pay equal attention to all classes, regardless of their frequency.
    """

    def __init__(self, num_classes, epsilon=1e-8):
        """
        Args:
            num_classes (int): Total number of classes
            epsilon (float): Small constant to avoid division by zero
        """
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model outputs of shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Soft Macro-F1 loss (scalar)
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)  # [batch_size, num_classes]

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Initialize per-class metrics
        f1_per_class = []

        # Calculate F1 for each class separately
        for class_idx in range(self.num_classes):
            # Extract predictions and targets for this class
            class_probs = probs[:, class_idx]  # [batch_size]
            class_targets = targets_one_hot[:, class_idx]  # [batch_size]

            # Calculate TP, FP, FN for this class
            tp = torch.sum(class_probs * class_targets)
            fp = torch.sum(class_probs * (1 - class_targets))
            fn = torch.sum((1 - class_probs) * class_targets)

            # Calculate precision, recall, F1 for this class
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1 = 2 * precision * recall / (precision + recall + self.epsilon)

            f1_per_class.append(f1)

        # Stack and calculate Macro-F1 (average across all classes)
        f1_per_class = torch.stack(f1_per_class)
        macro_f1 = torch.mean(f1_per_class)

        # Return loss (1 - macro_f1 to minimize)
        return 1.0 - macro_f1


class CombinedMacroF1CELoss(nn.Module):
    """
    Combined loss that merges CrossEntropyLoss and SoftMacroF1Loss.

    This allows the model to optimize for both accuracy (via CE) and
    balanced performance across all classes (via Macro-F1).
    """

    def __init__(self, num_classes, lambda_macro=0.5, epsilon=1e-8):
        """
        Args:
            num_classes (int): Total number of classes
            lambda_macro (float): Weight for Macro-F1 loss (0 to 1)
            epsilon (float): Small constant to avoid division by zero
        """
        super().__init__()
        self.lambda_macro = lambda_macro
        self.ce_loss = nn.CrossEntropyLoss()
        self.macro_f1_loss = SoftMacroF1Loss(num_classes, epsilon)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model outputs of shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels of shape (batch_size,)

        Returns:
            dict: Dictionary containing total loss and individual components
        """
        ce_loss = self.ce_loss(logits, targets)
        macro_f1_loss = self.macro_f1_loss(logits, targets)

        total_loss = self.lambda_macro * macro_f1_loss + (1 - self.lambda_macro) * ce_loss

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'macro_f1_loss': macro_f1_loss
        }


class AdaptiveMacroF1CELoss(nn.Module):
    """
    Adaptive version that dynamically adjusts the weight of Macro-F1 loss
    during training. Starts with more emphasis on CE (accuracy) and gradually
    increases focus on Macro-F1 (balanced performance).
    """

    def __init__(self, num_classes, initial_lambda=0.1, final_lambda=0.7, total_epochs=100, epsilon=1e-8):
        """
        Args:
            num_classes (int): Total number of classes
            initial_lambda (float): Starting weight for Macro-F1 loss
            final_lambda (float): Final weight for Macro-F1 loss
            total_epochs (int): Total training epochs for schedule calculation
            epsilon (float): Small constant to avoid division by zero
        """
        super().__init__()
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.total_epochs = total_epochs
        self.ce_loss = nn.CrossEntropyLoss()
        self.macro_f1_loss = SoftMacroF1Loss(num_classes, epsilon)

    def get_current_lambda(self, current_epoch):
        """Calculate current lambda based on linear schedule."""
        progress = min(1.0, current_epoch / self.total_epochs)
        return self.initial_lambda + (self.final_lambda - self.initial_lambda) * progress

    def forward(self, logits, targets, current_epoch=0):
        """
        Args:
            logits (torch.Tensor): Model outputs of shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels of shape (batch_size,)
            current_epoch (int): Current training epoch for adaptive scheduling

        Returns:
            dict: Dictionary containing total loss, individual components, and current lambda
        """
        current_lambda = self.get_current_lambda(current_epoch)

        ce_loss = self.ce_loss(logits, targets)
        macro_f1_loss = self.macro_f1_loss(logits, targets)

        total_loss = current_lambda * macro_f1_loss + (1 - current_lambda) * ce_loss

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'macro_f1_loss': macro_f1_loss,
            'current_lambda': current_lambda
        }