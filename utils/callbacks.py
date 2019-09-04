import torch
import numpy as np

from catalyst.dl import MetricCallback
from catalyst.utils import get_activation_fn


class DiceCallback(MetricCallback):
    """
    Dice metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "dice",
            eps: float = 1e-7,
            threshold: float = None,
            activation: str = "Sigmoid"
    ):
        """
        :param input_key: input key to use for dice calculation;
            specifies our `y_true`.
        :param output_key: output key to use for dice calculation;
            specifies our `y_pred`.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=self.dice,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    def dice(self, outputs, targets, eps=1e-7, threshold=0.5,
             activation='Sigmoid'):
        activation_fn = get_activation_fn(activation)
        outputs = activation_fn(outputs)

        if threshold is not None:
            outputs = (outputs > threshold).float()

        batch_size = len(targets)
        outputs = outputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = torch.sum(targets * outputs, dim=1)
        union = torch.sum(targets, dim=1) + torch.sum(outputs, dim=1)
        dice = (2 * intersection / (union + eps)).cpu().numpy()

        result = []
        for i, d in enumerate(dice):
            if d >= eps:
                result.append(d)
                continue

            s = torch.sum(targets[i]).cpu().numpy()
            result.append(1 if s < eps else d)

        return np.mean(result)


class IouCallback(MetricCallback):
    """
    IoU (Jaccard) metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "iou",
            eps: float = 1e-7,
            threshold: float = None,
            activation: str = "Sigmoid",
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        super().__init__(
            prefix=prefix,
            metric_fn=self.iou,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    def iou(self, outputs: torch.Tensor,
            targets: torch.Tensor,
            eps: float = 1e-7,
            threshold: float = None,
            activation: str = "Sigmoid"
            ):
        """
        Args:
            outputs (torch.Tensor): A list of predicted elements
            targets (torch.Tensor):  A list of elements that are to be predicted
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ["none", "Sigmoid", "Softmax2d"]

        Returns:
            float: IoU (Jaccard) score
        """
        activation_fn = get_activation_fn(activation)
        outputs = activation_fn(outputs)

        if threshold is not None:
            outputs = (outputs > threshold).float()

        batch_size = len(targets)
        outputs = outputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)

        intersection = torch.sum(targets * outputs, dim=1)
        union = torch.sum(targets, dim=1) + torch.sum(outputs, dim=1)
        iou = (intersection / (union - intersection + eps)).cpu().numpy()

        result = []
        for i, d in enumerate(iou):
            if d >= eps:
                result.append(d)
                continue

            s = torch.sum(targets[i]).cpu().numpy()
            result.append(1 if s < eps else d)

        return np.mean(result)


class AccuracyCallback(MetricCallback):
    """
        IoU (Jaccard) metric callback.
    """
    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "accuracy",
            eps: float = 1e-7,
            threshold: float = None,
            activation: str = "Sigmoid",
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        super().__init__(
            prefix=prefix,
            metric_fn=self.accuracy,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    @staticmethod
    def accuracy(outputs: torch.Tensor,
                   targets: torch.Tensor,
                   eps: float = 1e-7,
                   threshold: float = None,
                   activation: str = "Sigmoid"):
        activation_fn = get_activation_fn(activation)
        outputs = activation_fn(outputs)

        if threshold is not None:
            outputs = (outputs > threshold).float()

        eq = (outputs == targets).long().float().sum()
        return eq / (targets.view(-1).size(0) + eps)
