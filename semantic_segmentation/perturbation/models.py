import torch
from torch import nn


class BasePerturbator(nn.Module):
    """
    Base class for perturbation modules.

    This class wraps a segmentation model to apply perturbations to its output logits.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize the BasePerturbator.

        Args:
            model (nn.Module): The base segmentation model.
        """
        super().__init__()
        self.model = model


class SingleClassPerturbator(BasePerturbator):
    """
    Perturbation module that adds noise to a single target class in the model output.

    Gaussian noise is applied to the logits of the target class. The noise can be forced
    to be strictly positive or negative based on the 'positive' flag.
    """
    def __init__(self, model: nn.Module, target_idx: int, pert_ratio: float, positive: bool = True):
        """
        Initialize the SingleClassPerturbator.

        Args:
            model (nn.Module): The base segmentation model.
            target_idx (int): Index of the target class to perturb.
            pert_ratio (float): Standard deviation of the Gaussian noise.
            positive (bool): If True, force noise to be positive; otherwise, negative.
        """
        super().__init__(model)
        self.target_idx = target_idx
        self.pert_ratio = pert_ratio
        self.positive = positive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies perturbation to the target class logits.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits with added perturbation for the target class.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            logits = output['out'] if isinstance(output, dict) else output

        target_logits = logits[:, self.target_idx, :, :]
        noise = torch.normal(mean=0, std=self.pert_ratio, size=target_logits.shape, device=x.device)
        noise = noise.abs() if self.positive else -noise.abs()

        logits[:, self.target_idx, :, :] += noise
        return logits


class MultiClassPerturbator(BasePerturbator):
    """
    Perturbation module that adds noise to multiple specified classes.

    For each class specified in the 'perturbations' list, Gaussian noise is applied to
    the corresponding logits. Each entry in the list should contain an 'index' and an
    optional 'positive' flag.
    """
    def __init__(self, model: nn.Module, perturbations: list, pert_ratio: float):
        """
        Initialize the MultiClassPerturbator.

        Args:
            model (nn.Module): The base segmentation model.
            perturbations (list): List of dictionaries, each with:
                - 'index' (int): The target class index.
                - 'positive' (bool, optional): Whether to use positive noise (default: True).
            pert_ratio (float): Standard deviation of the Gaussian noise.
        """
        super().__init__(model)
        self.perturbations = perturbations
        self.pert_ratio = pert_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies perturbations to multiple classes.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits with added perturbations for specified classes.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            logits = output['out'] if isinstance(output, dict) else output

        for pert in self.perturbations:
            idx = pert['index']
            positive = pert.get('positive', True)
            target_logits = logits[:, idx, :, :]
            noise = torch.normal(mean=0, std=self.pert_ratio, size=target_logits.shape, device=x.device)
            noise = noise.abs() if positive else -noise.abs()
            logits[:, idx, :, :] += noise

        return logits


class AllClassPerturbator(BasePerturbator):
    """
    Perturbation module that adds noise to all classes.

    Gaussian noise is applied to the logits of every class. The noise polarity is controlled
    by the 'positive' flag.
    """
    def __init__(self, model: nn.Module, pert_ratio: float, positive: bool = False):
        """
        Initialize the AllClassPerturbator.

        Args:
            model (nn.Module): The base segmentation model.
            pert_ratio (float): Standard deviation of the Gaussian noise.
            positive (bool): If True, force noise to be positive; otherwise, negative.
        """
        super().__init__(model)
        self.pert_ratio = pert_ratio
        self.positive = positive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies perturbations to all class logits.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits with added perturbations for all classes.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            logits = output['out'] if isinstance(output, dict) else output

        n_classes = logits.shape[1]
        for idx in range(n_classes):
            target_logits = logits[:, idx, :, :]
            noise = torch.normal(mean=0, std=self.pert_ratio, size=target_logits.shape, device=x.device)
            noise = noise.abs() if self.positive else -noise.abs()
            logits[:, idx, :, :] += noise

        return logits
