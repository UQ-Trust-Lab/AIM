#!/usr/bin/env python3
"""
infer_perturbation.py

This module implements a semantic segmentation inference pipeline with support for
perturbation-based modifications. It loads a segmentation model and applies either a
single-class or multi-class perturbation as specified via configuration. The module
processes input images, applies the appropriate preprocessing, and outputs a segmented
image—with optional overlay—saved to disk.

Usage:
    python infer_perturbation.py --cfg <path_to_config_yaml> [additional perturbation options]
"""

import argparse
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision import io, transforms as T
from PIL import Image
import yaml

# Local modules for semantic segmentation and perturbation
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.metrics import *
from tools.perturbation.models import SingleClassPerturbator, MultiClassPerturbator

from rich.console import Console

console = Console()


class SemSeg:
    """
    Semantic segmentation inference with perturbation support.

    This class loads a segmentation model with weights, applies perturbations according
    to the configuration, preprocesses input images, and produces a color-mapped segmentation
    output. The output can optionally be overlayed on the original image.
    """

    def __init__(self, cfg, pert_ratio: float = 0.00, target_only_positive: bool = True) -> None:
        self.pert_ratio = pert_ratio  # Default perturbation ratio (can be overridden by config)

        # Read perturbation configuration from YAML (if available)
        self.perturbation_cfg = cfg.get("PERTURBATION", {})
        self.pert_method = self.perturbation_cfg.get("METHOD", "single")  # Default to 'single'

        # Default target indices (can also be moved entirely to the configuration)
        self.person_idx = 12
        self.car_idx = 20
        self.tree_idx = 4
        self.bicycle_idx = 127
        self.bus_idx = 80
        self.streetlight_idx = 87
        self.trafficlight_idx = 136

        # Set the inference device (e.g., "cpu" or "cuda")
        self.device = torch.device(cfg['DEVICE'])
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES
        self.classes = len(self.labels)

        # Initialize the segmentation model and load weights
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH']))

        # Choose the perturbation method based on the configuration
        if self.pert_method == "single":
            target_idx = self.perturbation_cfg.get("TARGET_IDX", self.trafficlight_idx)
            pert_ratio = self.perturbation_cfg.get("RATIO", self.pert_ratio)
            positive = self.perturbation_cfg.get("POSITIVE", True)
            perturb_model = SingleClassPerturbator(
                model=self.model,
                target_idx=target_idx,
                pert_ratio=pert_ratio,
                positive=positive
            )
        elif self.pert_method == "multi":
            default_targets = [
                {'index': self.trafficlight_idx, 'positive': False},
                {'index': self.car_idx, 'positive': True}
            ]
            target_perturbations = self.perturbation_cfg.get("TARGETS", default_targets)
            perturb_model = MultiClassPerturbator(
                model=self.model,
                perturbations=target_perturbations,
                pert_ratio=pert_ratio
            )
        else:
            raise ValueError(f"Unknown perturbation method: {self.pert_method}")

        self.model = perturb_model.to(self.device)
        self.model.eval()

        # Preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        """
        Preprocess the input image tensor for model inference.

        Scales the image such that its shorter side matches the target size, adjusts dimensions
        to be multiples of 32, and applies normalization.

        Args:
            image (Tensor): Input image tensor.

        Returns:
            Tensor: Preprocessed image tensor.
        """
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H * scale_factor), round(W * scale_factor)

        if image.shape[0] == 4:
            image = image[:3, :, :]  # Retain only RGB channels if an alpha channel exists

        # Ensure dimensions are divisible by the model's stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        image = T.Resize((nH, nW))(image)
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Image.Image:
        """
        Postprocess the model output to generate a color-mapped segmentation image.

        Resizes the segmentation map to the original image size, assigns colors to different
        classes, and optionally overlays the segmentation on the original image.

        Args:
            orig_img (Tensor): The original image tensor.
            seg_map (Tensor): The raw segmentation output from the model.
            overlay (bool): If True, overlays the segmentation on the original image.

        Returns:
            Image.Image: The final segmented PIL image.
        """
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int).squeeze(0)

        # Define colors for each class (RGB)
        yellow = torch.tensor([255, 255, 0], dtype=torch.uint8)  # person
        blue = torch.tensor([0, 0, 255], dtype=torch.uint8)  # car
        green = torch.tensor([0, 255, 0], dtype=torch.uint8)  # tree
        orange = torch.tensor([255, 165, 0], dtype=torch.uint8)  # streetlight
        red = torch.tensor([255, 0, 0], dtype=torch.uint8)  # traffic light
        gray = torch.tensor([128, 128, 128], dtype=torch.uint8)  # bicycle
        purple = torch.tensor([128, 0, 128], dtype=torch.uint8)  # bus

        seg_image = torch.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=torch.uint8)
        seg_image[seg_map == self.person_idx] = yellow
        seg_image[seg_map == self.car_idx] = blue
        seg_image[seg_map == self.tree_idx] = green
        seg_image[seg_map == self.streetlight_idx] = orange
        seg_image[seg_map == self.trafficlight_idx] = red
        seg_image[seg_map == self.bus_idx] = purple

        if orig_img.shape[0] == 4:
            orig_img = orig_img[:3, :, :]

        if overlay:
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        seg_image = seg_image.byte()
        seg_image_pil = Image.fromarray(seg_image.numpy())
        return seg_image_pil

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        """
        Perform a forward pass through the model.

        Args:
            img (Tensor): Preprocessed image tensor.

        Returns:
            Tensor: The raw segmentation map output from the model.
        """
        return self.model(img)

    def predict(self, img_fname: str, overlay: bool) -> Image.Image:
        """
        Run the full inference pipeline on a given image file.

        Args:
            img_fname (str): Path to the input image file.
            overlay (bool): Whether to overlay the segmentation on the original image.

        Returns:
            Image.Image: The final segmented output as a PIL image.
        """
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Segmentation Inference with Perturbation Support."
    )
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml', help="Path to the YAML configuration file.")
    parser.add_argument('--pert_method', type=str, default=None, help="Perturbation method: 'single' or 'multi'.")
    parser.add_argument('--pert_ratio', type=float, default=None, help="Perturbation ratio to override config.")
    parser.add_argument('--pert_target_idx', type=int, default=None, help="Target index for single-class perturbation.")
    parser.add_argument('--pert_positive', action='store_true', help="Use positive perturbation for single-class.")
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    if args.pert_method is not None:
        cfg.setdefault("PERTURBATION", {})["METHOD"] = args.pert_method
    if args.pert_ratio is not None:
        cfg.setdefault("PERTURBATION", {})["RATIO"] = args.pert_ratio
    if args.pert_target_idx is not None:
        cfg.setdefault("PERTURBATION", {})["TARGET_IDX"] = args.pert_target_idx
    if args.pert_positive:
        cfg.setdefault("PERTURBATION", {})["POSITIVE"] = True

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Dataset > [red]{cfg['DATASET']['NAME']}[/red]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define a list of perturbation ratios for testing
    pert_ratio_list = [
        round(cfg['PERTURBATION']['RATIO'] + i * cfg['PERTURBATION']['STEP_SIZE'], 2)
        for i in range(cfg['PERTURBATION']['STEP_LIMIT'])
    ]

    with console.status("[bright_green]Processing..."):
        for pert_ratio in pert_ratio_list:
            console.rule(f"[yellow]Perturbation Ratio: {pert_ratio}[/yellow]")
            semseg = SemSeg(cfg, pert_ratio=pert_ratio)

            if test_file.is_file():
                console.rule(f"[green]{test_file.name}")
                segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
                ratio_dir = save_dir / f"pert_ratio_{pert_ratio}"
                ratio_dir.mkdir(parents=True, exist_ok=True)
                segmap.save(ratio_dir / f"{test_file.stem}.png")
            else:
                for file in test_file.glob('*.*'):
                    console.rule(f"[green]{file.name}")
                    segmap = semseg.predict(str(file), cfg['TEST']['OVERLAY'])
                    ratio_dir = save_dir / f"pert_ratio_{pert_ratio}"
                    ratio_dir.mkdir(parents=True, exist_ok=True)
                    segmap.save(ratio_dir / f"{file.stem}.png")

    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")


if __name__ == '__main__':
    main()
