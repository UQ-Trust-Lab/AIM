#!/usr/bin/env python3
"""
This script evaluates a semantic segmentation model on a validation dataset.
Evaluation is performed over a range of perturbation ratios on the "person" class.
For each ratio, the evaluation results are saved to a text file.
"""

import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Import models, datasets, augmentations, metrics, and utilities
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from tools.perturbation.models import SingleClassPerturbator, MultiClassPerturbator, AllClassPerturbator


def save_results_to_txt(results, file_path):
    """
    Save the evaluation results to a text file.

    Args:
        results (list of str): List of result strings.
        file_path (Path): Path to the text file.
    """
    with open(file_path, 'w') as f:
        for result in results:
            f.write(result + '\n')


@torch.no_grad()
def evaluate(model, dataloader, device, perturb_model=None, person_class_idx=None):
    """
    Evaluate the model on the validation set.

    Args:
        model: The segmentation model.
        dataloader: DataLoader for the validation dataset.
        device: Device for computation.
        perturb_model: (Optional) A perturbation model to use.
        person_class_idx: (Optional) Index of the "person" class.

    Returns:
        Tuple: (acc, macc, f1, mf1, ious, miou, person_acc, person_iou)
    """
    print('Evaluating...')
    model.eval()
    if perturb_model is not None:
        perturb_model.eval()

    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        with autocast():
            if perturb_model is not None:
                logits = perturb_model(images)
            else:
                logits = model(images)
            preds = logits.softmax(dim=1)
        metrics.update(preds, labels)

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    if person_class_idx is not None:
        person_iou = ious[person_class_idx]
        person_acc = acc[person_class_idx]
    else:
        person_iou = None
        person_acc = None

    return acc, macc, f1, mf1, ious, miou, person_acc, person_iou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip, perturb_model=None, person_class_idx=None):
    """
    Evaluate the model using multi-scale and flip augmentations.

    Args:
        model: The segmentation model.
        dataloader: DataLoader for the validation dataset.
        device: Device for computation.
        scales (list): List of scales.
        flip (bool): Whether to use horizontal flip.
        perturb_model: (Optional) A perturbation model to use.
        person_class_idx: (Optional) Index of the "person" class.

    Returns:
        Tuple: (acc, macc, f1, mf1, ious, miou, person_acc, person_iou)
    """
    model.eval()
    if perturb_model is not None:
        perturb_model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader, desc="Evaluating Multi-scale"):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            # Ensure dimensions are multiples of 32
            new_H = int(math.ceil(new_H / 32)) * 32
            new_W = int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)

            with autocast():
                if perturb_model is not None:
                    logits = perturb_model(scaled_images)
                else:
                    logits = model(scaled_images)
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

                if flip:
                    scaled_images_flipped = torch.flip(scaled_images, dims=(3,))
                    if perturb_model is not None:
                        logits_flipped = perturb_model(scaled_images_flipped)
                    else:
                        logits_flipped = model(scaled_images_flipped)
                    logits_flipped = torch.flip(logits_flipped, dims=(3,))
                    logits_flipped = F.interpolate(logits_flipped, size=(H, W), mode='bilinear', align_corners=True)
                    scaled_logits += logits_flipped.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()

    if person_class_idx is not None:
        person_iou = ious[person_class_idx]
        person_acc = acc[person_class_idx]
    else:
        person_iou = None
        person_acc = None

    return acc, macc, f1, mf1, ious, miou, person_acc, person_iou


def main(cfg):
    """
    Main function to perform evaluation using various perturbation ratios.
    """
    device = torch.device(cfg['DEVICE'])
    eval_cfg = cfg['EVAL']

    # Set up validation transformation and dataset
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists():
        model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating model from {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path)))
    model = model.to(device)

    class_names = dataset.CLASSES

    # Define the perturbation ratio list (0.0 to 20.0 with a step of 0.5)
    pert_ratio_list = [round(x * 0.5, 1) for x in range(0, 41)]

    # Find the index for the "person" class
    person_class_idx = None
    for idx, name in enumerate(class_names):
        if name.lower() == 'person':
            person_class_idx = idx
            break

    if person_class_idx is None:
        raise ValueError("Class 'person' not found in the class names.")

    for pert_ratio in pert_ratio_list:
        ratio_folder = Path(cfg['SAVE_DIR']) / f"pert_ratio_{pert_ratio}"
        txt_file_path = ratio_folder / 'evaluation_results.txt'
        ratio_folder.mkdir(parents=True, exist_ok=True)

        if pert_ratio == 0.0:
            current_perturb_model = None  # Use the original model
            print(f"\nEvaluating with perturbation ratio: {pert_ratio} (Original Model)")
        else:
            current_perturb_model = SingleClassPerturbator(
                model=model,
                target_idx=person_class_idx,
                pert_ratio=pert_ratio,
                positive=True
            ).to(device)
            print(f"\nEvaluating with perturbation ratio: {pert_ratio} (SingleClassPerturbator)")

        if eval_cfg.get('MSF', {}).get('ENABLE', False):
            acc, macc, f1, mf1, ious, miou, person_acc, person_iou = evaluate_msf(
                model,
                dataloader,
                device,
                scales=eval_cfg['MSF'].get('SCALES', []),
                flip=eval_cfg['MSF'].get('FLIP', False),
                perturb_model=current_perturb_model,
                person_class_idx=person_class_idx
            )
        else:
            acc, macc, f1, mf1, ious, miou, person_acc, person_iou = evaluate(
                model,
                dataloader,
                device,
                perturb_model=current_perturb_model,
                person_class_idx=person_class_idx
            )

        # Prepare and save evaluation results
        result_txt = f"Perturbation Ratio: {pert_ratio}\n"
        for idx_class, class_name in enumerate(class_names):
            result_txt += f"  {class_name} - IoU: {ious[idx_class]:.4f}, F1: {f1[idx_class]:.4f}, Acc: {acc[idx_class]:.4f}\n"
        result_txt += f"  Mean IoU: {miou:.4f}, Mean F1: {mf1:.4f}, Mean Acc: {macc:.4f}\n"
        result_txt += f"  Person IoU: {person_iou:.4f}, Person Acc: {person_acc:.4f}\n"

        save_results_to_txt([result_txt], txt_file_path)
        print(result_txt)

    print(f"All evaluation results saved in {cfg['SAVE_DIR']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate semantic segmentation model with perturbations."
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default='configs/custom.yaml',
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
