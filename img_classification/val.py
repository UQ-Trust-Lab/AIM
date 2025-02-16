import argparse
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

# Import custom models
from model.resnet import cifar100_resnet56, cifar10_resnet56
from model.vgg import cifar100_vgg11_bn, cifar10_vgg11_bn
from model.mobilenetv2 import cifar10_mobilenetv2_x0_5, cifar100_mobilenetv2_x0_5
from model.shufflenetv2 import cifar10_shufflenetv2_x0_5, cifar100_shufflenetv2_x0_5
from model.vit import cifar10_vit_b16, cifar100_vit_b16

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gaussian Noise Perturbation on Pre-trained Models"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar100',
        choices=['cifar100', 'cifar10'],
        help="Select the dataset: 'cifar100' or 'cifar10'"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='resnet56',
        choices=['resnet56', 'vgg11', 'mobilenetv2', 'shufflenetv2', 'vit_b16'],
        help="Select the pre-trained model: 'resnet56', 'vgg11', 'mobilenetv2', 'shufflenetv2', or 'vit_b16'"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="Batch size for data loading (default: 64)"
    )
    parser.add_argument(
        '--num_perturb_steps',
        type=int,
        default=101,
        help="Number of perturbation steps (e.g., 101 steps correspond to standard deviations: 0.0, 0.2, ..., 20.0)"
    )
    parser.add_argument(
        '--perturb_step',
        type=float,
        default=0.2,
        help="Step size for perturbation standard deviation (default: 0.2)"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='results.txt',
        help="File path to save the results"
    )
    parser.add_argument(
         '--plot',
         action='store_true',
         help="If set, display the plot of accuracy vs perturbation standard deviation."
    )
    parser.add_argument(
         '--save_plot',
         action='store_true',
         help="If set, save the plot as 'accuracy_vs_std.png'."
    )
    return parser.parse_args()


def check_accuracy(model, dataloader, device):
    total_samples = len(dataloader.dataset)
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()

    accuracy = total_correct / total_samples
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


class PerturbModel(nn.Module):
    def __init__(self, model, pert_ratio):
        super(PerturbModel, self).__init__()
        self.model = model
        self.pert_ratio = pert_ratio

    def forward(self, x):
        logits = self.model(x)
        noise_std = self.pert_ratio
        noise = torch.normal(mean=0.0, std=noise_std, size=logits.size()).to(x.device)
        logits_noisy = logits + noise
        return logits_noisy


def plot_results(std_list, acc_list, do_plot, do_save):
    plt.figure(figsize=(8, 6))
    plt.plot(std_list, acc_list, marker='o', linestyle='-')
    plt.xlabel("Standard Deviation (std)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Perturbation Standard Deviation")
    plt.grid(True)
    if do_save:
        plt.savefig("accuracy_vs_std.png")
        print("Plot saved as 'accuracy_vs_std.png'.")
    if do_plot:
        plt.show()


def main():
    args = parse_args()

    # Set normalization parameters and dataset class based on the selected dataset
    if args.dataset == 'cifar100':
        norm_mean = [0.5071, 0.4867, 0.4408]
        norm_std = [0.2675, 0.2565, 0.2761]
        dataset_cls = torchvision.datasets.CIFAR100
    else:  # cifar10
        norm_mean = [0.4914, 0.4822, 0.4465]
        norm_std = [0.2023, 0.1994, 0.2010]
        dataset_cls = torchvision.datasets.CIFAR10

    # Load the corresponding pre-trained model based on the dataset and model arguments
    if args.dataset == 'cifar100':
        if args.model == 'resnet56':
            load_model = cifar100_resnet56(pretrained=True)
            load_model_name = 'resnet56_cifar100'
        elif args.model == 'vgg11':
            load_model = cifar100_vgg11_bn(pretrained=True)
            load_model_name = 'vgg11_cifar100'
        elif args.model == 'mobilenetv2':
            load_model = cifar100_mobilenetv2_x0_5(pretrained=True)
            load_model_name = 'mobilenetv2_cifar100'
        elif args.model == 'shufflenetv2':
            load_model = cifar100_shufflenetv2_x0_5(pretrained=True)
            load_model_name = 'shufflenetv2_cifar100'
        elif args.model == 'vit_b16':
            load_model = cifar100_vit_b16(pretrained=True)
            load_model_name = 'vit_b16_cifar100'
        else:
            raise ValueError("Unsupported model for CIFAR-100")
    elif args.dataset == 'cifar10':
        if args.model == 'resnet56':
            load_model = cifar10_resnet56(pretrained=True)
            load_model_name = 'resnet56_cifar10'
        elif args.model == 'vgg11':
            load_model = cifar10_vgg11_bn(pretrained=True)
            load_model_name = 'vgg11_cifar10'
        elif args.model == 'mobilenetv2':
            load_model = cifar10_mobilenetv2_x0_5(pretrained=True)
            load_model_name = 'mobilenetv2_cifar10'
        elif args.model == 'shufflenetv2':
            load_model = cifar10_shufflenetv2_x0_5(pretrained=True)
            load_model_name = 'shufflenetv2_cifar10'
        elif args.model == 'vit_b16':
            load_model = cifar10_vit_b16(pretrained=True)
            load_model_name = 'vit_b16_cifar10'
        else:
            raise ValueError("Unsupported model for CIFAR-10")
    else:
        raise ValueError("Unsupported dataset")

    # Print the model name and dataset information
    print(f"Loading model '{load_model_name}' for dataset '{args.dataset}' on device {device}...")

    # Create transformation and test dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    test_dataset = dataset_cls(root='img_classification/dataset', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    load_model = load_model.to(device)

    # Freeze all model parameters
    for param in load_model.parameters():
        param.requires_grad = False

    print("Model successfully loaded.")
    print("Evaluating original model:")
    original_accuracy = check_accuracy(load_model, test_loader, device)

    # Lists for storing standard deviations and accuracies for plotting
    std_list = []
    acc_list = []

    # Open a file to save the results with a header
    with open(args.output_file, 'w') as results_file:
        results_file.write("std\taccuracy\n")
        # Create a list of perturbation standard deviations, e.g., 0.0, 0.2, ..., (num_perturb_steps-1)*perturb_step
        pert_ratio_list = [round(args.perturb_step * i, 1) for i in range(args.num_perturb_steps)]
        for idx, pert_ratio in enumerate(pert_ratio_list):
            print(f"Perturbation ratio (std): {pert_ratio}")
            perturb_model = PerturbModel(model=load_model, pert_ratio=pert_ratio).to(device)
            print("Evaluating perturbed model:")
            pert_accuracy = check_accuracy(perturb_model, test_loader, device)
            relative_ratio = pert_accuracy / original_accuracy if original_accuracy > 0 else 0.0
            print(f"Relative Accuracy Ratio: {relative_ratio:.4f}\n")
            results_file.write(f"{pert_ratio}\t{pert_accuracy:.4f}\n")
            std_list.append(pert_ratio)
            acc_list.append(pert_accuracy)
    print(f"Results have been saved to '{args.output_file}'.")

    # Plot results if either plotting or saving of the plot is requested
    if args.plot or args.save_plot:
        plot_results(std_list, acc_list, do_plot=args.plot, do_save=args.save_plot)


if __name__ == '__main__':
    main()
