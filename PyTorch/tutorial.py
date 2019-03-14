import torch
import torchvision
import torchvision.transform as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

tainset = torchvision.dataset.CIFAR10(root='./data', train=True, download=True, transform=transform)
