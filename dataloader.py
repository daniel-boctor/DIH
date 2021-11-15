import torch
import torchvision
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_datasets(dataset, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
    
    if dataset == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)#shuffle=True
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader