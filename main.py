import torch
import torchvision
import argparse
import os
from models.resnets import *
from dataloader import load_datasets
from train_teacher import train_teacher
from evaluate import evaluate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher', default='resret18', type=str, help='The teacher model. default: ResNet 110')
    parser.add_argument('--student', default='resnet18', type=str, help='The student model. default: ResNet 8')
    parser.add_argument('--training_type', default='dih', type=str, help='The mode for training, could be either "ce" (regular cross-entropy), "kd" (canonical knowledge distillation), "fine_tune" (fine_tuning the intermediate heads), "fitnets", "dml" (deep mutual learning), or dih. default: dih')
    parser.add_argument('--dataset', default='CIFAR100', type=str, help='Input the name of dataset: default: CIFAR100')
    parser.add_argument('--lr', default=0.1, type=float, help='Input the learning rate: default: 0.1')
    parser.add_argument('--momentum', default=0.9, type=float, help='Input the momentum: default: 0.9')
    parser.add_argument('--wd', default=5e-4, type=float, help='Input the weight decay rate: default: 5e-4')
    parser.add_argument('--epochs', default=200, type=int, help='Input the number of epochs: default: 200')
    parser.add_argument('--batch_size', default=64, type=int, help='Input the batch size: default: 128')

    parser.add_argument('--schedule', type=list, nargs='+', default=[60, 120, 180], help='Decrease learning rate at these epochs.')
    parser.add_argument('--schedule_gamma', type=float, default=0.2, help='multiply the learning rate to this factor at pre-defined epochs in schedule. default : 0.2')
    parser.add_argument('--kd_alpha', default=0.1, type=float, help='alpha weigth in knowedge distiilation loss function')
    parser.add_argument('--kd_temperature', default=5, type=int, help='Temperature in knowedge distiilation loss function')

    parser.add_argument('--seed', default=3, type=int, help='seed value for reproducibility')

    parser.add_argument('--path_to_save_teacher', type=str, help='the path to save the model and/or headers after training')
    parser.add_argument('--saved_path_teacher', type=str, help='the path of the saved model')

    args = parser.parse_args()

    for (arg,value) in args._get_kwargs():
        print(f"{arg} : {value}\n{'*'*30}")

    return args

def main():
    args = parse_args()

    models_dict = {"res34": ResNet34,
                    "res18": ResNet18}
    terminal_layer = {"CIFAR10":10,
                        "CIFAR100":100}

    trainloader, testloader = load_datasets(args.dataset, args.batch_size)

    if args.saved_path_teacher:
        state_dict = torch.load(args.saved_path_teacher)
        teacher = models_dict[args.teacher](num_classes=terminal_layer[args.dataset])
        teacher.load_state_dict(state_dict)
    else:
        teacher = train_teacher(models_dict[args.teacher](num_classes=terminal_layer[args.dataset]), trainloader, args.lr, args.wd, args.epochs, args.momentum, args.schedule, args.schedule_gamma, args.seed)

        if args.path_to_save_teacher:
            torch.save(teacher.state_dict(), args.path_to_save_teacher)
            print(f"Success! Model saved to: {args.path_to_save_teacher}.pt")
        else:
            if not(os.path.isdir(f'./models/saved/{args.dataset}')):
                os.makedirs(f'./models/saved/{args.dataset}')
            torch.save(teacher.state_dict(), f'./models/saved/{args.dataset}/{args.teacher}.pt')
            print(f"Success! Model saved to: ./models/saved/{args.dataset}/{args.teacher}.pt")

    evaluate(teacher, testloader)

if __name__ == '__main__':
    main()