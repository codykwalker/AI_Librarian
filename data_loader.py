import os
import torchvision
from torchvision import transforms
import torch.utils.data as data

path = './data/'
transform_imgnet = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_normal = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


def get_data_loader(batch_size, transforms=transform_normal, shuffle=False):
    train_data = torchvision.datasets.ImageFolder(root=path, transform=transforms)
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return train_data_loader
