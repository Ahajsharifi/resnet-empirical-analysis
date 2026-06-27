import torch
import torchvision
from torch.utils.data import random_split
from torchvision.transforms import v2


def get_dataloaders(batch_size=128,num_workers=2,val_split=0.2,seed=42):
    train_transform=v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(32, padding=4),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
    ])
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.4914,0.4822,0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )
    ])
    #Data Set
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform =train_transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform = test_transform)

    #Train / Val
    val_size = int(len(trainset) * val_split)
    train_size = len(trainset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset,val_dataset=random_split(trainset, [train_size,val_size], generator=generator)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #val dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,   shuffle=False, num_workers=num_workers)
    #test dataset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader,val_loader,test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    images, labels = next(iter(train_loader))
    print(f"Images shape : {images.shape}")
    print(f"Labels shape : {labels.shape}")
    print(f"dtype        : {images.dtype}")