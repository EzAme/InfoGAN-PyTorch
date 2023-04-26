import torch
import torchvision.transforms as T
import torchvision.datasets as dsets
from HandGestureDataset import HandGestureDataSet as HGD
from pathlib import Path
# Directory containing the data.
path = str(Path.home())+'/Documents/data/leapGestRecog/'

def get_data(dataset, batch_size):



    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = T.Compose([
                T.ToPILImage(),
                T.CenterCrop(240),
                T.Resize((28,28)),
                T.ToTensor()])

        dataset = HGD(root = path,
                    train= True, 
                    transform=transform)
        
    # Get SVHN dataset.
    if dataset == 'SVHN':
        transform = T.Compose([
                T.ToPILImage(),
                T.CenterCrop(240),
                T.Resize((32,32)),
                T.ToTensor()])

        dataset = HGD(root = path,
                    train= True, 
                    transform=transform)
    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = T.Compose([
            T.Resize(28),
            T.CenterCrop(28),
            T.ToTensor()])

        dataset = dsets.FashionMNIST(root+'fashionmnist/', train='train', 
                                download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = T.Compose([
            T.Resize(32),
            T.CenterCrop(32),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root+'celeba/', transform=transform)

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers = 8)

    return dataloader