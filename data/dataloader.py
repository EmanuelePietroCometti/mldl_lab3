from dataset import get_tiny_imagenet_datasets
import torch

def get_tiny_imagenet_loaders(root="dataset/tiny-imagenet-200", batch_size=32):
    train_dataset, val_dataset = get_tiny_imagenet_datasets(root, batch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader