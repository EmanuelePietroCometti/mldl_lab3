from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def get_tiny_imagenet_datasets(root="dataset/tiny-imagenet-200", batch_size=32):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=f"{root}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{root}/val", transform=transform)
    
    return train_dataset, val_dataset