import torch
import wandb
import os 

def validate(model, val_loader, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    print(f"Validation: Loss {val_loss:.4f}, Acc {val_acc:.2f}%")
    return val_acc

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss, correct, total = 0, 0, 0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    wandb.log({'Train Loss': train_loss, 'Train Accuracy': train_acc, 'Epoch': epoch})
    wandb.save(os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth"))
    print(f"Epoch {epoch}: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")