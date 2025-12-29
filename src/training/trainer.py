import torch
import numpy as np
import time

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs, 1)

        # Backward + Optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    val_loss = running_loss / total_samples
    val_acc = running_corrects.double() / total_samples

    return val_loss, val_acc.item()

def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, early_stopping=None):
    """
    Main training loop.
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler Step
        if scheduler:
            # Check if scheduler is ReduceLROnPlateau (needs metric) or StepLR (no metric)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        current_lr = optimizer.param_groups[0]['lr']
        
        duration = time.time() - start
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {duration:.0f}s")
        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} LR: {current_lr:.6f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Early Stopping
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
    # Load best model weights if early stopping was used (it saves the best one)
    if early_stopping:
        model.load_state_dict(torch.load(early_stopping.path))
        
    return model, history
