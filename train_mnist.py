"""MNIST training script for Spiking Neural Networks.

This module provides a training pipeline for Spiking Neural Networks (SNNs) on the MNIST dataset.
It includes:
- Spike encoding functions for converting images to spike trains
- Training loop with progress tracking
- Model saving and loading functionality
- Support for both MNIST and Fashion-MNIST datasets

The script uses the SNNCNN model defined in the snn_cnn module and implements a surrogate gradient
approach for training spiking neural networks.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    ReduceLROnPlateau
)

import torchvision

import torchvision.transforms as transforms
from tqdm import tqdm  # Import tqdm library for progress bars

from snn_cnn import SNNCNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def encoder(images: torch.Tensor, window_t: int) -> torch.Tensor:
#     images = images.unsqueeze(0)
#     steps = torch.arange(window_t + 1, device=images.device)[:,None,None]
#     steps = (steps * images + 0.5).floor()
#     steps1, steps2 = steps[:-1], steps[1:]
#     spikes = steps2 - steps1
#     spikes = spikes.view(window_t,-1,1, 28,28)
#     return spikes

def encoder(images: torch.Tensor, window_t) -> torch.Tensor:
    """Convert images to spike trains using Bernoulli sampling.
    
    Args:
        images: Input images tensor of shape (batch_size, 784)
        window_t: Number of time steps for the spike train
        
    Returns:
        Spike trains tensor of shape (window_t, batch_size, 1, 28, 28)
    """
    images = images.unsqueeze(0)
    expanded = images.expand(window_t, -1, 784)
    spikes = torch.bernoulli(expanded)
    spikes = spikes.view(window_t, -1, 1, 28, 28)
    return spikes


# def encoder(images: torch.Tensor, window_t) -> torch.Tensor:
#     images = images.unsqueeze(0)
#     expanded = images.expand(window_t, -1, 784)
#     spikes = torch.poisson(expanded)
#     spikes = spikes.view(window_t,-1,1, 28,28)
#     return spikes


transform = transforms.ToTensor()


def optimizer_make(model: nn.Module):
    """Create and configure an optimizer for the model.
    
    Args:
        model: The neural network model to optimize
        
    Returns:
        Configured Adam optimizer
    """
    lr = 3.3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


criterion = lambda x, y: f.mse_loss(x, y, reduction='mean')


def train(model: SNNCNN, name, epochs=25, batch_size=50, device=DEVICE,
          encoder=encoder, transform=transform, criterion=criterion,
          optimizer_make=optimizer_make, model_dir='model', mnist_dir='data/',
          resume=False):  # New: resume=True means continue training
    if isinstance(model, SNNCNN):
        model: SNNCNN = torch.compile(model.to(device))

    window_t = model.window_t
    average_tau = model.average_tau

    # Create model save directory
    model_path = os.path.join(model_dir, f'mnist_{name}.pth')
    history_path = os.path.join(model_dir, f'mnist_{name}_history.pth')

    # Check for existing model
    if os.path.exists(model_path) and os.path.exists(history_path):
        print(f"Loading pre-trained {name} model and history...")

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            print(Exception)

        history = torch.load(history_path)
        epochs = len(history["Test Loss"])
        # Display history loading progress with tqdm
        for epoch in tqdm(range(epochs), desc="Loading history"):
            print(f'Epoch {epoch + 1}/{epochs} - '
                  f'Train Loss: {history["Train Loss"][epoch]:.6f}, '
                  f'Train Acc: {history["Train Accuracy"][epoch]:.4f}; '
                  f'Test Loss: {history["Test Loss"][epoch]:.6f}, '
                  f'Test Acc: {history["Test Accuracy"][epoch]:.4f}')
        print(f'Minimum Train Loss: {min(history["Train Loss"]):.6f}')
        print(f'Minimum Test Loss: {min(history["Test Loss"]):.6f}')
        print(f'Maximum Test Acc: {max(history["Test Accuracy"]):.4f}')
        if resume:
            print("Resuming training...")
        else:
            return model, history

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Data loading
    train_data = torchvision.datasets.MNIST(
        mnist_dir, train=True, transform=transform, download=False
    )

    test_data = torchvision.datasets.MNIST(
        mnist_dir, train=False, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size
    )

    optimizer = optimizer_make(model)
    scheduler = StepLR(optimizer, step_size=1, gamma=1)
    history = {
        'Test Loss': [],
        'Test Accuracy': [],
        'Train Loss': [],
        'Train Accuracy': []
    }
    best_loss = float('inf')
    best_model = None

    print("Training from scratch...")

    # Wrap epochs loop with tqdm to display overall training progress
    epoch_iter = tqdm(range(epochs), desc="Epochs", position=0)
    for epoch in epoch_iter:
        # Training phase
        model.train()
        train_loss, correct, total_loss, total = 0, 0, 0, 0

        # Wrap training data loader with tqdm to display batch progress
        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            position=1, leave=False
        )

        for images, labels in train_iter:
            model.reset()
            images = images.to(device).view(-1, 784)
            spikes = encoder(images, window_t)
            labels = labels.to(device)
            targets = f.one_hot(labels, num_classes=10).float()

            for t in range(0, window_t, average_tau):
                optimizer.zero_grad()
                spikes_input = spikes[t:t + average_tau]
                _, hat_spk = model.step(spikes_input, images.view(-1, 1, 28, 28))
                loss = criterion(hat_spk, targets)
                loss.backward()
                optimizer.step()
                total_loss += 1
                train_loss += loss.item()
                _, predicted = torch.max(hat_spk, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Update batch progress bar information
            train_iter.set_postfix({
                'Train Loss': train_loss / max(total_loss, 1),
                'Train Acc': correct / max(total, 1)
            })
        scheduler.step()
        avg_train_loss = train_loss / total_loss
        train_acc = correct / total

        # Testing phase
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0

        # Wrap test data loader with tqdm
        test_iter = tqdm(
            test_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Test]",
            position=1, leave=True
        )

        with torch.no_grad():
            for images, labels in test_iter:
                model.reset()
                images = images.to(device).view(-1, 784)
                labels = labels.to(device)
                targets = f.one_hot(labels, num_classes=10)
                spikes = encoder(images, window_t*2)

                with torch.no_grad():
                    spikes, _ = model.step(spikes)

                hat_spk = spikes.mean(dim=0)

                loss = criterion(hat_spk, targets)
                test_loss += loss.item()
                _, predicted = torch.max(hat_spk, 1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

                # Update test progress bar information
                test_iter.set_postfix({
                    'Test Loss': test_loss / len(test_iter),
                    'Test Acc': test_correct / test_total
                })

        avg_test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total

        # Update history record
        history['Train Loss'].append(avg_train_loss)
        history['Train Accuracy'].append(train_acc)
        history['Test Loss'].append(avg_test_loss)
        history['Test Accuracy'].append(test_acc)

        # Save best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model = model.state_dict()

        torch.save(best_model, model_path)
        torch.save(history, history_path)

        # Update main progress bar information
        epoch_iter.set_postfix({
            'Train Loss': avg_train_loss,
            'Train Acc': f'{train_acc:.5f}',
            'Test Loss': avg_test_loss,
            'Test Acc': f'{test_acc:.5f}',
        })

    # Save final results
    model.load_state_dict(best_model)
    print(f'Minimum Test Loss :{best_loss:.4f}')
    torch.save(history, history_path)

    return model, history


if __name__ == '__main__':
    import snn_cnn
    model = snn_cnn.SNNCNN()
    # Train on regular MNIST dataset
    train(model, 'test', mnist_dir='data/')
    # Train on Fashion-MNIST dataset
    train(model, 'test', mnist_dir='data/Fashion-Mnist/')
