import copy
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from dataset import get_loaders
from train_valid import evaluate
from seed import set_seed

def unpack_batch(batch):
    imgs, labels = batch
    labels = labels.squeeze().long()
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    return imgs, labels


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        imgs, labels = unpack_batch(batch)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, n_classes, device, criterion=None):
    """回傳 (avg_loss, accuracy, macro-AUC(OvR), probs, labels)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for batch in loader:
        imgs, labels = unpack_batch(batch)
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
        probs = F.softmax(logits, dim=1)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    acc = correct / total
    avg_loss = total_loss / total if criterion is not None else None
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro',
                             labels=list(range(n_classes)))
    except ValueError:
        auc = float('nan')
    return avg_loss, acc, auc, all_probs, all_labels


def fit(model_fn, lr, wd, epochs, device, patience=5, verbose=False, seed=SEED):
    """標準（無蒸餾）訓練迴圈，內含 early stopping，模型選擇依據 validation accuracy。
    model_fn: 一個回傳全新模型實例的函式（確保每次呼叫從相同初始化開始）。
    """
    set_seed(seed)
    tr_loader, va_loader, _ = get_loaders(seed=seed)
    model = model_fn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state, no_improve = -1.0, None, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, va_loader, device, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_auc'].append(val_auc)

        if verbose:
            print(f"  epoch {epoch+1:>3}/{epochs} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_state, no_improve = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_acc


def plot_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_title(f'{title} - Loss'); axes[0].set_xlabel('epoch'); axes[0].legend()
    axes[1].plot(history['train_acc'], label='train')
    axes[1].plot(history['val_acc'], label='val')
    axes[1].set_title(f'{title} - Accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def measure_latency_ms_per_sample(model, loader, device, n_warmup_batches=3):
    model.eval()
    it = iter(loader)
    for _ in range(min(n_warmup_batches, len(loader))):
        imgs, _ = unpack_batch(next(it))
        imgs = imgs.to(device)
        _ = model(imgs)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_time, total_samples = 0.0, 0
    for batch in loader:
        imgs, _ = unpack_batch(batch)
        imgs = imgs.to(device)
        start = time.perf_counter()
        _ = model(imgs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start
        total_samples += imgs.size(0)
    return (total_time / total_samples) * 1000.0
