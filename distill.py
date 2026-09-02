import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_loaders
from train_valid import evaluate
from seed import set_seed
from model import StudentNet
from train_valid import evaluate, unpack_batch

SEED = 42

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_teacher = F.log_softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    # log_target=True: 兩邊都是 log-prob，數值較穩定
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean', log_target=True) * (T * T)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def train_kd_epoch(student, teacher, loader, optimizer, device, T, alpha):
    student.train()
    teacher.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        imgs, labels = unpack_batch(batch)
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(imgs)
        student_logits = student(imgs)
        loss = distillation_loss(student_logits, teacher_logits, labels, T, alpha)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (student_logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def fit_kd(teacher, tr_loader, va_loader, lr, wd, T, alpha, epochs, n_classes, n_channels, device, patience=5, verbose=False, seed=SEED):
    set_seed(seed)
    student = StudentNet(num_classes=n_classes, in_channels=n_channels).to(device)
    teacher = teacher.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state, no_improve = -1.0, None, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_kd_epoch(student, teacher, tr_loader, optimizer, device, T, alpha)
        val_loss, val_acc, val_auc, _, _ = evaluate(student, va_loader, n_classes=n_classes, device=device, criterion=criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_auc'].append(val_auc)

        if verbose:
            print(f"  epoch {epoch+1:>3}/{epochs} | KD_loss {tr_loss:.4f} train_acc {tr_acc:.4f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_state, no_improve = val_acc, copy.deepcopy(student.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (best val_acc={best_val_acc:.4f})")
                break

    student.load_state_dict(best_state)
    return student, history, best_val_acc
