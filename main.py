import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import medmnist

from sklearn.metrics import confusion_matrix, classification_report

from parse_config import load_config
from dataset import get_data, get_loaders, plot_data
from model import TeacherNet, StudentNet, count_params, get_model_size_mb
from train_valid import evaluate, fit, plot_history, measure_latency_ms_per_sample
from distill import fit_kd
from seed import set_seed


config = load_config()

""""""
SEED = config["seed"]["random_seed"]

set_seed(SEED)

""""""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"medmnist version: {medmnist.__version__}")

""""""
train_dataset, val_dataset, test_dataset, n_channels, n_classes, class_names = get_data(config["database"]["data_flag"])
train_loader, val_loader, test_loader = get_loaders(train_dataset, val_dataset, test_dataset)

plot_data(train_dataset, val_dataset, test_dataset, class_names, n_classes)

""""""
_teacher_probe = TeacherNet(num_classes=n_classes, in_channels=n_channels)
_student_probe = StudentNet(num_classes=n_classes, in_channels=n_channels)

teacher_params = count_params(_teacher_probe)
student_params = count_params(_student_probe)
ratio = teacher_params / student_params

print(f"Teacher 參數量: {teacher_params:,}")
print(f"Student 參數量: {student_params:,}")
print(f"Teacher / Student 比例: {ratio:.2f}x")

assert 5.0 <= ratio <= 10.0, f"參數量比例 {ratio:.2f}x 不在規定的 5-10x 範圍內"

# 檢查 forward pass 形狀正確
_x = torch.randn(2, n_channels, 28, 28)
assert _teacher_probe(_x).shape == (2, n_classes)
assert _student_probe(_x).shape == (2, n_classes)

del _teacher_probe, _student_probe, _x

""""""
teacher_search_space = [(lr, wd) for lr in config["train"]["lr_candidates"] for wd in config["train"]["wd_candidates"]]
teacher_search_results = []

print(type(config["train"]["lr_candidates"][0]))
print("=== Teacher 超參數搜尋（validation accuracy）===")
for lr, wd in teacher_search_space:
    _, _, best_val_acc = fit(
        model_fn=lambda: TeacherNet(num_classes=n_classes, in_channels=n_channels),
        tr_loader=train_loader, va_loader=val_loader, lr=lr, wd=wd, epochs=12,
        device=device, n_classes=n_classes, patience=4, verbose=False, seed=SEED
    )
    teacher_search_results.append({'lr': lr, 'wd': wd, 'val_acc': best_val_acc})
    print(f"lr={lr:<8} wd={wd:<8} -> best val_acc={best_val_acc:.4f}")

teacher_search_df = pd.DataFrame(teacher_search_results).sort_values('val_acc', ascending=False)
print(teacher_search_df)

best_teacher_cfg = teacher_search_df.iloc[0].to_dict()
print("\n選定 Teacher 超參數:", best_teacher_cfg)

print("=== 開始訓練 Teacher（early stopping 依 val accuracy）===")
teacher_model, teacher_history, teacher_val_acc = fit(
    model_fn=lambda: TeacherNet(num_classes=n_classes, in_channels=n_channels),
    tr_loader=train_loader, va_loader=val_loader,
    lr=best_teacher_cfg['lr'], wd=best_teacher_cfg['wd'],
    epochs=40, device=device, n_classes=n_classes, patience=8, verbose=True, seed=SEED
)
print(f"\nTeacher 最終 val_acc = {teacher_val_acc:.4f}")
plot_history(teacher_history, "Teacher")

""""""
student_search_space = [(lr, wd) for lr in config["train"]["lr_candidates"] for wd in config["train"]["wd_candidates"]]
student_search_results = []

print("=== Student Baseline 超參數搜尋（validation accuracy）===")
for lr, wd in student_search_space:
    _, _, best_val_acc = fit(
        model_fn=lambda: StudentNet(num_classes=n_classes, in_channels=n_channels),
        tr_loader=train_loader, va_loader=val_loader,
        lr=lr, wd=wd, epochs=12, device=device, n_classes=n_classes, patience=4, verbose=False, seed=SEED
    )
    student_search_results.append({'lr': lr, 'wd': wd, 'val_acc': best_val_acc})
    print(f"lr={lr:<8} wd={wd:<8} -> best val_acc={best_val_acc:.4f}")

student_search_df = pd.DataFrame(student_search_results).sort_values('val_acc', ascending=False)
print(student_search_df)

best_student_cfg = student_search_df.iloc[0].to_dict()
print("\n選定 Student Baseline 超參數:", best_student_cfg)

print("=== 開始訓練 Student Baseline（無蒸餾）===")
student_baseline_model, student_baseline_history, student_baseline_val_acc = fit(
    model_fn=lambda: StudentNet(num_classes=n_classes, in_channels=n_channels),
    tr_loader=train_loader, va_loader=val_loader,
    lr=best_student_cfg['lr'], wd=best_student_cfg['wd'],
    epochs=40, device=device, n_classes=n_classes, patience=8, verbose=True, seed=SEED
)
print(f"\nStudent Baseline 最終 val_acc = {student_baseline_val_acc:.4f}")
plot_history(student_baseline_history, "Student Baseline (no KD)")

""""""
kd_search_space = [(T, alpha) for T in config["train"]["T_candidates"] for alpha in config["train"]["alpha_candidates"]]
kd_search_results = []

print("=== KD 超參數搜尋 (T, alpha)（lr/wd 沿用 Student baseline 最佳設定）===")
for T, alpha in kd_search_space:
    _, _, best_val_acc = fit_kd(
        teacher=teacher_model, tr_loader=train_loader, va_loader=val_loader,
        lr=best_student_cfg['lr'], wd=best_student_cfg['wd'],
        T=T, alpha=alpha, epochs=12, n_classes=n_classes, n_channels=n_channels, device=device, patience=4, verbose=False, seed=SEED
    )
    kd_search_results.append({'T': T, 'alpha': alpha, 'val_acc': best_val_acc})
    print(f"T={T:<5} alpha={alpha:<5} -> best val_acc={best_val_acc:.4f}")

kd_search_df = pd.DataFrame(kd_search_results).sort_values('val_acc', ascending=False)
print(kd_search_df)

best_kd_cfg = kd_search_df.iloc[0].to_dict()
print("\n選定 KD 超參數:", best_kd_cfg)

print("=== 開始訓練 KD Student ===")
student_kd_model, student_kd_history, student_kd_val_acc = fit_kd(
    teacher=teacher_model, tr_loader=train_loader, va_loader=val_loader,
    lr=best_student_cfg['lr'], wd=best_student_cfg['wd'],
    T=best_kd_cfg['T'], alpha=best_kd_cfg['alpha'],
    epochs=40, n_classes=n_classes, n_channels=n_channels, device=device, patience=8, verbose=True, seed=SEED
)
print(f"\nKD Student 最終 val_acc = {student_kd_val_acc:.4f}")
plot_history(student_kd_history, "Student (KD)")

""""""
selection_summary = pd.DataFrame([
    {'Model': 'Teacher', 'Hyperparams': f"lr={best_teacher_cfg['lr']}, wd={best_teacher_cfg['wd']}",
     'Val Acc (model selection)': round(teacher_val_acc, 4)},
    {'Model': 'Student Baseline (no KD)', 'Hyperparams': f"lr={best_student_cfg['lr']}, wd={best_student_cfg['wd']}",
     'Val Acc (model selection)': round(student_baseline_val_acc, 4)},
    {'Model': 'Student (KD)', 'Hyperparams': f"lr={best_student_cfg['lr']}, wd={best_student_cfg['wd']}, "
                                              f"T={best_kd_cfg['T']}, alpha={best_kd_cfg['alpha']}",
     'Val Acc (model selection)': round(student_kd_val_acc, 4)},
])
print(selection_summary)

""""""
final_models = {
    'Teacher': (teacher_model, test_loader),
    'Student (Baseline, no KD)': (student_baseline_model, test_loader),
    'Student (KD)': (student_kd_model, test_loader),
}

model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)

print(f"Saving all trained models to '{model_save_dir}'...")

for name, (model, _) in final_models.items():
    # Replace special characters for filename safety
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_plus_').replace(',', '_').replace('/', '_').replace('-', '_').replace('.', '_')
    model_path = os.path.join(model_save_dir, f"{safe_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {name} to {model_path}")

print("All models saved successfully!")

""""""
criterion = nn.CrossEntropyLoss()

test_predictions = {}
rows = []
print("=== 最終 Test Set 評估 ===")
for name, (model, loader) in final_models.items():
    model = model.to(device)
    _, acc, auc, probs, labels = evaluate(model, loader, n_classes, device, criterion)
    n_params = count_params(model)
    size_mb = get_model_size_mb(model)
    latency_ms = measure_latency_ms_per_sample(model, loader, device)
    test_predictions[name] = (probs, labels)
    rows.append({
        'Model': name,
        'Params': n_params,
        'Params vs Student': f"{n_params / student_params:.2f}x",
        'Size (MB)': round(size_mb, 3),
        'Test Acc': round(acc, 4),
        'Test AUC (macro OvR)': round(auc, 4),
        'Latency (ms/sample)': round(latency_ms, 4),
    })
    print(f"{name:<48} Test Acc={acc:.4f}  Test AUC={auc:.4f}  Params={n_params:,}  "
          f"Size={size_mb:.3f}MB  Latency={latency_ms:.4f}ms/sample")

results_df = pd.DataFrame(rows)
print(results_df)

""""""
# 混淆矩陣
n_models = len(test_predictions)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = np.array(axes).reshape(-1)
for ax, (name, (probs, labels)) in zip(axes, test_predictions.items()):
    preds = probs.argmax(1)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(name, fontsize=20) # Increased title fontsize
    ax.set_xlabel('Predicted', fontsize=20); ax.set_ylabel('True', fontsize=20) # Added/Increased label fontsize
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(range(n_classes), fontsize=18) # Increased tick label fontsize
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(range(n_classes), fontsize=18) # Increased tick label fontsize
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=16,
                     color='white' if cm[i, j] > cm.max() / 2 else 'black') # Increased annotation fontsize
for ax in axes[n_models:]:
    ax.axis('off')
plt.tight_layout()
plt.show()

print("類別對照表:", {i: name for i, name in enumerate(class_names)})

""""""
# 逐類別表現比較
for name in ['Student (Baseline, no KD)', 'Student (KD)']:
    probs, labels = test_predictions[name]
    preds = probs.argmax(1)
    print(f"\n=== {name} — Classification Report (Test Set) ===")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

""""""
teacher_acc  = results_df.loc[results_df.Model == 'Teacher', 'Test Acc'].values[0]
baseline_acc = results_df.loc[results_df.Model == 'Student (Baseline, no KD)', 'Test Acc'].values[0]
kd_acc       = results_df.loc[results_df.Model == 'Student (KD)', 'Test Acc'].values[0]

teacher_lat  = results_df.loc[results_df.Model == 'Teacher', 'Latency (ms/sample)'].values[0]
kd_lat       = results_df.loc[results_df.Model == 'Student (KD)', 'Latency (ms/sample)'].values[0]

print(f"參數壓縮比 (Teacher / Student)      : {teacher_params / student_params:.2f}x")
print(f"模型大小壓縮比 (Teacher / Student)   : "
      f"{results_df.loc[results_df.Model=='Teacher','Size (MB)'].values[0] / results_df.loc[results_df.Model=='Student (KD)','Size (MB)'].values[0]:.2f}x")
print(f"推論加速比 (Teacher / KD Student)   : {teacher_lat / kd_lat:.2f}x")
print()
print(f"KD Student 相對 Teacher 的效能保留率 : {kd_acc / teacher_acc * 100:.2f}%")
print(f"KD 相對「無蒸餾 baseline」的準確率提升: {(kd_acc - baseline_acc) * 100:+.2f} 個百分點")
print()
