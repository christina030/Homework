import medmnist
from medmnist import INFO

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from seed import set_seed

SEED = 42

def get_data(data_flag):
    info = INFO[data_flag]
    n_channels = info['n_channels']       # 3 (RGB)
    n_classes = len(info['label'])        # 8
    DataClass = getattr(medmnist, info['python_class'])
    
    print("Dataset:", info['description'][:120], "...")
    print("n_channels:", n_channels, "| n_classes:", n_classes)
    print("Official n_samples:", info['n_samples'])
    print("Class labels:", info['label'])

    # 影像正規化到 [-1, 1]（3 channel RGB）
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
    ])
    
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset   = DataClass(split='val',   transform=data_transform, download=True)
    test_dataset  = DataClass(split='test',  transform=data_transform, download=True)
    
    print("Train / Val / Test sizes:", len(train_dataset), len(val_dataset), len(test_dataset))
    assert len(train_dataset) == info['n_samples']['train']
    assert len(val_dataset)   == info['n_samples']['val']
    assert len(test_dataset)  == info['n_samples']['test']
    print("確認採用官方切分")
  
    class_names = [info['label'][str(i)].split(' ')[0] for i in range(n_classes)]

    return train_dataset, val_dataset, test_dataset, n_channels, n_classes, class_names


def get_loaders(train_dataset, val_dataset, test_dataset, batch_size_train=128, batch_size_eval=256, seed=SEED):
    """每次呼叫都會重建 DataLoader 並重設隨機種子，確保每個超參數組合的訓練過程可重現、公平比較。"""
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                               num_workers=2, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader


def plot_data(train_dataset, val_dataset, test_dataset, class_names, n_classes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # 類別分布（train / val / test）
    for ax_idx, (name, ds) in enumerate([('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]):
        counts = np.bincount(ds.labels.squeeze(), minlength=n_classes)
        axes[0].plot(range(n_classes), counts, marker='o', label=name)
    axes[0].set_xticks(range(n_classes))
    axes[0].set_xticklabels(class_names, rotation=60, ha='right', fontsize=16) # Adjusted fontsize
    axes[0].set_ylabel('Count', fontsize=16) # Adjusted fontsize
    axes[0].set_title('Class distribution', fontsize=18) # Adjusted fontsize
    axes[0].legend(fontsize=10) # Adjusted fontsize
    
    # 範例影像
    sample_idx = np.random.choice(len(train_dataset), 1, replace=False)
    axes[1].axis('off')
    fig2, axes2 = plt.subplots(1, 8, figsize=(16, 2.2))
    ploted = []
    i = 0
    while i < 8:
        img, label = train_dataset[sample_idx[0]]
        class_name = class_names[int(np.asarray(label).squeeze())]
        if class_name in ploted:
            sample_idx = np.random.choice(len(train_dataset), 1, replace=False)
            continue
        img_np = (img.numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
        axes2[i].imshow(img_np)
        axes2[i].set_title(class_name, fontsize=20) # Adjusted fontsize
        axes2[i].axis('off')
        ploted.append(class_name)
        i += 1
    
    plt.tight_layout()
    plt.show()
