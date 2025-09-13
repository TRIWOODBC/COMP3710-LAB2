import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ========== 1. 加载数据 ==========
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images.astype("float32")   # shape: (n_samples, h, w)
Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = len(target_names)
n_samples, h, w = X.shape

print("Total dataset size:")
print("n_samples:", n_samples, " image shape:", (h, w), " n_classes:", n_classes)
print("X_min:", X.min(), "X_max:", X.max())

# ========== 2. 划分数据集 ==========
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# CNN 需要 4D 输入 [N, C, H, W]
X_train = X_train[:, np.newaxis, :, :]
X_test  = X_test[:, np.newaxis, :, :]
print("X_train shape:", X_train.shape)

# 转成 Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

# ========== 3. 标准化 (零均值, 单位方差) ==========
mean = X_train_t.mean()
std = X_train_t.std().clamp_min(1e-6)   # 防除零
X_train_t = (X_train_t - mean) / std
X_test_t  = (X_test_t  - mean) / std

# ========== 4. 类别权重 (应对不平衡) ==========
unique, counts = np.unique(y_train, return_counts=True)
class_count = torch.tensor(counts, dtype=torch.float32)
weights = (class_count.sum() / class_count).to(torch.float32)
weights = weights / weights.sum() * len(unique)   # 归一化
criterion = nn.CrossEntropyLoss(weight=weights)

# ========== 5. DataLoader ==========
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

# ========== 6. 定义 CNN ==========
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (h // 2) * (w // 2), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========== 7. 训练配置 ==========
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print("Using device:", device)

model = CNN(num_classes=n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ========== 8. 训练循环 ==========
epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")

# ========== 9. 测试 ==========
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\n=== CNN Test Accuracy: {acc:.4f} ===")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names))
