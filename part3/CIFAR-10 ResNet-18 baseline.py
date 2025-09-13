import math, time, argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# --------------------- ResNet-18 (CIFAR版，自实现) ---------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # CIFAR-10: 输入32x32，首层做7x7/2不合适，换成3x3
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*BasicBlock.expansion, num_classes)
    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        return self.fc(x)

# --------------------------- 训练与评估 ---------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--num-workers', type=int, default=8)
    args = ap.parse_args()

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f"Device: {device}")

    # 提速设置
    torch.backends.cudnn.benchmark = True
    channels_last = (device.type == 'cuda')

    # 数据增强（快速&有效）
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),  # 很关键
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),(0.247, 0.243, 0.261))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_set,  batch_size=1024, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    model = ResNet18(num_classes=10).to(device)
    if channels_last: model = model.to(memory_format=torch.channels_last)

    # Label Smoothing 可以更稳
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp)

    # 线性 warmup 到 base lr（前5个epoch）
    def lr_warmup(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_warmup)

    best = 0.0
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if channels_last: imgs = imgs.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        # 调整学习率：先 warmup，再 cosine
        if epoch < 5: warmup.step()
        else: scheduler.step()

        train_loss = running / len(train_loader)
        acc = evaluate(model, test_loader, device)
        best = max(best, acc)
        print(f"Epoch {epoch+1:03d}/{args.epochs} | loss {train_loss:.4f} | test acc {acc*100:.2f}% | best {best*100:.2f}%")

    total_t = time.time() - start
    print(f"\nFinished. Best Test Acc = {best*100:.2f}% | Total Train Time = {total_t:.1f}s")

if __name__ == '__main__':
    main()
