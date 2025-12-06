import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def _train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _eval_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _make_train_loaders(in_domain_path, out_domain_path, batch_size=32):
    in_ds = datasets.ImageFolder(in_domain_path, transform=_train_transforms())
    out_ds = datasets.ImageFolder(out_domain_path, transform=_train_transforms())

    in_loader = DataLoader(
        in_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    out_loader = DataLoader(
        out_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    num_classes = len(in_ds.classes)
    return in_loader, out_loader, num_classes


def _make_eval_loader(eval_path, batch_size=64):
    eval_ds = datasets.ImageFolder(eval_path, transform=_eval_transforms())
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return loader


def learn(in_domain_path, out_domain_path,
          num_epochs=10, batch_size=32, lr=1e-4, alpha=0.2):
    in_loader, out_loader, num_classes = _make_train_loaders(
        in_domain_path, out_domain_path, batch_size=batch_size
    )

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    out_iter = iter(out_loader)
    warmup_epochs = int(num_epochs * 0.7)

    model.train()
    for epoch in range(num_epochs):
        for x_in, y_in in in_loader:
            x_in = x_in.to(DEVICE)
            y_in = y_in.to(DEVICE)

            optimizer.zero_grad()

            logits_in = model(x_in)
            loss_in = ce_loss(logits_in, y_in)

            if epoch < warmup_epochs or alpha <= 0.0:
                loss = loss_in
            else:
                try:
                    x_out, _ = next(out_iter)
                except StopIteration:
                    out_iter = iter(out_loader)
                    x_out, _ = next(out_iter)

                x_out = x_out.to(DEVICE)

                logits_out = model(x_out)
                p_out = torch.softmax(logits_out, dim=1)
                entropy_reg = (p_out * torch.log(p_out + 1e-8)).sum(dim=1).mean()

                loss = loss_in + alpha * entropy_reg

            loss.backward()
            optimizer.step()

    return model


def compute_accuracy(eval_path, model, batch_size=64):
    loader = _make_eval_loader(eval_path, batch_size=batch_size)
    model = model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    if total == 0:
        return 0.0

    return correct / total
