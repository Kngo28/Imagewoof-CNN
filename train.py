"""Train a baseline and verify the exact submission interface on validation data."""
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision

from submission import CLASS_NAMES, DogCNN, load_model, predict


class OfficialImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        found = {p.name for p in Path(directory).iterdir() if p.is_dir()}
        if found != set(CLASS_NAMES):
            raise ValueError(f'Unexpected class folders: {found}')
        return list(CLASS_NAMES), {name: i for i, name in enumerate(CLASS_NAMES)}


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


def verify_submission(model, dataset, expected_accuracy):
    correct = 0
    for start in range(0, len(dataset.samples), 64):
        samples = dataset.samples[start:start + 64]
        raw = []
        for path, _ in samples:
            with Image.open(path) as image:
                raw.append(np.array(image.convert('RGB')))
        predictions = predict(model, np.stack(raw))
        labels = np.array([label for _, label in samples])
        assert predictions.shape == labels.shape
        assert predictions.dtype == np.int64
        assert ((predictions >= 0) & (predictions < 10)).all()
        correct += int((predictions == labels).sum())
    accuracy = correct / len(dataset)
    if abs(accuracy - expected_accuracy) > 1e-8:
        raise RuntimeError(f'predict accuracy {accuracy} != validation {expected_accuracy}')
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='AdamW weight decay (default: 0.0001)')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0:
        parser.error('epochs, patience, batch size, and learning rate must be positive')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('weight decay must be finite and nonnegative')
    # Refuse accidental overwrites of a previous experiment.
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    mapping = {}
    for line in (args.data / 'classes.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            index, name = line.split(maxsplit=1)
            mapping[name.strip()] = int(index)
    if mapping != {name: i for i, name in enumerate(CLASS_NAMES)}:
        raise ValueError('classes.txt does not match the submission class order')
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)

    # Convert image pixels into a PyTorch tensor, changing their layout from height–width–channels to channels–height–width.
    train_transform = transforms.Compose([
        # Keep 80-100% of the image area, then resize back to 128x128.
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_data = OfficialImageFolder(args.data / 'train', transform=train_transform)
    val_data = OfficialImageFolder(args.data / 'val', transform=val_transform)
    if (len(train_data), len(val_data)) != (9065, 1295):
        raise ValueError('Dataset counts do not match the instructor splits')
    for dataset in (train_data, val_data):
        for path, _ in dataset.samples:
            with Image.open(path) as image:
                if image.size != (128, 128) or image.mode != 'RGB':
                    raise ValueError(f'Expected RGB 128x128: {path}')
    
    # Data loaders organize images into batches            
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)

    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(python=sys.version, torch=torch.__version__,
                  torchvision=torchvision.__version__, numpy=np.__version__,
                  device=str(device), classes=list(CLASS_NAMES),
                  gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    
    # Preserve the exact inference architecture alongside its weights.
    for name in ('submission.py', 'train.py'):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)

    model = DogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy, stale_epochs = -1.0, 0
    weights_path = args.output / 'model.pth'
    with (args.output / 'history.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'lr'])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr']])
            handle.flush()
            print(f'Epoch {epoch:02d}: train loss={train_loss:.4f}, acc={train_acc:.2%}; '
                  f'val loss={val_loss:.4f}, acc={val_acc:.2%}', flush=True)
            if val_acc > best_accuracy:
                best_accuracy, stale_epochs = val_acc, 0
                temporary = args.output / 'model.tmp.pth'
                torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
                temporary.replace(weights_path)
                print('Saved new best model', flush=True)
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= args.patience:
                print('Early stopping', flush=True)
                break

    best_model = load_model(weights_path)
    # Use the same batch size as predict for the preprocessing parity check.
    reference_loader = DataLoader(val_data, batch_size=64)
    _, reference_acc = run_epoch(best_model, reference_loader, criterion, device)
    verified_acc = verify_submission(best_model, val_data, reference_acc)
    # Confirm the same saved weights can execute on CPU too.
    best_model.cpu()
    cpu_result = predict(best_model, np.zeros((1, 128, 128, 3), dtype=np.uint8))
    assert cpu_result.shape == (1,)
    (args.output / 'verification.json').write_text(json.dumps({
        'best_training_validation_accuracy': best_accuracy,
        'predict_validation_accuracy': verified_acc,
        'cpu_inference_passed': True,
    }, indent=2))
    print(f'PASS: predict validation accuracy={verified_acc:.2%}; CPU inference passed.', flush=True)
    print(f'Submit {args.output / "submission.py"} and {weights_path}', flush=True)


if __name__ == '__main__':
    main()
