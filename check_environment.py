"""Run before training; use --require-cuda in a Colab GPU session."""
import argparse
import platform
import importlib.metadata

import numpy as np
import torch
import torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    print('Python:', platform.python_version())
    for package in ('torch', 'torchvision', 'numpy', 'Pillow', 'matplotlib', 'scikit-learn', 'tqdm'):
        print(f'{package}: {importlib.metadata.version(package)}')
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select a GPU runtime before training.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    torch.manual_seed(42)
    np.random.seed(42)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(8, 10),
    ).to(device)
    output = model(torch.randn(2, 3, 128, 128, device=device))
    output.sum().backward()
    assert output.shape == (2, 10)
    print('PASS: imports, forward pass, and backward pass. This is not a trained model.')


if __name__ == '__main__':
    main()
