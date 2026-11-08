"""Self-contained inference file. Submit this file with its matching model.pth."""
import numpy as np
import torch
from torch import nn

CLASS_NAMES = (
    'shih_tzu', 'rhodesian_ridgeback', 'beagle', 'english_foxhound',
    'border_terrier', 'australian_terrier', 'golden_retriever',
    'old_english_sheepdog', 'samoyed', 'dingo',
)


class DogCNN(nn.Module):
    """Eight convolution layers in four stages; no pretrained weights or skips."""
    def __init__(self):
        super().__init__()
        # Layers execute from top to bottom. Each Conv2d's input channels must
        # match the previous layer's output channels, as must BatchNorm2d.
        self.network = nn.Sequential(
            # Stage 1: (3, 128, 128) -> (32, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Stage 2: (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Stage 3: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Stage 4: (128, 16, 16) -> (128, 8, 8)
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Classifier: 128 feature maps -> 256 features -> 10 class scores.
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            # Match in_features to the last convolution's output channels.
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.network(images)


def load_model(path='model.pth'):
    # Load on CPU first, so a GPU-trained checkpoint works on either platform.
    model = DogCNN()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device).eval()


def predict(model, images):
    """uint8 RGB (N,128,128,3) -> int64 class IDs (N,) in official order."""
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8:
        raise TypeError('images must be a uint8 NumPy array')
    if images.ndim != 4 or images.shape[1:] != (128, 128, 3):
        raise ValueError('Expected shape (N, 128, 128, 3)')
    if len(images) == 0:
        return np.empty(0, dtype=np.int64)
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(images), 64):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + 64]))
            # Match ToTensor + Normalize exactly: contiguous NCHW, CPU float32
            # arithmetic, then transfer to the model's device. A permuted NHWC
            # view otherwise selects a different GPU convolution memory layout.
            batch = batch.permute(0, 3, 1, 2).contiguous().float().div(255.0)
            batch.sub_(0.5).div_(0.5)
            batch = batch.to(device)
            predictions.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64, copy=False)
