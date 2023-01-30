import torch
import torch.nn as nn
from math import sqrt


class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))
    

class VDSR(nn.Module):
    def __init__(self) -> None:
        super(VDSR, self).__init__()
        # Input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction blocks
        feature_extractor = []
        for _ in range(18):
            feature_extractor.append(ConvReLU(64))
        self.feature_extractor = nn.Sequential(*feature_extractor)
        
        # Output layer
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.feature_extractor(out)
        out = self.conv2(out)
        
        out = torch.add(out, identity)
        
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))
        
