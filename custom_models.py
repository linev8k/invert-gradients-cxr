import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LeNet(nn.Module):
    def __init__(self, channel=3, hidden=768, num_classes=10, loss='CE'):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        if loss == 'CE':
            self.fc = nn.Sequential(
                nn.Linear(hidden, num_classes)
                )
        if loss == 'BCE':
            self.fc = nn.Sequential(
                nn.Linear(hidden, num_classes),
                nn.Sigmoid()
                )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class DenseNet121(nn.Module):

    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, pre_trained=False):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained = pre_trained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class ResNet18(nn.Module):

    """Model modified.
    The architecture of our model is the same as standard ResNet18
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, pre_trained=False):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained = pre_trained)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet18(x)
        return x


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.05, 0.05)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.05, 0.05)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())
    # try:
    #     if hasattr(m, "weight"):
    #         nn.init.uniform_(m, a=-0.5, b=0.5)
    # except Exception:
    #     print('warning: failed in weights_init for %s.weight' % m._get_name())
    # try:
    #     if hasattr(m, "bias"):
    #         nn.init.uniform_(m, a=-0.5, b=0.5)
    # except Exception:
    #     print('warning: failed in weights_init for %s.bias' % m._get_name())
