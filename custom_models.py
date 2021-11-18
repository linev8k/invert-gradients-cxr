import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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
    def __init__(self, out_size, colour_input='RGB', pre_trained=False):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = pre_trained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if colour_input == 'L':
            self.rgb_to_grey_input()

    def forward(self, x):
        x = self.densenet121(x)
        return x

    def rgb_to_grey_input(self):

        """Replace the first convolutional layer that takes a 3-dimensional (RGB) input
        with a 1-dimensional layer, adding the weights of each existing dimension
        in order to retain pretrained parameters"""

        conv0_weight = self.densenet121.features.conv0.weight.clone()
        self.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.densenet121.features.conv0.weight = nn.Parameter(conv0_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights

class ResNet50(nn.Module):

    """Model modified.
    The architecture of our model is the same as standard ResNet18
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, colour_input = 'RGB', pre_trained=False):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained = pre_trained)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if colour_input == 'L':
            self.rgb_to_grey_input()

    def forward(self, x):
        x = self.resnet50(x)
        return x

    def rgb_to_grey_input(self):

        """Replace the first convolutional layer that takes a 3-dimensional (RGB) input
        with a 1-dimensional layer, adding the weights of each existing dimension
        in order to retain pretrained parameters"""

        conv1_weight = self.resnet50.conv1.weight.clone()
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.resnet50.conv1.weight = nn.Parameter(conv1_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights

class ResNet18(nn.Module):

    """Model modified.
    The architecture of our model is the same as standard ResNet18
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, colour_input = 'RGB', pre_trained=False):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained = pre_trained)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

        if colour_input == 'L':
            self.rgb_to_grey_input()

    def forward(self, x):
        x = self.resnet18(x)
        return x

    def rgb_to_grey_input(self):

        """Replace the first convolutional layer that takes a 3-dimensional (RGB) input
        with a 1-dimensional layer, adding the weights of each existing dimension
        in order to retain pretrained parameters"""

        conv1_weight = self.resnet18.conv1.weight.clone()
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            self.resnet18.conv1.weight = nn.Parameter(conv1_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights

def freeze_batchnorm(model):

    """Modify model to not track gradients of batch norm layers
    and set them to eval() mode (no running stats updated)"""

    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

def freeze_all_but_last(model):

    """Modify model to not track gradients of all but the last classification layer.
    Note: This is customized to the module naming of ResNet and DenseNet architectures."""

    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)


# class DenseNet121(nn.Module):
#
#     """Model modified.
#     The architecture of our model is the same as standard DenseNet121
#     except the classifier layer which has an additional sigmoid function.
#     """
#
#     def __init__(self, out_size, pre_trained=False):
#         super(DenseNet121, self).__init__()
#         self.densenet121 = models.densenet121(pretrained = pre_trained)
#         num_ftrs = self.densenet121.classifier.in_features
#         self.densenet121.classifier = nn.Sequential(
#             nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.densenet121(x)
#         return x
#
# class ResNet18(nn.Module):
#
#     """Model modified.
#     The architecture of our model is the same as standard ResNet18
#     except the classifier layer which has an additional sigmoid function.
#     """
#
#     def __init__(self, out_size, pre_trained=False):
#         super(ResNet18, self).__init__()
#         self.resnet18 = models.resnet18(pretrained = pre_trained)
#         num_ftrs = self.resnet18.fc.in_features
#         self.resnet18.fc = nn.Sequential(
#             nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.resnet18(x)
#         return x
#
# class ResNet50(nn.Module):
#
#     """Model modified.
#     The architecture of our model is the same as standard ResNet18
#     except the classifier layer which has an additional sigmoid function.
#     """
#
#     def __init__(self, out_size, pre_trained=False):
#         super(ResNet50, self).__init__()
#         self.resnet50 = models.resnet50(pretrained = pre_trained)
#         num_ftrs = self.resnet50.fc.in_features
#         self.resnet50.fc = nn.Sequential(
#             nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.resnet50(x)
#         return x


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
