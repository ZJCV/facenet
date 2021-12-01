import torch.nn as nn
import torchvision.models as models

from zcls.config.key_word import KEY_OUTPUT


class EmbeddingNet(nn.Module):
    def __init__(self, num_classes=1000, arch='resnet50'):
        super().__init__()

        self.arch = arch
        assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        self.model = eval(f'models.{arch}')(pretrained=True)

        self.init_weight(num_classes)

    def init_weight(self, num_classes):
        if num_classes != 1000:
            old_fc = self.model.fc
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = nn.Linear(in_features, num_classes, bias=True)
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.zeros_(new_fc.bias)

            self.model.fc = new_fc

    def forward(self, x):
        x = self.model(x)
        x = nn.functional.normalize(x, dim=1)

        return {KEY_OUTPUT: x}


def get_resnet(num_classes=1000, arch='resnet50'):
    return EmbeddingNet(num_classes=num_classes, arch=arch)
