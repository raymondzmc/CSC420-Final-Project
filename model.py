import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from torchvision import models
import pdb
from PIL import Image

class ReOrganize(nn.Module):
    """
    Re-organize the conv output similar to approach employed by
    YOLOv2 (https://arxiv.org/pdf/1612.08242.pdf), used for
    concatenating the higher resolution features with low
    resolution features by stacking adjacent features into
    different channels instead of spatial locations.
    """
    def __init__(self, scale):
        super(ReOrganize, self).__init__()

        # The scale for which we are reducing the spatial size
        self.scale = scale

    def forward(self, x):
        c, h, w = x.size()[1], x.size()[2], x.size()[3]
        x = x.view(-1, c, h // self.scale, self.scale, w // self.scale, self.scale).transpose(3, 4).contiguous()
        x = x.view(-1, c, h // self.scale * w // self.scale, self.scale * self.scale).transpose(2, 3).contiguous()
        x = x.view(-1, c, self.scale * self.scale, h // self.scale, w // self.scale).transpose(1, 2).contiguous()
        return x.view(-1, self.scale * self.scale * c, h // self.scale, w // self.scale)


class FashionNet(nn.Module):
    """
    The FCN used for semantic segmentation on the Fashion Photographs dataset.
    Taking an (3 x 224 x 224) image as input, the network produce an (C x 224 x 224) 
    output tensor representing the segmentation mask for each of the C classes.
    """
    def __init__(self, n_classes=7):
        super(FashionNet, self).__init__()

        vgg = models.vgg11(pretrained=True)


        # Keep the layers of VGG-11 up to maxpool4
        self.features1 = nn.Sequential(*vgg.features[:11])
        self.features2 = nn.Sequential(*vgg.features[11:16])

        # Pointwise convolution to reduce the dimensions of the features 
        self.pw_conv = nn.Sequential(nn.Conv2d(256, 64, 1),
                                     nn.ReLU())

        # For reorganizing the feature map from the lower dimension
        self.reorg = ReOrganize(2)

        # Deconvolution layers for upsampling and producing the final segmentation output
        self.classifier = nn.Sequential(nn.ConvTranspose2d(768, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64, n_classes, kernel_size=1))


    def forward(self, x):
        # Output feature maps at different dimensions
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x1 = self.pw_conv(x1)
        x1 = self.reorg(x1)
        x = torch.cat((x1, x2), dim=1)

        out = self.classifier(x)
        return out



if __name__ == "__main__":
    vgg = models.vgg11(pretrained=True)
    net = FashionNet()
    image = Image.open('images/0001.jpg')
    test = torch.Tensor(1, 3, 600, 400)
    out = net(test)
    pdb.set_trace()
