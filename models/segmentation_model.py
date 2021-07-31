#from __future__ import division

import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
#from torch.nn.functional import upsample
from torch.nn.functional import interpolate
from typing import Type, Any, Callable, Union, List, Optional

def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is the product of two gaussian distributions
    # for two different variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3:
        padding = 1
    elif kernel_size == 5:
        padding = 2
    else:
        padding = 0

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def get_laplaceOfGaussian_filter(kernel_size=3, sigma=2, channels=3):
    '''
    laplacian of Gaussian 2D filter
    '''
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.

    used_sigma = sigma
    # Calculate the 2-dimensional gaussian kernel which is
    log_kernel = (-1./(math.pi*(used_sigma**4))) \
        * (1-(torch.sum((xy_grid - mean)**2., dim=-1) / (2*(used_sigma**2)))) \
        * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*(used_sigma**2)))

    # Make sure sum of values in gaussian kernel equals 1.
    log_kernel = log_kernel / torch.sum(log_kernel)

    # Reshape to 2d depthwise convolutional weight
    log_kernel = log_kernel.view(1, 1, kernel_size, kernel_size)
    log_kernel = log_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3: padding = 1
    elif kernel_size == 5: padding = 2
    else: padding = 0

    log_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                           groups=channels, bias=False, padding=padding)

    log_filter.weight.data = log_kernel
    log_filter.weight.requires_grad = False

    return log_filter


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.enable_cbs = False
        self.planes = planes

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def get_new_kernels(self, fil2, fil3, kernel_size, std):
        self.enable_cbs = True
        if (fil2 == 'gau'): self.kernel1 = get_gaussian_filter( kernel_size=kernel_size, sigma=std, channels=self.planes)
        elif (fil2 == 'LOG'): self.kernel1 = get_laplaceOfGaussian_filter( kernel_size=kernel_size, sigma=std, channels=self.planes)

        if (fil3 == 'gau'): self.kernel2 = get_gaussian_filter( kernel_size=kernel_size, sigma=std, channels=self.planes)
        elif (fil3 == 'LOG'): self.kernel2 = get_laplaceOfGaussian_filter( kernel_size=kernel_size, sigma=std, channels=self.planes)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # f indicates feature extraction for VSGAT, if False, it is feature extraction for the Segmentation Pipeline -------------------------------------------------------------------------

    def _forward_impl(self, x, f) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x) if f else x
        return x
 

    def forward(self, x: Tensor, f) -> Tensor:
        return self._forward_impl(x, f)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
    ) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print("Loading pre-trained ImageNet weights")
        state_dict = torch.load('models/r18/resnet18-f37072fd.pth')
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


class Resnet18_8s_CBS(nn.Module):
    def __init__(self, args, num_classes=1000):

        super(Resnet18_8s_CBS, self).__init__()
        resnet18_8s = resnet18(
            pretrained=True)

        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)

        self.resnet18_8s = resnet18_8s
        self._normal_initialization(self.resnet18_8s.fc)

        self.in_planes = 64
        self.std = args.std
        self.enable_cbs = args.use_cbs
        self.factor = args.std_factor
        self.epoch = args.cbs_epoch
        self.kernel_size = args.kernel_size

        self.fil1 = args.fil1
        self.fil2 = args.fil2
        self.fil3 = args.fil3

    def _normal_initialization(self, layer):

        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False, f=False):
        #input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_8s(x, f)
        #x = x if f else nn.functional.upsample(x, size=input_spatial_dim, mode='bilinear', align_corners=True)
        return x


    def get_new_kernels(self, epoch_count):
        if epoch_count % self.epoch == 0 and epoch_count is not 0:
            self.std *= self.factor
        if (self.fil1 == 'gau'):
            self.kernel1 = get_gaussian_filter( kernel_size=self.kernel_size, sigma=self.std, channels=64)
        elif (self.fil1 == 'LOG'):
            self.kernel1 = get_laplaceOfGaussian_filter( kernel_size=self.kernel_size, sigma=self.std, channels=64)

        for child in self.resnet18_8s.layer1.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.resnet18_8s.layer2.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.resnet18_8s.layer3.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.resnet18_8s.layer4.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def resnet18_8s_model_cbs(pretrained=True, root='~/.encoding/models', **kwargs):
    args = Namespace(alg='res', batch_size=1, cbs_epoch=20, checkpointfile='checkpoint/incremental/testing', cuda=True,
                     decay=0.0001, dist_loss='ce', dist_loss_act='softmax', dist_ratio=0.5, epoch_base=30, epoch_finetune=15,
                     fil1='LOG', fil2='gau', fil3='gau', ft_lr_factor=0.1, gamma=0.8, kernel_size=3, lr=0.001, memory_size=50, momentum=0.6,
                     num_class_novel=[0, 9, 11], num_classes=8, period_train=2, save_model=False,
                     schedule_interval=3, std=1.0, std_factor=0.975, stop_acc=0.998, tnorm=3.0, use_cbs=True,
                     use_ls=False, use_tnorm=True)

    model = Resnet18_8s_CBS(args, num_classes=8)
    return model


up_kwargs = {'mode': 'bilinear', 'align_corners': True}


def get_backbone(name, **kwargs):
    models = {
        'resnet18_8s_model_cbs': resnet18_8s_model_cbs,
    }
    name = name.lower()
    print(name)
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net


class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, dilated=True, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/.encoding/models', *args, **kwargs):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        self.backbone = backbone

        self.pretrained = get_backbone(backbone, pretrained=True, dilated=dilated,
                                       norm_layer=norm_layer, root=root,
                                       *args, **kwargs)
        self.pretrained.fc = None
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if self.backbone.startswith('wideresnet'):
            x = self.pretrained.mod1(x)
            x = self.pretrained.pool2(x)
            x = self.pretrained.mod2(x)
            x = self.pretrained.pool3(x)
            x = self.pretrained.mod3(x)
            x = self.pretrained.mod4(x)
            x = self.pretrained.mod5(x)
            c3 = x.clone()
            x = self.pretrained.mod6(x)
            x = self.pretrained.mod7(x)
            x = self.pretrained.bn_out(x)
            return None, None, c3, x

        elif self.backbone.startswith('resnet18_8s_model'):
            x = self.pretrained.resnet18_8s.conv1(x)
            x = self.pretrained.resnet18_8s.bn1(x)
            x = self.pretrained.resnet18_8s.relu(x)
            x = self.pretrained.resnet18_8s.maxpool(x)
            c1 = self.pretrained.resnet18_8s.layer1(x)
            c2 = self.pretrained.resnet18_8s.layer2(c1)
            c3 = self.pretrained.resnet18_8s.layer3(c2)
            c4 = self.pretrained.resnet18_8s.layer4(c3)

        elif self.backbone.startswith('r34_dil'):
            x = self.pretrained.resnet34_8s.conv1(x)
            x = self.pretrained.resnet34_8s.bn1(x)
            x = self.pretrained.resnet34_8s.relu(x)
            x = self.pretrained.resnet34_8s.maxpool(x)
            c1 = self.pretrained.resnet34_8s.layer1(x)
            c2 = self.pretrained.resnet34_8s.layer2(c1)
            c3 = self.pretrained.resnet34_8s.layer3(c2)
            c4 = self.pretrained.resnet34_8s.layer4(c3)

        elif self.backbone.startswith('r50_dil'):
            x = self.pretrained.resnet50_8s.conv1(x)
            x = self.pretrained.resnet50_8s.bn1(x)
            x = self.pretrained.resnet50_8s.relu(x)
            x = self.pretrained.resnet50_8s.maxpool(x)
            c1 = self.pretrained.resnet50_8s.layer1(x)
            c2 = self.pretrained.resnet50_8s.layer2(c1)
            c3 = self.pretrained.resnet50_8s.layer3(c2)
            c4 = self.pretrained.resnet50_8s.layer4(c3)

        elif self.backbone.startswith('r101_dil'):
            x = self.pretrained.resnet101_8s.conv1(x)
            x = self.pretrained.resnet101_8s.bn1(x)
            x = self.pretrained.resnet101_8s.relu(x)
            x = self.pretrained.resnet101_8s.maxpool(x)
            c1 = self.pretrained.resnet101_8s.layer1(x)
            c2 = self.pretrained.resnet101_8s.layer2(c1)
            c3 = self.pretrained.resnet101_8s.layer3(c2)
            c4 = self.pretrained.resnet101_8s.layer4(c3)

        elif self.backbone.startswith('resnet18_8s_endo'):
            x = self.pretrained.resnet18_8s.conv1(x)
            x = self.pretrained.resnet18_8s.bn1(x)
            x = self.pretrained.resnet18_8s.relu(x)
            x = self.pretrained.resnet18_8s.maxpool(x)
            c1 = self.pretrained.resnet18_8s.layer1(x)
            c2 = self.pretrained.resnet18_8s.layer2(c1)
            c3 = self.pretrained.resnet18_8s.layer3(c2)
            c4 = self.pretrained.resnet18_8s.layer4(c3)

        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c = self.pretrained.layer1(x)
            c = self.pretrained.layer2(c)
            c = self.pretrained.layer3(c)
            c = self.pretrained.layer4(c)

        return None, None, None, c

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(
            pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


def module_inference(module, image, flip=True):
    output = module.evaluate(image)
    if flip:
        fimg = flip_image(image)
        foutput = module.evaluate(fimg)
        output += flip_image(foutput)
    return output.exp()


def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)


def pad_image(img, mean, std, crop_size):
    b, c, h, w = img.size()
    assert(c == 3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b, c, h+padh, w+padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:, i, :, :] = F.pad(
            img[:, i, :, :], (0, padw, 0, padh), value=pad_values[i])
    assert(img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size)
    return img_pad


def crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def flip_image(img):
    assert(img.dim() == 4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)


class GCN(nn.Module):

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h


class GloRe_Unit(nn.Module):
    def __init__(self, num_in, num_mid, stride=(1, 1), kernel=1):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)

        # reduce dimension
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),groups=1, bias=False)

        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, x):
        '''
        :param x: (n, c, h, w)
        '''
        batch_size = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul( x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])

        # -----------------
        # final
        out = x + self.blocker(self.fc_2(x_state))

        return out


class GCNet_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(GCNet_Decoder, self).__init__()

        inter_channels = in_channels // 2
        self.decoder = nn.Sequential(nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                                   norm_layer(inter_channels),
                                                   nn.ReLU()),

                                     nn.Sequential(nn.Dropout2d(
                                         0.1), nn.Conv2d(256, out_channels, 1))
                                     )

    def forward(self, x, imsize):
        x = list(tuple([self.decoder(x)]))
        outputs = []
        for i in range(len(x)):
            outputs.append(
                interpolate(x[i], imsize, mode='bilinear', align_corners=True))
        return tuple(outputs)


class GCNet_mod(BaseNet):
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, gcn_search=None, **kwargs):
        super(GCNet_mod, self).__init__(nclass, backbone, aux,  se_loss, norm_layer=norm_layer, **kwargs)

        in_channels = 512
        inter_channels = in_channels // 2

        self.gcn_block = GCN_Unit(in_channels, nclass, norm_layer, gcn_search)
        self.decoder = GCNet_Decoder(in_channels, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]

        # Encoder module
        _, _, _, c4 = self.base_forward(x)

        # GCN with 1 conv block to bridge to GloRE Unit
        x = self.gcn_block(c4)

        # Decoder module
        x = self.decoder(x, imsize)
        return x


class GCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, gcn_search):
        super(GCN_Unit, self).__init__()

        inter_channels = in_channels // 2
        # print(inter_channels, in_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.gcn = nn.Sequential(OrderedDict([("GCN%02d" % i,
                                             GloRe_Unit(
                                                 inter_channels, 64, kernel=1)
                                               ) for i in range(1)]))

    def forward(self, x):
        feat = self.conv51(x)
        feat = self.gcn(feat)

        return feat


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_gcnet(dataset='endovis18', backbone='resnet18_8s_model', num_classes=8, pretrained=False, root='./pretrain_models', **kwargs):
    if backbone == 'resnet18_8s_model_cbs':
        model = GCNet_mod(nclass=num_classes, backbone=backbone, root=root, **kwargs)
    else:
        backbone = 'vanilla_r18'
        model = GCNet_mod(nclass=num_classes, backbone=backbone, root=root, **kwargs)

    return model
