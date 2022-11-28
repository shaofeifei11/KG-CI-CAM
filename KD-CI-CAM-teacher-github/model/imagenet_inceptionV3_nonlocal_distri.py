from collections import namedtuple
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable, Any, Optional, Tuple, List
from torch import nn
import torch
import config as cfg
from utils import create_rois, get_pre_two_source_inds, compute_gt_rois, get_mask

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs

class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels,
        pool_features,
        conv_block = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels,
        conv_block = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):
    def __init__(
        self,
        in_channels,
        channels_7x7,
        conv_block = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionD(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_block = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels,
        conv_block = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        conv_block = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        **kwargs
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.mix_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.mix_conv.weight.data.fill_(0.0)
        # self.mix_conv.bias.data.fill_(0.0)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm2d(in_dim)
        nn.init.constant_(self.bn.weight, 0.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.relu = nn.ReLU()

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
        # out = self.relu(self.bn(self.mix_conv(out)))
        out = (self.bn(self.mix_conv(out)))
        out = out + x
        # print(self.gamma)
        return out

class Non_Local_InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, inception_blocks=None, init_weights=None, pretrain=False):
        super(Non_Local_InceptionV3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # self.nonlocal_3 = Self_Attn(80)
        self.nonlocal_4 = Self_Attn(192)
        self.nonlocal_5 = Self_Attn(288)
        self.nonlocal_6 = Self_Attn(768)

        if pretrain:
            self.weights_pretrain()
            # self.weights_init()
            print('pretrained weight load complete..')

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # print("1", x.shape)
        # x = self.nonlocal_3(x) # add sff
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # print("2", x.shape)
        x = self.nonlocal_4(x)  # add sff
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # print("3", x.shape)
        x = self.nonlocal_5(x)  # add sff
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # print("4", x.shape)
        x = self.nonlocal_6(x)  # add sff
        # N x 768 x 17 x 17
        aux = None
        # if self.AuxLogits is not None:
        #     if self.training:
        #         aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        # x = self.Mixed_7a(x)
        # # N x 1280 x 8 x 8
        # x = self.Mixed_7b(x)
        # # N x 2048 x 8 x 8
        # x = self.Mixed_7c(x)
        # # N x 2048 x 8 x 8
        # # Adaptive average pooling
        # x = self.avgpool(x)
        # # N x 2048 x 1 x 1
        # x = self.dropout(x)
        # # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 2048
        # x = self.fc(x)
        # # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    # def forward(self, x: Tensor) -> InceptionOutputs:
    #     x = self._transform_input(x)
    #     x, aux = self._forward(x)
    #     aux_defined = self.training and self.aux_logits
    #     if torch.jit.is_scripting():
    #         if not aux_defined:
    #             warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
    #         return InceptionOutputs(x, aux)
    #     else:
    #         return self.eager_outputs(x, aux)

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        return x

    def weights_pretrain(self):
        pretrained_weights = torch.load('pre_train/inception_v3.pth')
        pretrained_list = pretrained_weights.keys()
        for i, layer_name in enumerate(pretrained_list):
            layer_comp_list = layer_name.split('.')
            if self._find_1(layer_comp_list[0]) == None:
                continue
            elif layer_comp_list[0] in ["Mixed_7a", "Mixed_7b", "Mixed_7c", "fc"]:
                continue

            if len(layer_comp_list) == 2:
                gt = self._find_2(self._find_1(layer_comp_list[0]), layer_comp_list[1])
                assert gt.data.size() == \
                       pretrained_weights[
                           layer_name].size(), "size error!"
                gt.data = pretrained_weights[layer_name]
            elif len(layer_comp_list) == 3:
                gt = self._find_3(self._find_2(self._find_1(layer_comp_list[0]), layer_comp_list[1]), layer_comp_list[2])
                assert gt.data.size() == pretrained_weights[
                            layer_name].size(), "size error!"
                gt.data = pretrained_weights[layer_name]
            elif len(layer_comp_list) == 4:
                gt = self._find_4(self._find_3(self._find_2(self._find_1(layer_comp_list[0]), layer_comp_list[1]),
                                  layer_comp_list[2]), layer_comp_list[3])
                assert gt.data.size() == pretrained_weights[
                            layer_name].size(), "size error!"
                gt.data = pretrained_weights[layer_name]
            else:
                raise "load weights_pretrain error"
        print(" end ###############################################################")
        print("load pre_train inception v3 weights !")

    def _find_4(self, father, comp_name):
        if comp_name == "weight":
            return father.weight
        elif comp_name == "bias":
            return father.bias
        elif comp_name == "running_mean":
            return father.running_mean
        elif comp_name == "running_var":
            return father.running_var
        else:
            raise "not find " + comp_name

    def _find_3(self, father, comp_name):
        if comp_name == "conv":
            return father.conv
        elif comp_name == "bn":
            return father.bn
        elif comp_name == "weight":
            return father.weight
        elif comp_name == "bias":
            return father.bias
        elif comp_name == "running_mean":
            return father.running_mean
        elif comp_name == "running_var":
            return father.running_var
        else:
            raise "not find " + comp_name

    def _find_2(self, father, comp_name):
        if comp_name == "conv":
            return father.conv
        elif comp_name == "bn":
            return father.bn
        elif comp_name == "weight":
            return father.weight
        elif comp_name == "bias":
            return father.bias
        elif comp_name == "branch1x1":
            return father.branch1x1
        elif comp_name == "branch5x5_1":
            return father.branch5x5_1
        elif comp_name == "branch5x5_2":
            return father.branch5x5_2
        elif comp_name == "branch3x3dbl_1":
            return father.branch3x3dbl_1
        elif comp_name == "branch3x3dbl_2":
            return father.branch3x3dbl_2
        elif comp_name == "branch3x3dbl_3":
            return father.branch3x3dbl_3
        elif comp_name == "branch_pool":
            return father.branch_pool
        elif comp_name == "branch3x3":
            return father.branch3x3
        elif comp_name == "branch7x7_1":
            return father.branch7x7_1
        elif comp_name == "branch7x7_2":
            return father.branch7x7_2
        elif comp_name == "branch7x7_3":
            return father.branch7x7_3
        elif comp_name == "branch7x7dbl_1":
            return father.branch7x7dbl_1
        elif comp_name == "branch7x7dbl_2":
            return father.branch7x7dbl_2
        elif comp_name == "branch7x7dbl_3":
            return father.branch7x7dbl_3
        elif comp_name == "branch7x7dbl_4":
            return father.branch7x7dbl_4
        elif comp_name == "branch7x7dbl_5":
            return father.branch7x7dbl_5
        elif comp_name == "branch_pool":
            return father.branch_pool
        elif comp_name == "conv0":
            return father.conv0
        elif comp_name == "conv1":
            return father.conv1
        elif comp_name == "fc":
            return father.fc
        else:
            raise "not find " + comp_name

    def _find_1(self, comp_name):
        if comp_name == "aux_logits":
            return self.aux_logits
        elif comp_name == "transform_input":
            return self.transform_input
        elif comp_name == "Conv2d_1a_3x3":
            return self.Conv2d_1a_3x3
        elif comp_name == "Conv2d_2a_3x3":
            return self.Conv2d_2a_3x3
        elif comp_name == "Conv2d_2b_3x3":
            return self.Conv2d_2b_3x3
        elif comp_name == "maxpool1":
            return self.maxpool1
        elif comp_name == "Conv2d_3b_1x1":
            return self.Conv2d_3b_1x1
        elif comp_name == "Conv2d_4a_3x3":
            return self.Conv2d_4a_3x3
        elif comp_name == "maxpool2":
            return self.maxpool2
        elif comp_name == "Mixed_5b":
            return self.Mixed_5b
        elif comp_name == "Mixed_5c":
            return self.Mixed_5c
        elif comp_name == "Mixed_5d":
            return self.Mixed_5d
        elif comp_name == "Mixed_6a":
            return self.Mixed_6a
        elif comp_name == "Mixed_6b":
            return self.Mixed_6b
        elif comp_name == "Mixed_6c":
            return self.Mixed_6c
        elif comp_name == "Mixed_6d":
            return self.Mixed_6d
        elif comp_name == "Mixed_6e":
            return self.Mixed_6e
        elif comp_name == "AuxLogits":
            return self.AuxLogits
        elif comp_name == "Mixed_7a":
            return self.Mixed_7a
        elif comp_name == "Mixed_7b":
            return self.Mixed_7b
        elif comp_name == "Mixed_7c":
            return self.Mixed_7c
        elif comp_name == "avgpool":
            return self.avgpool
        elif comp_name == "dropout":
            return self.dropout
        elif comp_name == "fc":
            return self.fc
        else:
            raise "not find " + comp_name


class Imagenet_InceptionV3_NonLocal_Distri(nn.Module):
    def __init__(self, args):
        super(Imagenet_InceptionV3_NonLocal_Distri, self).__init__()
        self.num_classes = args.num_classes
        self.args = args
        backbone = Non_Local_InceptionV3(pretrain=self.args.pretrain)

        self.backbone = backbone
        self.gap = nn.AvgPool2d(cfg.attention_size)
        # up branch
        self.up_classifier = nn.Sequential(
            nn.Linear(768, self.num_classes),
        )
        if self.args.shared_classifier:
            self.down_classifier = self.up_classifier
        else:
            self.down_classifier = nn.Sequential(
                nn.Linear(768, self.num_classes),
            )

        self.mask = torch.zeros(size=[self.num_classes, cfg.attention_size, cfg.attention_size], requires_grad=False).to(
            cfg.device)
        self.mask2attention = nn.Conv2d(self.num_classes, 768, 1, 1, 0)

        self.mask_bn = nn.BatchNorm2d(self.num_classes)
        self.upsample = nn.Upsample(size=[cfg.crop_size, cfg.crop_size], mode='bilinear')

    def forward(self, x):
        feature_map = self.backbone(x)
        ############ up ###############
        self.up_feature_map = feature_map
        # print(self.up_feature_map.shape)
        up_vector = self.gap(self.up_feature_map).view(self.up_feature_map.size(0), -1)
        self.up_out = self.up_classifier(up_vector)
        self.pred_sort_up, self.pred_ids_up = torch.sort(self.up_out, dim=-1, descending=True)
        self.up_cam = self._compute_cam(self.up_feature_map, self.up_classifier[0].weight)

        ############ down ###############
        # attention
        if self.args.attention:
            context_list = torch.zeros(
                size=[self.pred_ids_up.shape[0], self.num_classes, cfg.attention_size, cfg.attention_size],
                requires_grad=False).to(cfg.device)
            source_inds = get_pre_two_source_inds(shap=context_list.shape)
            context_list = context_list[source_inds, self.pred_ids_up]  # reverse
            context_list[:, 0] = self.mask[self.pred_ids_up[:, 0]]
            context_list = context_list[source_inds, torch.argsort(self.pred_ids_up, dim=1)]  # reverse back
            temp_attention = self.mask2attention(context_list)

            self.down_feature_map = torch.add(feature_map, temp_attention.mul(feature_map))
        else:
            self.down_feature_map = feature_map

        down_vector = self.gap(self.down_feature_map).view(self.down_feature_map.size(0), -1)
        # down
        self.down_out = self.down_classifier(down_vector)
        self.pred_sort_down, self.pred_ids_down = torch.sort(self.down_out, dim=-1, descending=True)
        self.down_cam = self._compute_cam(self.down_feature_map, self.down_classifier[0].weight)
        # return self.up_cam, self.up_out, self.pred_sort_up, self.pred_ids_up, self.down_cam, self.down_out, self.pred_sort_down, self.pred_ids_down

        diff_out, diff_ids, aux_out, aux_ids, fore_out, fore_ids, back_out = self.aux_classification(x)
        return self.up_cam, self.up_out, self.pred_sort_up, self.pred_ids_up, self.down_cam, self.down_out, self.pred_sort_down, self.pred_ids_down, diff_out, diff_ids, aux_out, aux_ids, fore_out, fore_ids, back_out


    def aux_classification(self, x):
        foreground_mask, background_mask = get_mask(self.pred_ids_down, self.down_cam, self.upsample,
                                                    seg_thr=self.args.seg_thr,
                                                    combination=self.args.combination, function=self.args.function,
                                                    mean_num=self.args.mean_num)
        input = x.permute(1, 0, 2, 3)
        nc, bz, h, w = input.shape
        input = input.reshape((nc, bz, h * w))
        # foreground
        foreground = torch.mul(input, foreground_mask)
        foreground_input = foreground.reshape(nc, bz, h, w)
        foreground_input = foreground_input.permute(1, 0, 2, 3)
        foreground_feature = self.backbone(foreground_input)
        foreground_vector = self.gap(foreground_feature).view(foreground_feature.size(0), -1)
        foreground_out = self.down_classifier(foreground_vector)
        # background
        background = torch.mul(input, background_mask)
        background_input = background.reshape(nc, bz, h, w)
        background_input = background_input.permute(1, 0, 2, 3)
        background_feature = self.backbone(background_input)
        background_vector = self.gap(background_feature).view(background_feature.size(0), -1)
        background_out = self.down_classifier(background_vector)
        # diff
        diff_out = torch.sub(foreground_out, background_out).to(cfg.device)
        aux_out = torch.div(torch.add(diff_out, self.down_out), 2.0).to(cfg.device)
        # _, aux_ids = torch.sort(aux_out, dim=-1, descending=True)
        # _, diff_ids = torch.sort(diff_out, dim=-1, descending=True)
        # _, foreground_ids = torch.sort(foreground_out, dim=-1, descending=True)
        return diff_out, None, aux_out, None, foreground_out, None, background_out


    def update_pred_ids(self, pred_ids_up, pred_ids_down):
        """
        :param up:
        :param down:
        :return:
        """
        self.pred_ids_up = pred_ids_up
        self.pred_ids_down = pred_ids_down
        return pred_ids_up, pred_ids_down


    def update_mask(self, tmp_mask):
        tmp_mask = self.mask_bn(tmp_mask.unsqueeze(dim=0)).to(cfg.device)[0].detach()
        self.mask = torch.add(self.mask, torch.mul(tmp_mask, self.args.update_rate).to(cfg.device)).to(cfg.device)
        self.mask = self.mask_bn(self.mask.unsqueeze(dim=0)).to(cfg.device)[0].detach()


    def _compute_cam(self, input, weight):
        """
        :param input:
        :param weight:
        :return:
        """
        input = input.permute(1, 0, 2, 3)
        nc, bz, h, w = input.shape
        input = input.reshape((nc, bz * h * w))
        cams = torch.matmul(weight, input)
        cams = cams.reshape(self.num_classes, bz, h, w)
        cams = cams.permute(1, 0, 2, 3)
        return cams


    def compute_rois_up(self, seg_thr, topk, combination):
        """
        :param pred:  [batch_size, classes]
        :param cam:  [batch_size, channel, height, width]
        :return:
        """
        return create_rois(self.pred_sort_up, self.pred_ids_up, self.up_cam, self.upsample, seg_thr=seg_thr, topk=topk,
                           combination=combination, function=self.args.function,
                           mean_num=self.args.mean_num)  # [batch, topk, 4]


    def compute_rois_down(self, seg_thr, topk, combination):
        """
        :param pred:  [batch_size, classes]
        :param cam:  [batch_size, channel, height, width]
        :return:
        """
        return create_rois(self.pred_sort_down, self.pred_ids_down, self.down_cam, self.upsample, seg_thr=seg_thr,
                           topk=topk,
                           combination=combination, function=self.args.function,
                           mean_num=self.args.mean_num)  # [batch, topk, 4]


    def compute_gt_rois_down(self, seg_thr, labels, combination):
        """
        :param pred:  [batch_size, classes]
        :param cam:  [batch_size, channel, height, width]
        :return:
        """
        return compute_gt_rois(self.pred_sort_down, self.pred_ids_down, self.down_cam, self.upsample, seg_thr=seg_thr,
                               labels=labels,
                               combination=combination, function=self.args.function,
                               mean_num=self.args.mean_num)  # [batch, 4]


    def compute_gt_rois_up(self, seg_thr, labels, combination):
        """
        :param pred:  [batch_size, classes]
        :param cam:  [batch_size, channel, height, width]
        :return:
        """
        return compute_gt_rois(self.pred_sort_up, self.pred_ids_up, self.up_cam, self.upsample, seg_thr=seg_thr,
                               labels=labels,
                               combination=combination, function=self.args.function,
                               mean_num=self.args.mean_num)  # [batch, 4]

