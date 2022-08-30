import torch
import torch.nn as nn

from T4T.Block.ConvBlock import ConvBn2D

##############################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

##############################################################

class CodingBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(CodingBlock, self).__init__()
        self.conv_add = ConvBn2D(in_planes, planes, stride=stride, kernel_size=3)
        self.conv1 = ConvBn2D(planes, planes, stride=stride, kernel_size=3)
        self.conv2 = ConvBn2D(planes, planes, stride=stride, kernel_size=1)

    def forward(self, x):
        out_add = self.conv_add(x)
        out = self.conv1(out_add)
        out = self.conv2(out)
        out = out + out_add
        return out


##############################################################
class EncodingPart(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(EncodingPart, self).__init__()
        self.encoding_block1 = CodingBlock(in_planes, planes, stride)
        self.pool1 = nn.MaxPool2d(2)
        self.encoding_block2 = CodingBlock(planes, planes * 2, stride)
        self.pool2 = nn.MaxPool2d(2)
        self.encoding_block3 = CodingBlock(planes * 2, planes * 4, stride)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        encoding_list = []

        conv1 = self.encoding_block1(x)
        pool1 = self.pool1(conv1)
        encoding_list.append(conv1)

        conv2 = self.encoding_block2(pool1)
        pool2 = self.pool2(conv2)
        encoding_list.append(conv2)

        conv3 = self.encoding_block3(pool2)
        pool3 = self.pool3(conv3)
        encoding_list.append(conv3)

        return encoding_list, pool3


class DecodingPart(nn.Module):
    def __init__(self, in_planes, planes, stride=1, encode_num=1):
        super(DecodingPart, self).__init__()
        self.decoding_block1 = CodingBlock(in_planes*(4*encode_num), planes, stride)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.decoding_block2 = CodingBlock(in_planes*(8+4*encode_num), planes // 2, stride)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.decoding_block3 = CodingBlock(in_planes*(4+2*encode_num), planes // 4, stride)
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.decoding_block4 = CodingBlock(in_planes*(2+1*encode_num), planes // 8, stride)


    def forward(self, input_list, encoding_list):
        inputs = torch.cat(input_list, dim=1)
        conv1 = self.decoding_block1(inputs)
        up1 = self.upsample1(conv1)

        conv2 = self.decoding_block2(torch.cat((up1, encoding_list[2]), dim=1))
        up2 = self.upsample2(conv2)

        conv3 = self.decoding_block3(torch.cat((up2, encoding_list[1]), dim=1))
        up3 = self.upsample3(conv3)

        conv4 = self.decoding_block4(torch.cat((up3, encoding_list[0]), dim=1))

        return conv4

class LastPart(nn.Module):
    def __init__(self, in_planes, planes, out_channels=1, stride=1):
        super(LastPart, self).__init__()
        self.conv1 = ConvBn2D(in_planes, planes, stride=stride, kernel_size=3)
        self.conv2 = ConvBn2D(planes, planes, stride=stride, kernel_size=1)

        self.last_conv = nn.Conv2d(planes, out_channels, stride=1, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.last_conv(x)
        if self.out_channels == 1:
            return torch.sigmoid(x)
        else:
            return nn.functional.softmax(x, dim=1)


##############################################################
class TrumpetNetWithROI(nn.Module):
    def __init__(self, in_planes, planes, stride=1, encode_num=1):
        super(TrumpetNetWithROI, self).__init__()
        self.encoding_path1 = EncodingPart(in_planes, planes, stride)
        self.encoding_path2 = EncodingPart(in_planes, planes, stride)
        self.encoding_path3 = EncodingPart(in_planes, planes, stride)

        self.deconding = DecodingPart(planes, planes*8, stride, encode_num=encode_num)

        self.pirads_last = LastPart(planes + 3, planes, out_channels=4)
        self.pca_last = LastPart(planes + 7, planes, out_channels=1)


    def forward(self, t2, adc, dwi, roi):
        t2_encoding, t2_out = self.encoding_path1(t2)
        adc_encoding, adc_out = self.encoding_path1(adc)
        dwi_encoding, dwi_out = self.encoding_path1(dwi)

        out_list = [t2_out, adc_out, dwi_out]
        encoding_list = []
        for index in range(len(t2_encoding)):
            encoding_list.append(torch.cat([t2_encoding[index], adc_encoding[index], dwi_encoding[index]], dim=1))

        out = self.deconding(out_list, encoding_list)

        pirads_out = torch.cat([out, roi], dim=1)
        pirads_out = self.pirads_last(pirads_out)

        pca_out = torch.cat([out, roi, pirads_out], dim=1)
        pca_out = self.pca_last(pca_out)

        return pca_out, pirads_out


if __name__ == '__main__':
    model = TrumpetNetWithROI(in_planes=3, planes=32, stride=1, encode_num=3)
    # print(model)
    inputs1 = torch.randn(1, 3, 192, 192)
    inputs2 = torch.randn(1, 3, 192, 192)
    inputs3 = torch.randn(1, 3, 192, 192)
    inputs4 = torch.randn(1, 3, 192, 192)
    pca, pirads = model(inputs1, inputs2, inputs3, inputs4)
    print(pca.shape, pirads.shape)

