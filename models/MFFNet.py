import time

import torch
import torch.nn as nn
import torch.nn.functional as F
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 主分支卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out

# 下采样卷积
class DownUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownUnit, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.conv_fuse = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out = torch.cat([out3, out5], dim=1)
        out = self.bn(self.conv_fuse(out))
        residual = self.residual_conv(x)
        out += residual
        out = self.relu(out)
        return out

class AttentionFusionModule(nn.Module):
    def __init__(self, high_res_channels, low_res_channels):
        super(AttentionFusionModule, self).__init__()
        self.conv_low_res = nn.Conv2d(low_res_channels, high_res_channels, kernel_size=1, stride=1)
        self.attention = nn.Sequential(
            nn.Conv2d(high_res_channels * 2, high_res_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(high_res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_res_channels, high_res_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.final = nn.Sequential(
            nn.Conv2d(high_res_channels * 2, high_res_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(high_res_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, high_res_feat, low_res_feat):
        low_res_feat_up = F.interpolate(low_res_feat, size=high_res_feat.shape[2:], mode='bilinear', align_corners=False)
        low_res_feat_up = self.conv_low_res(low_res_feat_up)
        concat_feat = torch.cat([high_res_feat, low_res_feat_up], dim=1)
        attention_weights = self.attention(concat_feat)
        low_res_feat_weighted = low_res_feat_up * attention_weights
        low_res_feat_weighted = torch.cat([high_res_feat, low_res_feat_weighted], dim=1)
        fused_feat = self.final(low_res_feat_weighted)

        return fused_feat

class MutiFusion(nn.Module):
    def __init__(self, h_channels, m_channels, l_channels):
        super(MutiFusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.low_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=m_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        self.low_2 = nn.Sequential(
            nn.Conv2d(in_channels=m_channels, out_channels=l_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(l_channels),
        )
        self.mid_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=m_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        self.mid_2 = nn.Sequential(
            nn.Conv2d(in_channels=l_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        self.high_1 = nn.Sequential(
            nn.Conv2d(in_channels=l_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        self.high_2 = nn.Sequential(
            nn.Conv2d(in_channels=m_channels, out_channels=h_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(h_channels),
        )

    def forward(self, h, m, l):
        l1 = self.relu(self.low_1(h) + m)
        l1 = self.relu(self.low_2(l1) + l)

        h1 = self.relu(F.interpolate(self.high_1(l), size=m.shape[2:], mode='bilinear', align_corners=False) + m)
        h1 = self.relu(F.interpolate(self.high_2(h1), size=h.shape[2:], mode='bilinear', align_corners=False) + h)
        m = self.relu(F.interpolate(self.mid_2(l), size=m.shape[2:], mode='bilinear', align_corners=False) + m)

        m = self.relu(self.mid_1(h) + m)

        return h1 , m , l1

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, num_channels)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, num_channels, 1, 1)
        return y  # 返回通道注意力权重
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_pool, max_pool], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return y  # 返回空间注意力权重
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        residual = x  # 保存输入的残差连接

        # 计算通道注意力和空间注意力权重
        channel_weight = self.channel_attention(x)
        spatial_weight = self.spatial_attention(x)

        # 将通道注意力和空间注意力并行作用于输入x
        x_channel = x * channel_weight  # 通道注意力作用
        x_spatial = x * spatial_weight  # 空间注意力作用

        # 将两者加和
        out = x_channel + x_spatial + residual
        return out

# 分割头
class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear')

        return out

# 分割网络主干网络
class SemanticSegmentationNet(nn.Module):
    def __init__(self, num_classes=19, aux_mode='train'):
        super(SemanticSegmentationNet, self).__init__()
        self.aux_mode = aux_mode

        self.s1 = nn.Sequential(
            DownUnit(3, 32),
            ResidualBlock(32, 32),
            DownUnit(32, 32),
            ResidualBlock(32, 32),
            DownUnit(32, 64)
        )

        self.s1_p1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.s1_p2 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.s1_p3 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )

        self.s1_p = nn.Sequential(
            ResidualBlock(64, 64),
            CBAM(64)
        )


        self.s2 = nn.Sequential(
            DownUnit(64, 128),
        )
        self.s2_p1 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        self.s2_p2 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        self.s2_p3 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        self.s2_p = nn.Sequential(
            ResidualBlock(128, 128),
            CBAM(128)
        )

        self.s3 = nn.Sequential(
            DownUnit(128, 256),
        )
        self.s3_p1 = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        self.s3_p2 = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        self.s3_p3 = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        self.s3_p = nn.Sequential(
            ResidualBlock(256, 256),
            CBAM(256)
        )

        self.change1 = MutiFusion(64, 128, 256)
        self.change2 = MutiFusion(64, 128, 256)
        self.change3 = MutiFusion(64, 128, 256)

        self.up_mid = AttentionFusionModule(128, 256, )
        self.d_mid = ResidualBlock(128, 128)

        self.up_high = AttentionFusionModule(64, 128, )
        self.d_high = ResidualBlock(64, 64)
        self.conv = ResidualBlock(64, 64)
        self.final_layer = segmenthead(64, 128, num_classes, 8)
        self.loss1 = segmenthead(64, 128, num_classes, 8)

    def forward(self, x):
        # 输入层
        xtem = self.s1(x)  # [B, 128, H/8, W/8]
        x = self.s1_p1(xtem)
        x = self.s1_p2(x)
        x = self.s1_p3(x)


        xm = self.s2(x)  # [B, 256, H/16, W/16]
        xm = self.s2_p1(xm)  # [B, 256, H/16, W/16]
        xm = self.s2_p2(xm)  # [B, 256, H/16, W/16]
        xm = self.s2_p3(xm)  # [B, 256, H/16, W/16]


        xl = self.s3(xm)  # [B, 512, H/32, W/32]
        xl = self.s3_p1(xl)  # [B, 512, H/32, W/32]
        x, xm, xl = self.change1(x, xm, xl)
        xl = self.s3_p2(xl)  # [B, 512, H/32, W/32]
        x, xm, xl = self.change2(x, xm, xl)
        xl = self.s3_p3(xl)  # [B, 512, H/32, W/32]
        x, xm, xl = self.change3(x, xm, xl)

        x = self.s1_p(x)
        xm = self.s2_p(xm)
        xl = self.s3_p(xl)


        xm = self.d_mid(self.up_mid(xm, xl))
        x = self.d_high(self.up_high(x, xm))

        x = self.conv(x)

        out = self.final_layer(x)

        if self.aux_mode == 'train':
            loss0 = self.loss1(xtem)
            return out, loss0
        elif self.aux_mode == 'eval':
            return out,
        elif self.aux_mode == 'pred':
            pred = out.argmax(dim=1)
            return pred
        else:
            raise NotImplementedError


if __name__ == '__main__':
    device = torch.device('cuda')
    model = SemanticSegmentationNet(num_classes=19)
    model.eval()
    model.to(device)
    iterations = None
    input = torch.randn(1, 3, 640, 960).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)