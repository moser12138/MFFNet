import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import xlog1py

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

# 标准残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # 是否需要下采样（stride不为1时通常需要）
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 主分支卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 如果需要下采样，调整恒等映射
        if self.downsample is not None:
            identity = self.downsample(x)
        # 将恒等映射与卷积输出相加
        out += identity
        out = self.relu2(out)
        return out

# 下采样卷积
class DownUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownUnit, self).__init__()
        # 3x3卷积核下采样
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # 5x5卷积核下采样
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        # 拼接后的3x3卷积核处理
        self.conv_fuse = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        # 残差连接
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 两路卷积下采样
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        # 通道拼接
        out = torch.cat([out3, out5], dim=1)
        # 3x3卷积处理
        out = self.bn(self.conv_fuse(out))
        # 残差连接
        residual = self.residual_conv(x)
        # 残差相加
        out += residual
        out = self.relu(out)
        return out

# 注意力特征融合
class AttentionFusionModule(nn.Module):
    def __init__(self, high_res_channels, low_res_channels):
        super(AttentionFusionModule, self).__init__()
        # 上采样低分辨率特征使其分辨率与高分辨率特征一致
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # 将低分辨率特征的通道数降至与高分辨率特征一致
        self.conv_low_res = nn.Conv2d(low_res_channels, high_res_channels, kernel_size=1, stride=1)
        # 注意力机制：生成一个通道注意力权重
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
        # 上采样低分辨率特征
        # low_res_feat_up = self.upsample(low_res_feat)
        low_res_feat_up = F.interpolate(low_res_feat, size=high_res_feat.shape[2:], mode='bilinear', align_corners=False)
        # 通过1x1卷积将低分辨率特征的通道数缩减到与高分辨率特征一致
        low_res_feat_up = self.conv_low_res(low_res_feat_up)
        # 拼接高分辨率特征和上采样后的低分辨率特征，在通道维度上进行
        concat_feat = torch.cat([high_res_feat, low_res_feat_up], dim=1)
        # 通过注意力机制生成权重
        attention_weights = self.attention(concat_feat)
        # 使用注意力权重调整低分辨率特征的贡献

        low_res_feat_weighted = low_res_feat_up * attention_weights
        # 将加权后的低分辨率特征与高分辨率特征相加，进行融合
        low_res_feat_weighted = torch.cat([high_res_feat, low_res_feat_weighted], dim=1)
        # fused_feat = high_res_feat + low_res_feat_weighted
        fused_feat = self.final(low_res_feat_weighted)

        return fused_feat

# 多分支融合
class MutiFusion(nn.Module):
    def __init__(self, h_channels, m_channels, l_channels):
        super(MutiFusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.low_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=m_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        # self.low_modify1 = nn.Conv2d(in_channels=2 * m_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False)
        self.low_2 = nn.Sequential(
            nn.Conv2d(in_channels=m_channels, out_channels=l_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(l_channels),
        )
        # self.low_modify2 = nn.Conv2d(in_channels=2 * l_channels, out_channels=l_channels, kernel_size=1, stride=1, bias=False)


        self.mid_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=m_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        # self.mid_modify1 = nn.Conv2d(in_channels=2 * m_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False)
        self.mid_2 = nn.Sequential(
            nn.Conv2d(in_channels=l_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        # self.mid_modify2 = nn.Conv2d(in_channels=2 * m_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False)


        self.high_1 = nn.Sequential(
            nn.Conv2d(in_channels=l_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(m_channels),
        )
        # self.high_modify1 = nn.Conv2d(in_channels=2 * m_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False)
        self.high_2 = nn.Sequential(
            nn.Conv2d(in_channels=m_channels, out_channels=h_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(h_channels),
        )
        # self.high_modify2 = nn.Conv2d(in_channels=2 * h_channels, out_channels=h_channels, kernel_size=1, stride=1, bias=False)


    def forward(self, h, m, l):
        l1 = self.relu(self.low_1(h) + m)
        l1 = self.relu(self.low_2(l1) + l)

        # h1 = self.relu(F.interpolate(self.high_1(l), scale_factor=2, mode='bilinear', align_corners=False) + m)
        h1 = self.relu(F.interpolate(self.high_1(l), size=m.shape[2:], mode='bilinear', align_corners=False) + m)

        # h1 = self.relu(F.interpolate(self.high_2(h1), scale_factor=2, mode='bilinear', align_corners=False) + h)
        h1 = self.relu(F.interpolate(self.high_2(h1), size=h.shape[2:], mode='bilinear', align_corners=False) + h)

        # m = self.relu(F.interpolate(self.mid_2(l), scale_factor=2, mode='bilinear', align_corners=False) + m)
        m = self.relu(F.interpolate(self.mid_2(l), size=m.shape[2:], mode='bilinear', align_corners=False) + m)

        m = self.relu(self.mid_1(h) + m)

        return h1 , m , l1

# 混合注意力机制
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
        return x * y
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
        return x * y
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        residual = x  # 保存输入的残差连接
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = x + residual  # 残差连接
        return x

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
    def __init__(self, num_classes=19, augment=True):
        super(SemanticSegmentationNet, self).__init__()
        self.augment = augment

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
            # ResidualBlock(64, 64),
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

        # 输入层
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
        # return out
        if self.augment:
            loss0 = self.loss1(xtem)
            return [loss0, out]
        else:
            return out

if __name__ == '__main__':

    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = SemanticSegmentationNet(num_classes=19)
    model.eval()
    model.to(device)
    iterations = None

    # input = torch.randn(1, 3, 1024, 2048).cuda()
    input = torch.randn(1, 3, 640, 960).cuda()

    with torch.no_grad():  # 上下文管理器来禁用梯度计算，从而减少内存消耗，并且不记录计算历史
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
