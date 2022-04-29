import torch
import torch.nn as nn

class _ConvBlock(nn.Module):
    """
    Construct Basic Bottleneck Convolution Block module.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        stride : Stride of middle conv layer
        dilation : Dilation rate of middle conv layer
    Forwarding Path:
        input image - (BN-ReLU-Conv) * 3 - output
    """
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super(_ConvBlock, self).__init__()

        mid_c = out_c // 4
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
            nn.Conv2d(in_c, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(True),
            nn.Conv2d(mid_c, out_c, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.layers(x)

class TransferBlock(nn.Module):
    """
    Construct Transfer Block module.
    
    Args:
        ch : Number of channels in the input and output image
        num_blk : Number of Residual Blocks
    Forwarding Path:
        input image - (ConvBlock) * num_blk - output
    """
    def __init__(self, ch, num_blk):
        super().__init__()

        self.layers = nn.Sequential(
            *[_ConvBlock(ch, ch) for _ in range(0, num_blk)]
        )

    def forward(self, x):
        return self.layers(x)


class DR(nn.Module):
    """
    Construct Down-RefinementBlock module. (DRBlock from the original paper)
    Consisted of one Residual Block and Conv Blocks.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer
    Forwarding Path:
                    ⎡      (BN-ReLU-Conv)     ⎤
        input image - (ConvBlock) * num_blk -(sum)- feature - (MaxPool) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1):
        super().__init__()

        self.res = _ConvBlock(in_c, out_c)
        self.regular_connection = nn.Sequential(
            *[_ConvBlock(out_c, out_c) for _ in range(1, num_blk)]
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        out = self.res(x)
        shortcut = self.shortcut(x)
        out = self.regular_connection(out + shortcut)
        return self.pool(out)


class UR(nn.Module):
    """
    Construct Up-RefinementBlock module. (URBlock from the original paper)
    Consisted of Residual Block and Conv Blocks.
    Not like DRBlock, this module reduces the number of channels of concatenated feature maps in the shortcut path.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer
    Forwarding Path:
                    ⎡      (BN-ReLU-Conv)     ⎤
        input image - (ConvBlock) * num_blk -(sum)- feature - (UpSample) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1, dilation=1):
        super().__init__()
        self.k = in_c // out_c
        self.res = _ConvBlock(in_c, out_c, dilation=dilation)
        self.regular_connection = nn.Sequential(
            *[_ConvBlock(out_c, out_c, dilation=dilation) for _ in range(1, num_blk)]
        )
      
        self.upsample = nn.Upsample(scale_factor=2)

    def channel_reduction(self, x):
        n, c, *dim = x.shape
        return x.view(n, c // self.k, self.k, *dim).sum(2)
        
    def forward(self, x):
        out = self.res(x)
        shortcut = self.channel_reduction(x)
        out = self.regular_connection(out + shortcut)
        return self.upsample(out)
    
    
class FishTail(nn.Module):
    """
    Construct FishTail module.
    Each instances corresponds to each stages.
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
    Forwarding Path:
        input image - (DRBlock) - output
    """
    def __init__(self, in_c, out_c, num_blk):
        super().__init__()
        self.layer = DR(in_c, out_c, num_blk)

    def forward(self, x):
        return self.layer(x)

class Bridge(nn.Module):
    """
    Construct Bridge module.
    This module bridges the last FishTail stage and first FishBody stage.
    
    Args:
        ch : Number of channels in the input and output image
        num_blk : Number of Residual Blocks
    Forwarding Path:
                        r             (SEBlock)         ㄱ 
        input image - (stem) - (_ConvBlock)*num_blk - (mul & sum) - output
    """         
    def __init__(self, ch, num_blk):
        super().__init__()

        self.stem = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, ch//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch//2),
            nn.ReLU(True),
            nn.Conv2d(ch//2, ch*2, kernel_size=1, bias=True)
        )

        self.conv =_ConvBlock(ch*2, ch)
        self.layers = nn.Sequential(
            *[_ConvBlock(ch, ch) for _ in range(1, num_blk)]
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.Conv2d(ch*2, ch, kernel_size=1, bias=False)
        )

        # https://github.com/kevin-ssy/FishNet/blob/master/models/fishnet.py#L45
        self.se_block = nn.Sequential(
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch*2, ch//16, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(ch//16, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        att = self.se_block(x)
        
        out = self.conv(x)
        shortcut = self.shortcut(x)
        out = self.layers(out + shortcut)
        return (out * att) + att


  
class FishBody(nn.Module):
    r"""Construct FishBody module.
    Each instances corresponds to each stages.
    
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        dilation : Dilation rate of Conv in UR
        
    Forwarding Path:
        input image - (URBlock)  ㄱ
        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk,
                 trans_in_c, num_trans,
                 dilation=1):
        super().__init__()
        self.layer = UR(in_c, out_c, num_blk, dilation=dilation)
        self.transfer = TransferBlock(trans_in_c, num_trans)

    def forward(self, x, trans_x):
        x = self.layer(x)
        trans_x = self.transfer(trans_x)
        return torch.cat([x, trans_x], dim=1)

class FishHead(nn.Module):
    r"""Construct FishHead module.
    Each instances corresponds to each stages.
    Different with Offical Code : we used shortcut layer in this Module. (shortcut layer is used according to the original paper)
    
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        
    Forwarding Path:
        input image - (DRBlock)  ㄱ
        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk,
                 trans_in_c, num_trans):
        super().__init__()
        self.layer = DR(in_c, out_c, num_blk)
        self.transfer = TransferBlock(trans_in_c, num_trans)

    def forward(self, x, trans_x):
        x = self.layer(x)
        trans_x = self.transfer(trans_x)
        return torch.cat([x, trans_x], dim=1)