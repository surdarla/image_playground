import math
import torch
import torch.nn as nn
from .module import FishTail,FishBody,FishHead,Bridge

def _conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True))

def _bn_relu_conv(in_c, out_c, **conv_kwargs):
    return nn.Sequential(
        nn.BatchNorm2d(in_c),
        nn.ReLU(True),
        nn.Conv2d(in_c, out_c, **conv_kwargs),
    )


class Myfish(nn.Module):
    def __init__(self,in_channels=3,num_classes=10,start_ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels,start_ch//2,stride=2),
            _conv_bn_relu(start_ch//2,start_ch//2),
            _conv_bn_relu(start_ch//2,start_ch),
            nn.MaxPool2d(3, padding=1, stride=2)
        )
        self.tail_layer = nn.ModuleList(
            [FishTail(64,128,2), FishTail(128,256,2),FishTail(256,512,6)]
        )
        self.bridge = Bridge(512,2)
        self.body_layer = nn.ModuleList(
            [FishBody(512,512,1,256,1),
            FishBody(768,384,1,128,1,dilation=2),
            FishBody(512,256,1,64,1,dilation=4)]
        )
        self.head_layer = nn.ModuleList(
            [FishHead(320,320,1,512,1),
            FishHead(832,832,2,768,1),
            FishHead(1600,1600,2,512,4)]
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(2112),
            nn.ReLU(True),
            nn.Conv2d(2112, 2112//2, 1, bias=False),
            nn.BatchNorm2d(2112//2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2112 // 2, num_classes, 1, bias=True)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        stem = self.stem(x)
        tail_features = [stem]
        for t in self.tail_layer:
            last_feature = tail_features[-1]
            tail_features += [ t(last_feature) ]

        bridge = self.bridge(tail_features[-1])

        body_features = [bridge]
        for b, tail in zip(self.body_layer, reversed(tail_features[:-1])):
            last_feature = body_features[-1]
            body_features += [ b(last_feature, tail) ]

        head_features = [body_features[-1]]
        for h, body in zip(self.head_layer, reversed(body_features[:-1])):
            last_feature = head_features[-1]            
            head_features += [ h(last_feature, body) ]

        out = self.classifier(head_features[-1])
        return out
        
 

if __name__ == '__main__':
    model = Myfish()
    from torchsummary import summary
    summary(model,(3,32,32),device='cpu')