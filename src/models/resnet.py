import torch
import torch.nn as nn

class ResidualBlock (nn.Module):
    
    def __init__(self,in_channels, out_channels, stride):
        super().__init__()

        self.main_path = nn.Sequential(

            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels), 
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else :
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main_path (x) + self.shortcut (x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        #input
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        #residual blocks

        self.group1 = self._make_group(64, 64, num_blocks=2, stride=1)
        self.group2 = self._make_group(64, 128, num_blocks=2,stride=2)
        self.group3 = self._make_group(128, 256, num_blocks=2,stride=2)
        self.group4 = self._make_group(256, 512, num_blocks=2,stride=2)

        #classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512,num_classes)
        )

        self.initialize_weights()


    def _make_group(self, in_channels, out_channels, num_blocks, stride):
        
        blocks = [ResidualBlock(in_channels, out_channels, stride)]

        for i in range (1,num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels,stride=1))
        
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.classifier(x)

        return x
    
    def initialize_weights (self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == "__main__" :
    model = ResNet(num_classes=10)

    dummy = torch.randn(4,3,32,32)
    out=model(dummy)

    print(f"shape of dummy {dummy.shape}")
    print(f"shape of out{out.shape}")
    print(f"parameters {model.count_parameters()}")

    loss = out.sum()
    loss.backward()
    first_conv=model.stem[0].weight.grad
    print(f"{first_conv is not None}")
    print(f"{first_conv.norm()}")