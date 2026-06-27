import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,
                      stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.block(x)
    

class DeepCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        
        self.group1= nn.Sequential(
            ConvBlock(3,64),
            ConvBlock(64,64)
        )

        self.group2 = nn.Sequential(
            ConvBlock(64,128,stride=2),
            ConvBlock(128,128)
        )

        self.group3 = nn.Sequential(
            ConvBlock(128,256,stride=2),
            ConvBlock(256,256)
        )

        self.group4= nn.Sequential(
            ConvBlock(256,512,stride=2),
            ConvBlock(512,512)
        )



        #classifier

        self.classifier=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
                            
        )

        #initialize weights
        self.initialize_weights()


    def forward(self, x):
        x=self.group1(x)
        x=self.group2(x)
        x=self.group3(x)
        x=self.group4(x)
        x=self.classifier(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules() :
            if isinstance (m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                         nonlinearity="relu")
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)

    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = DeepCNN(num_classes=10) 
    # تست سریع با یک batch ساختگی
    dummy = torch.randn(4, 3, 32, 32)   # batch=4, CIFAR-10 size
    out   = model(dummy)
 
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {model.count_parameters():,}")
    print()
    print(model)