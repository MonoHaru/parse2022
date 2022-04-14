import torch
import torch.nn
from .layers import * 


class UNet(nn.Module):
    def __init__(self, in_channels=2, n_cls=1, n_filters=16, reduction=16, deep_supervision=True):
        super().__init__()

        assert n_filters >= reduction

        self.n_cls = n_cls
        self.deep_supervision = deep_supervision

        ## encoder
        self.left1 = SEConvBlock(in_channels=in_channels, out_channels=n_filters, reduction=reduction)   
        self.left2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            SEConvBlock(in_channels=n_filters, out_channels=n_filters * 2, reduction=reduction)
        )
        self.left3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            SEConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 4, reduction=reduction)
        )

        ## center
        self.center = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            SEConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 8, reduction=reduction)
        )

        ## decoder
        self.upconv3 = ConvTransBlock(
            in_channels=n_filters * 8,
            out_channels=n_filters * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.right3 = SEConvBlock(in_channels=n_filters * 8, out_channels=n_filters * 4, reduction=reduction)  
        
        self.upconv2 = ConvTransBlock(
            in_channels=n_filters * 4,
            out_channels=n_filters * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.right2 = SEConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 2, reduction=reduction)  

        self.upconv1 = ConvTransBlock(
            in_channels=n_filters * 2,
            out_channels=n_filters * 1,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.right1 = SEConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 1, reduction=reduction)  

        
        if self.deep_supervision:
            ## upsample
            self.upsample2 = UpsampleBlock(in_channels=n_filters * 4, out_channels=n_filters * 1, scale=4, reduction=reduction)
            self.upsample1 = UpsampleBlock(in_channels=n_filters * 2, out_channels=n_filters * 1, scale=2, reduction=reduction)

            self.score = nn.Sequential(
                nn.Conv3d(n_filters * 3, n_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ELU(inplace=True),
                nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0, bias=False),
                )
        else:
            self.score = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x): 
        left1 = self.left1(x) 
        left2 = self.left2(left1)
        left3 = self.left3(left2)

        center = self.center(left3)

        if self.deep_supervision:
            x = self.right3(torch.cat([self.upconv3(center), left3], 1))
            up2 = self.upsample2(x)
            x = self.right2(torch.cat([self.upconv2(x), left2], 1))
            up1 = self.upsample1(x)
            x = self.right1(torch.cat([self.upconv1(x), left1], 1))
            hypercol = torch.cat([x, up1, up2], dim=1) ## n_filters * 3, 160, 160, 160
            x = self.score(hypercol)
        else:
            x = self.right3(torch.cat([self.upconv3(center), left3], 1))
            x = self.right2(torch.cat([self.upconv2(x), left2], 1))
            x = self.right1(torch.cat([self.upconv1(x), left1], 1))
            x = self.score(x)        

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)


if __name__ == "__main__":
    from torchsummary import summary

    x = torch.randn(size=(1, 2, 164, 164, 212), device='cpu')
    model = UNet(in_channels=2, n_cls=1, n_filters=16, reduction=16)
    model.to('cpu')
    model.train()
    print(model(x).shape)