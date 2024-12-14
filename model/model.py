import torch
import torch.nn as nn
import antialiased_cnns
from torch.nn import Conv2d, LeakyReLU, Sequential, Upsample

#define a downlayer
class DownLaye(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownLayer, self).__init__()
        '''logic behind using the antialiased CNNS: 
           talked about in class, where aliasing artifacts discard high freq info. 
           but for faces, this stuff is important - like wrinkles, fine lines, are all important
           i simplify the downsampling but maybe we need to add the maxpooling as well? potensh ablation'''
        self.layer = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            antialiased_cnns.BlurPool(out_channels, stride=1), 
            LeakyReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

#define an uplayer
class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpLayer, self).__init__()
        '''upsample vs blur_upsample : if we have time, we can experiment with this as an ablation study. 

           upsample: interpolates pixel vals smoothly - increases spatial res. mayb not as good at preserving fine details
           blur_upsample:  reduces artifacts (antialiasing) and preserves high freq detail. computationally more expensive tho (which makes sense)
                           lowkey in the context of our project, we might need to tank this and just use blur_upsample but something tells me that gonna take
                           farrrr longer to train. '''
        # self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.blur_upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            antialiased_cnns.BlurPool(out_channels, stride=1)
        )

        self.layer = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            LeakyReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            LeakyReLU()
        )

    def forward(self, x, skip):
        x = self.blur_upsample(x)
        x = torch.cat([x, skip], dim=1)  # skip connection - part of unet u wanna connect down to up at every level
        return self.layer(x)

#define the unet architecture
 
#define the discriminator

'''Patch Gan Discriminator: often used for trasnlation tasks, image->image usually (see: https://paperswithcode.com/method/patchgan  ).
   It's been out there for awhile

   Patch gan vs regular gan is (as name imples) patch->patch checking for loss; similar to a cnn layer basically.. essentially used instead of reg because small details in face really matter.

   Thought this was a great resource: https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207 
   Output is n x n output vector instead of a single scalar value.

   
   
   ''' 