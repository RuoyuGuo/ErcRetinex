import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GuidedAtt(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, x_guide):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]       
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        
        # print(x_in.shape)
        # print(x_guide.shape)
        
        x = x_in.reshape(b, h * w, c)
        x_g = x_guide.reshape(b, h * w, c)
        
        q_inp = self.to_q(x)
        k_inp = self.to_k(x_g)
        v_inp = self.to_v(x_g)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))
        v = v 
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

    
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class FormerBlock(nn.Module):
    def __init__(
            self,
            base_ch,
            dim=32,
    ):
        super().__init__()
        self.block1 = GuidedAtt(dim=base_ch, dim_head=dim, heads=base_ch//dim)
        self.block2 = PreNorm(base_ch, FeedForward(dim=base_ch))

    def forward(self, x, x_guide):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        x_guide = x_guide.permute(0, 2, 3, 1)
        x = self.block1(x, x_guide) + x
        x = self.block2(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False),
                                nn.GELU(),
                                nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_guide):
        avg_out = self.fc(self.avg_pool(x)*self.avg_pool(x_guide))
        max_out = self.fc(self.max_pool(x)*self.max_pool(x_guide))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_guide):
        avg_out = torch.mean(x, dim=1, keepdim=True) * torch.mean(x_guide, dim=1, keepdim=True)
        
        max_out, _ = torch.max(x, dim=1, keepdim=True) 
        max_out_guide, _ = torch.max(x_guide, dim=1, keepdim=True)
        max_out = max_out * max_out_guide
        
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
    
class CnnBlock(nn.Module):
    def __init__(self, base_ch, ratio=16):
        super().__init__()
        self.conv1 = nn.Conv2d(base_ch, base_ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(base_ch, base_ch, 3, 1, 1, bias=False)
        self.gelu = nn.GELU()

        self.ca = ChannelAttention(in_planes=base_ch, ratio=ratio)
        self.sa = SpatialAttention()


    def forward(self, x, x_guide):
        residual = x

        out = self.conv1(x)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.gelu(out)

        out = self.ca(out, x_guide) * out
        out = self.sa(out, x_guide) * out

        out += residual
        out = self.gelu(out)

        return out

        
class TransformerBlock(nn.Module):
    def __init__(self, base_ch, num_blocks):
        super().__init__()
        self.basics = nn.ModuleList([])
        
        for i in range(num_blocks):
            self.basics.append(nn.ModuleList([
                FormerBlock(base_ch=base_ch),
                CnnBlock(base_ch=base_ch),
                nn.Conv2d(base_ch*2, base_ch, 3, 1, 1, bias=False),
            ]))
        
    def forward(self, x, guide_fea):
         
        for (former_basic, cnn_basic, fuse_basic) in self.basics:
            x_former = former_basic(x, guide_fea)
            x_cnn    = cnn_basic(x, guide_fea)
            x = fuse_basic(torch.cat([x_former, x_cnn], dim=1))
        
        return x
    
    def get_lg_feature(self, x, guide_fea):
        x_g = []
        x_l = []
        
        for (former_basic, cnn_basic, fuse_basic) in self.basics:
            x_former = former_basic(x, guide_fea)
            x_cnn    = cnn_basic(x, guide_fea)
            x = fuse_basic(torch.cat([x_former, x_cnn], dim=1))
            
            x_g.append(x_former)
            x_l.append(x_cnn)
        
        return x, x_g, x_l

class TransformerEncoder(nn.Module):
    def __init__(self, in_ch, base_ch, num_blocks):
        super().__init__()
        self.shallowEmbedding = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1, bias=False),)
        
        self.en1 = TransformerBlock(base_ch=base_ch*1, num_blocks=num_blocks[0])
        self.en2 = TransformerBlock(base_ch=base_ch*2, num_blocks=num_blocks[1])
        self.en3 = TransformerBlock(base_ch=base_ch*4, num_blocks=num_blocks[2])
        
        self.down1_x     = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1, bias=False)
        self.down1_guide = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1, bias=False)
        
        self.down2_x     = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1, bias=False)
        self.down2_guide = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1, bias=False)
    
    def forward(self, x, guide_fea):
        x_shallow = self.shallowEmbedding(x)      #32 * H * W
        x_en1 = self.en1(x_shallow, guide_fea)               #32 * H * W
        
        x_en2, guide_fea2 = self.down1_x(x_en1), self.down1_guide(guide_fea)
        x_en2 = self.en2(x_en2, guide_fea2)       #64 * H/2 * W/2
        
        x_en3, guide_fea3 = self.down2_x(x_en2),  self.down2_guide(guide_fea2)
        x_en3 = self.en3(x_en3, guide_fea3)       #128 * H/4 * W/4
        
        return (x_shallow, x_en1, x_en2, x_en3), (guide_fea, guide_fea2, guide_fea3)
        
    def get_lg_fea(self, x, guide_fea):
        x_shallow = self.shallowEmbedding(x)      #32 * H * W
        x_en1, x_g, x_l = self.en1.get_lg_feature(x_shallow, guide_fea)               #32 * H * W
        
        x_en2, guide_fea2 = self.down1_x(x_en1), self.down1_guide(guide_fea)
        x_en2, x_g, x_l = self.en2.get_lg_feature(x_en2, guide_fea2)       #64 * H/2 * W/2
        
        x_en3, guide_fea3 = self.down2_x(x_en2),  self.down2_guide(guide_fea2)
        x_en3 = self.en3(x_en3, guide_fea3)       #128 * H/4 * W/4
        
        return (x_shallow, x_en1, x_en2, x_en3), (guide_fea, guide_fea2, guide_fea3), x_g, x_l
        
        
class TransformerDecoder(nn.Module):
    def __init__(self, base_ch, num_blocks):
        super().__init__()
        
        self.de2 = TransformerBlock(base_ch=base_ch*2, num_blocks=num_blocks[0])
        self.de1 = TransformerBlock(base_ch=base_ch*1, num_blocks=num_blocks[1])
        
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0)
        self.fuse2 = nn.Conv2d(base_ch*4, base_ch*2, 1, 1, bias=False,)
        
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch*1, stride=2,
                                   kernel_size=2, padding=0, output_padding=0)
        self.fuse1 = nn.Conv2d(base_ch*2, base_ch*1, 1, 1, bias=False,)
        
        # self.fuse_layer = nn.Sequential(nn.Conv2d(base_ch*2, base_ch*1, 3, 1, 1, bias=False),
        #                               nn.Conv2d(base_ch*1, base_ch*1, 3, 1, 1, bias=False),)
        
        self.out_layer  = nn.Sequential(nn.Conv2d(base_ch*1, base_ch*1, 3, 1, 1, bias=False),
                                      nn.Conv2d(base_ch*1, base_ch*1, 3, 1, 1, bias=False),
                                        nn.Conv2d(base_ch*1, 3, 3, 1, 1, bias=False),
                                        nn.Sigmoid()) 
        
    def forward(self, encoder_feature, multi_guide_fea):
        x_shallow, x_en1, x_en2, x_en3 = encoder_feature
        guide_fea, guide_fea2, guide_fea3 = multi_guide_fea
        
        x_de2 = self.fuse2(torch.cat([self.up2(x_en3), x_en2], dim=1))
        x_de2 = self.de2(x_de2, guide_fea2)   
        
        x_de1 = self.fuse1(torch.cat([self.up1(x_de2), x_en1], dim=1))
        x_de1 = self.de1(x_de1, guide_fea) 
        
        # x_out = self.fuse_layer(x_de1)
        x_out = self.out_layer(x_de1)
        
        return x_out

class IllGuideLayer(nn.Module):
    def __init__(self, base_ch, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, 1, bias=True)
        self.dwconv1 = nn.Conv2d(base_ch, base_ch, 3, 1, 1, bias=True, groups=in_ch)
        self.dwconv2 = nn.Conv2d(base_ch, base_ch, 3, 1, 1, bias=True, groups=in_ch)
        self.dwconv3 = nn.Conv2d(base_ch, base_ch, 3, 1, 1, bias=True, groups=in_ch)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv1(x)
        x = self.dwconv2(x)
        x = self.dwconv3(x)

        return x

class EnhanceNet(nn.Module):

    def __init__(self, in_ch, base_ch, guide_ch, num_enblocks, num_deblocks):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"

        self.encoder = TransformerEncoder(in_ch=in_ch, base_ch=base_ch, num_blocks=num_enblocks)
        self.decoder = TransformerDecoder(base_ch=base_ch, num_blocks=num_deblocks)
        self.illumination_guide_layer = IllGuideLayer(base_ch=base_ch, in_ch=guide_ch)
        print('Using encoder decoder net')
    
    def forward(self, x, guide_in):
        guide_fea = self.illumination_guide_layer(guide_in)
        encoder_feature, multi_guide_fea = self.encoder(x, guide_fea)
        x_out = self.decoder(encoder_feature, multi_guide_fea)
        
        return x_out, guide_fea

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
    
    def get_lg_feature(self, x, guide_in):
        guide_fea = self.illumination_guide_layer(guide_in)
        encoder_feature, multi_guide_fea, x_g, x_l = self.encoder.get_lg_fea(x, guide_fea)
        x_out = self.decoder(encoder_feature, multi_guide_fea)
        
        return encoder_feature, multi_guide_fea, x_g, x_l
        