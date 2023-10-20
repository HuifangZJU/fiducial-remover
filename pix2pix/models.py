import torch
import torch.nn as nn
from .utils import trunc_normal_
import math

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
##############################
#           U-NET
##############################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class UNetUpNoSkip(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUpNoSkip, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


#######################################
#           ViT
######################################
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



##############################
#           attention Generators
##############################
def divide_batch_into_patches(images, patch_size):
    # Input:
    #   - images: Input batch of images tensor of shape (B, C, H, W)
    #   - patch_size: Size of the patches
    # Output:
    #   - patches: Extracted patches tensor of shape (B, N, C, patch_height, patch_width)

    B, C, H, W = images.shape

    # Extract patches using unfold
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, -1, C, patch_size, patch_size)

    return patches

def reconstruct_batch_images(patches, image_size):
    # Input:
    #   - patches: Extracted patches tensor of shape (B, N, C, patch_height, patch_width)
    #   - batch_size: Batch size
    #   - image_size: Original size of the image (original_height, original_width)
    #   - patch_size: Size of the patches (patch_height, patch_width)
    #   - stride: Stride for patch extraction (stride_height, stride_width)
    # Output:
    #   - reconstructed_images: Reconstructed batch of images tensor of shape (B, C, original_height, original_width)

    B, N, C, patch_height, patch_width = patches.shape
    original_height, original_width = image_size

    # Calculate the number of patches in each dimension
    num_patches_h = original_height //patch_height
    num_patches_w = original_width // patch_width

    # Reshape the patches tensor to (B, num_patches_h, num_patches_w, C, patch_height, patch_width)
    patches = patches.view(B, num_patches_h, num_patches_w, C, patch_height, patch_width)

    # Transpose and reshape the patches tensor to (B, C, H, W)
    reconstructed_images = patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, original_height, original_width)

    return reconstructed_images


class Attention_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, patch_size=32):
        super(Attention_Generator, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, 256))
        self.blocks = nn.ModuleList([
            Block(
                dim=256, num_heads=4)
            for i in range(5)])
        self.norm = nn.LayerNorm(256)
        self.patch_size = patch_size

        # self.down1 = UNetDown(in_channels, 16, normalize=False)
        # self.down2 = UNetDown(16, 32, dropout=0.5)
        # self.down3 = UNetDown(32, 64, dropout=0.5)
        # self.down4 = UNetDown(64, 128, normalize=False, dropout=0.5)
        # self.down5 = UNetDown(128, 128, normalize=False, dropout=0.5)
        #
        # self.up1 = UNetUp(128, 128, dropout=0.5)
        # self.up2 = UNetUp(256, 64, dropout=0.5)
        # self.up3 = UNetUp(128, 32, dropout=0.5)
        # self.up4 = UNetUp(64, 16)

        # self.final = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(32, out_channels, 4, padding=1),
        #     nn.Tanh()
        # )

        self.down1 = UNetDown(in_channels, 16, normalize=False)
        self.down2 = UNetDown(16, 32, dropout=0.5)
        self.down3 = UNetDown(32, 64, dropout=0.5)
        self.down4 = UNetDown(64, 64, normalize=False, dropout=0.5)

        self.up1 = UNetUp(64, 64, dropout=0.5)
        self.up2 = UNetUp(128, 32, dropout=0.5)
        self.up3 = UNetUp(64, 16)

        self.head = nn.Linear(256, 1)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, out_channels, 4, padding=1),
            nn.Tanh()
        )

        # self.apply(self._init_weights)

    def interpolate_pos_encoding(self, patches, w, h):

        npatch = patches.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = patch_pos_embed.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return  patch_pos_embed


    def forward(self, x):

        B, nc, w, h = x.shape

        patches = divide_batch_into_patches(x, self.patch_size)
        pos_embed = self.interpolate_pos_encoding(patches, w, h)


        patches = patches.squeeze()
        d1 = self.down1(patches)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d4_flatten = d4.flatten(1).unsqueeze(0)

        d4_flatten_with_pe = d4_flatten + 5*pos_embed
        for blk in self.blocks:
            d4_flatten_with_pe = blk(d4_flatten_with_pe)
        d4_flatten_with_pe = self.norm(d4_flatten_with_pe)

        y = self.head(d4_flatten_with_pe)
        num_patches_w = w // self.patch_size
        num_patches_h = h // self.patch_size
        y = y.view(1,num_patches_w,num_patches_h)

        # d4_attention = d4_flatten_with_pe.squeeze().view(d4.shape)
        # u1 = self.up1(d4_attention, d3)
        # # u1 = self.up1(d4, d3)
        # u2 = self.up2(u1, d2)
        # u3 = self.up3(u2, d1)
        # final = self.final(u3)
        # final = final.unsqueeze(0)
        # mask = reconstruct_batch_images(final, (2048,2048))
        return y



class Dot_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, patch_size=4):
        super(Dot_Generator, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, 32))
        self.blocks = nn.ModuleList([
            Block(
                dim=32, num_heads=4)
            for _ in range(6)])
        self.norm = nn.LayerNorm(32)
        self.patch_size = patch_size

        self.down1 = UNetDown(in_channels, 4, normalize=False)
        self.down2 = UNetDown(4, 8, dropout=0.5)
        self.down3 = UNetDown(8, 16, dropout=0.5)
        self.down4 = UNetDown(16, 32, normalize=False, dropout=0.5)
        self.down5 = UNetDown(32, 32, normalize=False, dropout=0.5)

        #self.up1 = UNetUp(32, 32, dropout=0.5)
        #self.up2 = UNetUp(64, 16, dropout=0.5)
        #self.up3 = UNetUp(32, 8, dropout=0.5)
        #self.up4 = UNetUp(16, 4)
        self.up1 = UNetUpNoSkip(32, 32, dropout=0.5)
        self.up2 = UNetUpNoSkip(32, 16, dropout=0.5)
        self.up3 = UNetUpNoSkip(16, 8, dropout=0.5)
        self.up4 = UNetUpNoSkip(8, 4)


        # self.down1 = UNetDown(in_channels, 16, normalize=False)
        # self.down2 = UNetDown(16, 32, dropout=0.5)
        # self.down3 = UNetDown(32, 64, dropout=0.5)
        # self.down4 = UNetDown(64, 64, normalize=False, dropout=0.5)
        #
        # self.up1 = UNetUp(64, 64, dropout=0.5)
        # self.up2 = UNetUp(128, 32, dropout=0.5)
        # self.up3 = UNetUp(64, 16)

        self.head = nn.Linear(32, 1)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(4, out_channels, 4, padding=1),
            nn.Sigmoid()
            #nn.Tanh()
        )

        # self.apply(self._init_weights)

    def interpolate_pos_encoding(self, patches, w, h):

        npatch = patches.shape[-2]*patches.shape[-1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed

        patch_pos_embed = self.pos_embed
        dim = patch_pos_embed.shape[-1]
        w0 = patches.shape[-2]
        h0 = patches.shape[-1]
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0/math.sqrt(N), h0 /math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return  patch_pos_embed


    def forward(self, x):
        B, nc, w, h = x.shape
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)


        pos_embed = self.interpolate_pos_encoding(d5,w,h)
        d5_flatten = d5.flatten(2).transpose(1,2)
        d5_flatten_with_pe = d5_flatten + 5*pos_embed
        for blk in self.blocks:
            d5_flatten_with_pe = blk(d5_flatten_with_pe)
        d5_flatten_with_pe = self.norm(d5_flatten_with_pe)

        d5_attention = d5_flatten_with_pe.transpose(1,2).squeeze().view(d5.shape)
        #u1 = self.up1(d5_attention, d4)
        #u2 = self.up2(u1, d3)
        #u3 = self.up3(u2, d2)
        #u4 = self.up4(u3, d1)
        u1 = self.up1(d5_attention)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)
        final = self.final(u4)
        return final

##############################

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Generator, self).__init__()
        self.down1 = UNetDown(in_channels, 16, normalize=False)
        self.down2 = UNetDown(16, 32, dropout=0.5)
        self.down3 = UNetDown(32, 64, dropout=0.5)
        self.down4 = UNetDown(64, 64, normalize=False, dropout=0.5)

        self.up1 = UNetUp(64, 64, dropout=0.5)
        self.up2 = UNetUp(128, 32, dropout=0.5)
        self.up3 = UNetUp(64, 16)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        return self.final(u3)


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 4, normalization=False),
            *discriminator_block(4, 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(8, 1, 4, padding=1, bias=False)
        )

    # def forward(self, img_A, img_B):
    #     # Concatenate image and condition image by channels to produce input
    #     img_input = torch.cat((img_A, img_B), 1)
    #     return self.model(img_input)

    def forward(self, img_B):

        return self.model(img_B)



