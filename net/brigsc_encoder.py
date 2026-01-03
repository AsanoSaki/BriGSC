import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict

from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import trunc_normal_, DropPath

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper


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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def save_importance_vector(self, importance_vector):
        self.importance_vector = importance_vector

    def get_importance_vector(self):
        return self.importance_vector

    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)
        imp = attn[:, :, 1:, 1:].clone().detach()  # remove cls token
        imp = torch.softmax(torch.sum(imp, dim=2), dim=2)  # sum over tokens
        imp = torch.mean(imp, dim=1)  # mean over heads
        self.save_importance_vector(imp)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=True):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)


class BriGSC_Encoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dims=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 use_grad_checkpointing=False, ckpt_layer=0, comp_size=2048, model="BriGSC"):
        super().__init__()
        self.num_features = self.embed_dims = embed_dims  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.comp_size = comp_size
        self.model = model
        self.layer_num = 7
        self.hidden_dims = int(self.embed_dims * 1.5)

        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims, self.hidden_dims))
        for i in range(self.layer_num):
            if i == self.layer_num - 1:
                outdim = self.embed_dims
            else:
                outdim = self.hidden_dims
            self.bm_list.append(AdaptiveModulator(self.hidden_dims))
            self.sm_list.append(nn.Linear(self.hidden_dims, outdim))
        self.sigmoid = nn.Sigmoid()

        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC)])

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i >= depth - ckpt_layer)
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dims)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, snr, compress=False, register_blk=-1):
        x = self.transform(x)

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)
        x = self.norm(x)

        vit_embeddings = x.clone()
        # vit_embeddings = x

        x = x[:, 1:, :]

        # compress feature according to saliency map
        if compress:
            x = self.compress_feature(x)

        if self.model == 'BriGSC':
            snr_cuda = torch.tensor(snr, dtype=torch.float).to(x.get_device())
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)

                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, 256, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val

        # return x
        return x, vit_embeddings

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    # 根据saliency_map对WITT编码后的特征进行压缩
    def compress_feature(self, x):
        saliency = [self.blocks[x].attn.get_importance_vector() for x in range(len(self.blocks))]
        saliency = torch.mean(torch.stack(saliency, dim=0), dim=0)

        saliency = saliency.view(saliency.shape[0], 16, 16)
        saliency = saliency / torch.sum(saliency, dim=(1, 2), keepdim=True)
        saliency = saliency.view(saliency.shape[0], 16 * 16)

        feature_size = x.shape[-1]
        assert x.shape[:2] == saliency.shape, "x[:2] and saliency_map must have the same shape, but got {} and {}".format(
            x.shape[:2], saliency.shape)
        sm = torch.round(self.comp_size * saliency).clip(1, self.comp_size - 1).long()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tmp = F.interpolate(x[i, j].unsqueeze(0).unsqueeze(0), size=int(sm[i, j].item()), mode='linear',
                                    align_corners=False).squeeze(0).squeeze(0)
                x[i, j] = F.interpolate(tmp.unsqueeze(0).unsqueeze(0), size=feature_size, mode='linear',
                                        align_corners=False).squeeze(0).squeeze(0)

        return x


@torch.no_grad()
def _load_weights(model: BriGSC_Encoder, checkpoint_path: str, prefix: str = ''):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model)
    new_state_dict = OrderedDict()

    for ckpt_key in list(state_dict.keys()):
        if not ckpt_key.startswith("visual_encoder") or not ckpt_key[15:] in list(model.state_dict().keys()):
            del state_dict[ckpt_key]
        else:
            new_state_dict[ckpt_key[15:]] = state_dict[ckpt_key]

    msg = model.load_state_dict(new_state_dict, strict=False)


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint


def create_BriGSCEncoder(**kwargs):
    model = BriGSC_Encoder(**kwargs)
    return model
