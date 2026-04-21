import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models_crossvit import CrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed, positional_encoding_1d
from torchvision import transforms
import einops
import functools
from dinov2.hub.backbones import dinov2_vitb14, dinov2_vitb14_reg

from transformers import BertModel, BertTokenizer
from timm.models.vision_transformer import Block
import os


class QUANet(nn.Module):
    def __init__(self, fim_depth: int = 4,
                 fim_num_heads: int = 8,
                 mlp_ratio: float = 4.,
                 unfreeze_bert: str = '',
                 image_encoder: str = 'dinov2_vitb14',
                 unfreeze_dino: str = '',
                 norm_layer: nn.Module = nn.LayerNorm,
                 decoder_arch: str = 'adapter', **kwargs):
        """
        Initialize the QUANet model.
        Parameters:
        fim_depth (int): Depth of the Feature Interaction Module (FIM).
        fim_num_heads (int): Number of attention heads in the FIM.
        mlp_ratio (float): Ratio of MLP hidden dimension to input dimension.
        norm_layer (nn.Module): Normalization layer to use.
        unfreeze_bert (str): Layers of BERT to unfreeze.
        image_encoder (str): Image encoder to use ('dinov2_vitb14').
        unfreeze_dino (str): Layers of DINO to unfreeze.
        decoder_arch (str): Architecture of the density decoder ('adapter').
        """
        super().__init__()

        self.img_encoder = DINOVIT(unfreeze_para=unfreeze_dino)
        self.n_patches = 16 * 16
        self.ie_hidden_dim = 768
        self.ie_out_dim = 512

        self.preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.Normalize(
                                                  mean=(0.48145466, 0.4578275, 0.40821073),
                                                  std=(0.26862954, 0.26130258, 0.27577711)
                                              )
                                              ])

        UNCASED = './bert_ckpt/bert-base-uncased'
        self.text_encoder = BERTTextTransformer(UNCASED, unfreeze_bert=unfreeze_bert)

        self.patch_feat_proj = nn.Linear(self.ie_hidden_dim, self.ie_out_dim, bias=True)
        nn.init.xavier_normal_(self.patch_feat_proj.weight)
        n_token = self.n_patches
        # the PE for the patch embeddings \mathcal{E}_p
        self.patch_emb_pos_embed = nn.Parameter(torch.zeros(1, n_token, self.ie_out_dim),
                                                requires_grad=False)  # fixed sin-cos embedding
        decoder_pos_embed = positional_encoding_1d(self.ie_out_dim, n_token)
        self.patch_emb_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))

        # --------------------------------------------------------------------------
        # The Hierarchical patch-text interaction module

        self.decoder_ln_pre = norm_layer(self.ie_out_dim)
        self.use_fim = True
        self.fim_blocks = nn.ModuleList([
            CrossAttentionBlock(self.ie_out_dim, fim_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                                norm_layer=norm_layer, drop=0.1, drop_path=0.1)
            for _ in range(fim_depth)])
        self.decoder_norm = norm_layer(self.ie_out_dim)

        # --------------------------------------------------------------------------
        # Density decoder
        self.patch_size = int(224 // math.sqrt(self.n_patches))
        self.decoder_arch = decoder_arch
        self.density_decoder = DAC_Decoder(
            num_patches=self.n_patches,
            patch_size=self.patch_size,
            embed_dim=self.ie_out_dim,
            decoder_embed_dim=self.ie_out_dim,
            decoder_arch=self.decoder_arch
        )

        self.global_prompt = ['a photo of a number of {}.',
                              'a photo of a number of small {}.',
                              'a photo of a number of medium {}.',
                              'a photo of a number of large {}.',
                              'there are a photo of a number of {}.',
                              'there are a photo of a number of small {}.',
                              'there are a photo of a number of medium {}.',
                              'there are a photo of a number of large {}.',
                              'a number of {} in the scene.',
                              'a photo of a number of {} in the scene.',
                              'there are a number of {} in the scene.',
                              ]

    def forward_visual_encoder(self, x, text_embedding):
        """
        input: x: images, [B, 3, 384, 384]
        text_embedding: [B, 1, 512]
        """
        x = self.preprocess(x)
        _, cls_token, x = self.img_encoder(x, text_embedding)
        return cls_token, x

    def forward_decoder(self, img_feat_patches, embeddings, cls_token):
        extra_out = {}
        x_cls = cls_token
        extra_out['x_cls'] = x_cls
        extra_out['text_embedding'] = embeddings

        patch_feat = img_feat_patches[:, 1:, :]
        patch_embedding = self.patch_feat_proj(patch_feat)
        extra_out['patch_embedding'] = patch_embedding
        extra_out['patch_token'] = patch_embedding.mean(dim=(1)).squeeze(1)
        extra_out['mix'] = patch_embedding.mean(dim=(1)).squeeze(1) + x_cls
        x = patch_embedding
        x = x + self.patch_emb_pos_embed  # [B, 196, 512]

        # apply Transformer blocks (cross-attention)
        if self.use_fim:
            for blk in self.fim_blocks:
                x = blk(x, embeddings)
        else:  # add
            x = x + embeddings

        x = self.decoder_norm(x)  # mixed_fim [B,784,512]

        out_decoder = self.density_decoder(x)
        pred_density = out_decoder['final_x']
        extra_out['cnn_x'] = out_decoder.get('cnn_x')
        extra_out['vit_x'] = out_decoder.get('vit_x')

        return pred_density, extra_out

    def forward(self, imgs, text, return_extra: bool = False, coop_require_grad: bool = False):
        # get the text embedding
        semantic_text = [gp.format(s) for s in text for gp in self.global_prompt]
        with torch.no_grad():
            semantic_embedding = self.text_encoder(semantic_text, imgs.device).float()
        # fuse the semantic embedding
        semantic_embedding = semantic_embedding.reshape(len(text), len(self.global_prompt), -1)
        semantic_embedding = torch.mean(semantic_embedding, dim=1, keepdim=True)
        semantic_embedding = F.normalize(semantic_embedding, dim=-1)

        cls_token, img_feat_patches = self.forward_visual_encoder(imgs, semantic_embedding)
        pred_density, extra_out = self.forward_decoder(img_feat_patches, semantic_embedding,
                                                       cls_token)  # [N, 384, 384]

        if return_extra:
            return pred_density, extra_out
        return pred_density

    def seq_2_2d(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x


class BERTTextTransformer(nn.Module):
    """
    BERT Transformer encoder (text)
    """

    def __init__(self, ckpt_path, unfreeze_bert) -> None:
        super().__init__()
        self.model = BertModel.from_pretrained(ckpt_path)
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(ckpt_path, 'vocab.txt'))
        self.linnear = nn.Linear(768, 512)

        # 冻结bert参数
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_bert:
                if ele in name:
                    param.requires_grad = True

    def forward(self, text, device):
        """
        Input:
            text: tokenized text, shape = [batch_size, n_ctx]
        """
        text_token = self.tokenizer(text, return_tensors='pt', padding=True).to(device)
        outputs = self.model(**text_token)
        last_hidden_state = outputs.last_hidden_state
        cls = last_hidden_state[:, 0, :]  # [batch_size, 768]
        cls = self.linnear(cls)
        return cls


class DINOVIT(nn.Module):
    def __init__(self, unfreeze_para: str = '', n_last_blocks: int = 1) -> None:
        super().__init__()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = functools.partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float16)
        self.backbone = dinov2_vitb14_reg(pretrained=False)
        # Load local DINOv2 weights from dinov2_ckpt/
        _ckpt_path = os.path.join('dinov2_ckpt', 'dinov2_vitb14_reg4.pth')
        if os.path.isfile(_ckpt_path):
            _state = torch.load(_ckpt_path, map_location='cpu')
            self.backbone.load_state_dict(_state, strict=True)
        else:
            import warnings
            warnings.warn(
                f"DINOv2 checkpoint not found at '{_ckpt_path}'. "
                "The backbone will run with random weights. "
                "Download dinov2_vitb14_reg4.pth from the Meta DINOv2 release and place it under dinov2_ckpt/."
            )
        self.token_layer = nn.Linear(768, 512)
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_para:
                if ele in name:
                    param.requires_grad = True

    def forward(self, x, text_embedding):
        with self.autocast_ctx():
            features_dict = self.backbone.get_intermediate_layers(
                x, self.n_last_blocks, return_class_token=True
            )
        cls_token = features_dict[0][1]
        img_feat_patches = features_dict[0][0]
        xx = torch.cat([cls_token.unsqueeze(1), img_feat_patches], dim=1)
        cls_token = self.token_layer(cls_token)
        return img_feat_patches, cls_token, xx


class DAC_Decoder(nn.Module):
    """Dual-branch Adapter-CNN Decoder"""

    def __init__(self, num_patches=14 * 14, target_hw: int = 384, patch_size=16, in_chans=1,
                 embed_dim=512, decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, decoder_arch='adapter') -> None:
        super().__init__()

        self.arch = decoder_arch
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.target_hw = [target_hw, target_hw]

        # ViT branch
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.vit_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, decoder_embed_dim),
                                              requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size ** 2 * in_chans, bias=True)

        # CNN branch
        self.n_levels = 2
        convs = []
        inverselayer = []
        crt_dim = embed_dim
        for i in range(self.n_levels):
            decode_head = nn.Sequential(
                nn.Conv2d(crt_dim, crt_dim // 2, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, crt_dim // 2),
                nn.GELU()
            )
            decode_invlayer = nn.Sequential(nn.Linear(embed_dim, crt_dim // 2, bias=True),
                                            nn.LayerNorm(crt_dim // 2), nn.GELU())
            convs.append(decode_head)
            inverselayer.append(decode_invlayer)
            crt_dim = crt_dim // 2

        self.convs = nn.ModuleList(convs)
        self.invlayers = nn.ModuleList(inverselayer)

        self.final_conv = nn.Sequential(
            nn.Conv2d(crt_dim, 1, kernel_size=1, stride=1)
        )

        # FIM module for cross-branch interaction
        self.fim_blocks1 = nn.ModuleList([
            CrossAttentionBlock(int(embed_dim / pow(2, i + 1)), 8, mlp_ratio, qkv_bias=True, qk_scale=None,
                                norm_layer=norm_layer, drop=0.1, drop_path=0.1)
            for i in range(decoder_depth)])

        self.fim_blocks2 = nn.ModuleList([
            SE_Block(int(embed_dim / pow(2, i + 1)))
            for i in range(decoder_depth)])

        # MoE gating
        self.moe = nn.Sequential(nn.Linear(embed_dim, 2, bias=True), nn.LayerNorm(2), nn.GELU())

        # Initialize weights
        self.initialize_weights()
        for conv in self.convs:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def seq_2_2d(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x):
        out = dict()
        w = self.moe(torch.mean(x, dim=[1]))
        w = F.softmax(w, dim=1)

        # CNN branch
        cnn_x = self.seq_2_2d(x)
        # ViT branch
        vit_x = self.decoder_embed(x)  # [B, 196, 512]
        cls_token = self.vit_cls_token.expand(vit_x.size(0), -1, -1)
        vit_x = torch.cat((cls_token, vit_x), dim=1)
        vit_x = vit_x + self.decoder_pos_embed

        def inner_attention(self, cnn_x, vit_x, level):
            cnn_x = self.convs[level](cnn_x)  # (b,d,h,w)
            _, d, h, w = cnn_x.shape
            cnn_x = cnn_x.reshape(-1, h * w, d)  # (b,d,h,w) -> (b,h*w,d)
            vit_x = self.decoder_blocks[level](vit_x)  # (b,p2,d_model) -> (b,p2,d_model)

            vit_token_global = vit_x[:, :1, :]  # (b,1,d)
            vit_token3 = self.invlayers[level](vit_token_global)

            cnn_token1 = self.fim_blocks1[level](cnn_x, vit_token3)
            cnn_token2 = self.fim_blocks2[level](cnn_x, vit_token3)

            cnn_token_inv1 = nn.Identity()(cnn_token1)
            cnn_token_inv2 = nn.Identity()(cnn_token2)
            cnn_x = cnn_x + cnn_token_inv1 + cnn_token_inv2
            cnn_x = cnn_x.reshape(-1, d, h, w)
            return cnn_x, vit_x, cnn_token1 + cnn_token2

        cnn_x, vit_x, _ = inner_attention(self, cnn_x, vit_x, 0)
        cnn_x = F.interpolate(cnn_x, scale_factor=2, mode='bilinear', align_corners=False)

        cnn_x, vit_x, cnn_token = inner_attention(self, cnn_x, vit_x, 1)
        cnn_x = F.interpolate(cnn_x, size=self.target_hw, mode='bilinear', align_corners=False)
        out['cnn_token'] = cnn_token if cnn_token.shape[1] == 1 else cnn_token.mean(dim=[1])
        cnn_x = self.final_conv(cnn_x)
        cnn_x = F.sigmoid(cnn_x)
        cnn_x = einops.rearrange(cnn_x, 'n 1 h w -> n h w')

        vit_x = self.decoder_norm(vit_x)
        out['vit_token'] = vit_x[:, :1, :]
        vit_x = vit_x[:, 1:, :]  # remove cls token
        vit_x = self.decoder_pred(vit_x)  # [B,196,512] -> [B,196,16*16*1]

        vit_x = self.unpatchify(vit_x)  # [B,196,16*16*1] -> [B,1,224,224]

        vit_x = F.interpolate(vit_x, size=self.target_hw, mode='bilinear', align_corners=False)
        vit_x = vit_x.squeeze(1)

        w = w.view(w.shape[0], 2, 1, 1).expand(-1, -1, self.target_hw[0], self.target_hw[1])

        moe_x = w[:, 0:1].squeeze() * cnn_x + w[:, 1:2].squeeze() * vit_x

        out['vit_x'], out['cnn_x'], out['final_x'] = vit_x, cnn_x, moe_x

        return out


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool1d((1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
 
    def forward(self, x, y):
            # 读取批数据图片数量及通道数
            b, _, d = y.size()
            # Fsq操作：经池化后输出b*d的矩阵
            y = self.gap(y.permute(0,2,1)).view(b, 1, d)
            # Fex操作：经全连接层输出（b，d，1）矩阵
            y = self.fc(y)
            # Fscale操作：将得到的权重乘以原来的特征图x
            x_scale = x * y.expand_as(x)
            return x_scale

if __name__ == "__main__":
    quanet = QUANet()