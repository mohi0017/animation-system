"""
Shared model definitions for Stage 1 animation cleanup.
Used by both training and inference scripts.
"""

import torch
import torch.nn as nn


class PhaseEmbedder(nn.Module):
    """Phase conditioning embedder."""
    
    def __init__(self, labels, embed_dim=16):
        super().__init__()
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.embed = nn.Embedding(len(labels), embed_dim)

    def forward(self, input_phase, target_phase, B, H, W, device):
        """Generate phase conditioning embeddings."""
        inp_idx = torch.tensor([self.label_to_idx[p] for p in input_phase], device=device)
        tgt_idx = torch.tensor([self.label_to_idx[p] for p in target_phase], device=device)
        inp_e = self.embed(inp_idx)   # (B, E)
        tgt_e = self.embed(tgt_idx)   # (B, E)
        cond = torch.cat([inp_e, tgt_e], dim=1)  # (B, 2E)
        cond = cond.unsqueeze(-1).unsqueeze(-1).expand(B, cond.shape[1], H, W)
        return cond


class UNetBlock(nn.Module):
    """UNet encoder/decoder block."""
    
    def __init__(self, in_c, out_c, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
    
    def forward(self, x):
        return self.dropout(self.block(x))


class UNetGenerator(nn.Module):
    """UNet generator for image-to-image translation."""
    
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        # Encoder (downsampling)
        self.d1 = UNetBlock(in_ch, 64, down=True)      # 256
        self.d2 = UNetBlock(64, 128, down=True)          # 128
        self.d3 = UNetBlock(128, 256, down=True)        # 64
        self.d4 = UNetBlock(256, 512, down=True)        # 32
        self.d5 = UNetBlock(512, 512, down=True)        # 16
        self.d6 = UNetBlock(512, 512, down=True)        # 8
        self.d7 = UNetBlock(512, 512, down=True)        # 4

        # Decoder (upsampling)
        self.u1 = UNetBlock(512, 512, down=False, use_dropout=True)   # 8
        self.u2 = UNetBlock(1024, 512, down=False, use_dropout=True)  # 16
        self.u3 = UNetBlock(1024, 512, down=False, use_dropout=True) # 32
        self.u4 = UNetBlock(1024, 256, down=False)                   # 64
        self.u5 = UNetBlock(512, 128, down=False)                     # 128
        self.u6 = UNetBlock(256, 64, down=False)                      # 256
        self.u7 = nn.ConvTranspose2d(128, out_ch, 4, 2, 1)          # 512
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Forward pass through UNet."""
        # Encoder
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)

        # Decoder with skip connections
        u1 = self.u1(d7)
        u2 = self.u2(torch.cat([u1, d6], dim=1))
        u3 = self.u3(torch.cat([u2, d5], dim=1))
        u4 = self.u4(torch.cat([u3, d4], dim=1))
        u5 = self.u5(torch.cat([u4, d3], dim=1))
        u6 = self.u6(torch.cat([u5, d2], dim=1))
        u7 = self.u7(torch.cat([u6, d1], dim=1))
        
        return self.tanh(u7)  # RGB in [-1,1], Alpha in [0,1] (after normalization)

