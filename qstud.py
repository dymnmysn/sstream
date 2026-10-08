# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
import os
#try:
import pytorch_lightning as pl
"""except ImportError:
    class pl:
        class LightningModule:
            pass

        class Callback:
            pass"""
from itertools import chain
import random
from pytorch_lightning.callbacks import ModelCheckpoint

DEBUG = False




class Embedding(nn.Module):
    def __init__(self, num_embeddings=2048, embedding_dim=2048,
                 start_t=1, end_t=10, total_epochs=1200, eps=1e-8):
        super().__init__()
        assert num_embeddings % 4 == 0, "num_embeddings must be divisible by 4"
        quarter = num_embeddings // 4

        # Separate embedding tables for each quarter
        self.sub_dim = embedding_dim // 4
        self.embeddings    = nn.Parameter(torch.randn(quarter, self.sub_dim))
        self.embeddings_r1 = nn.Parameter(torch.randn(quarter, self.sub_dim))
        self.embeddings_r2 = nn.Parameter(torch.randn(quarter, self.sub_dim))
        self.embeddings_r3 = nn.Parameter(torch.randn(quarter, self.sub_dim))

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.quarter = quarter

        # annealing parameters
        self.start_t = start_t
        self.end_t = end_t
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.t = end_t

        # for entropy calculation
        self.eps = eps

        # track usage frequency of embeddings
        self.register_buffer("usage_counts", torch.zeros(num_embeddings, dtype=torch.long))
        self.soft_weight = 0.00001 #0.015 
        self.justmax = False

    def update_epoch(self, epoch: int):
        """Call at start of each epoch"""
        #self.current_epoch = min(epoch, self.total_epochs)
        self.current_epoch = epoch
        cos_factor = 0.5 * (1 + torch.cos(
            torch.tensor(self.current_epoch / self.total_epochs * torch.pi,
                         device=self.embeddings.device)))
        t = self.end_t + (self.start_t - self.end_t) * cos_factor.item()
        self.t = 10.0 if not self.training else t

    def decode(self, codes: torch.Tensor):
        """
        codes: (B, 4, L) with values in [0, num_embeddings)
        """
        q = self.quarter
        codes_0 = codes[:, 0, :].long()
        codes_1 = codes[:, 1, :].long() - q
        codes_2 = codes[:, 2, :].long() - 2*q
        codes_3 = codes[:, 3, :].long() - 3*q

        decoded_0 = F.embedding(codes_0, self.embeddings)
        decoded_1 = F.embedding(codes_1, self.embeddings_r1)
        decoded_2 = F.embedding(codes_2, self.embeddings_r2)
        decoded_3 = F.embedding(codes_3, self.embeddings_r3)

        # permute for channel-first format
        decoded_0 = decoded_0.permute(0, 2, 1).contiguous()
        decoded_1 = decoded_1.permute(0, 2, 1).contiguous()
        decoded_2 = decoded_2.permute(0, 2, 1).contiguous()
        decoded_3 = decoded_3.permute(0, 2, 1).contiguous()

        if self.training:
            k = torch.randint(low=0, high=4, size=(1,), device=decoded_0.device).item()
            zeros = torch.zeros_like(decoded_0)
            if k == 0:
                decoded = torch.cat([decoded_0, zeros, zeros, zeros], dim=1)
            elif k == 1:
                decoded = torch.cat([decoded_0, decoded_1, zeros, zeros], dim=1)
            elif k == 2:
                decoded = torch.cat([decoded_0, decoded_1, decoded_2, zeros], dim=1)
            else:
                decoded = torch.cat([decoded_0, decoded_1, decoded_2, decoded_3], dim=1)
        else:
            # During eval, always use all terms
            decoded = torch.cat([decoded_0, decoded_1, decoded_2, decoded_3], dim=1)

        return decoded

    def process_block(self, block_logits, embeddings, offset=0):
        # Softmax attention weights
        attn_weights = F.softmax(block_logits, dim=1)
        # Weighted sum
        output = torch.einsum("bnl,nm->bml", attn_weights, embeddings)
        # Codes for this block (add offset so indices map into global embedding space)
        codes = torch.argmax(attn_weights, dim=1).detach() + offset
        return attn_weights, output, codes
    
    
    def process_block_max(self, block_logits, embeddings, offset=0, k=3):
        # Get top-k indices
        topk_vals, topk_idx = torch.topk(block_logits, k, dim=1)  # (B, k, L)

        # Create a mask for top-k positions
        mask = torch.zeros_like(block_logits).scatter_(1, topk_idx, 1.0)

        # Zero out other logits by setting them to -inf
        masked_logits = block_logits.masked_fill(mask == 0, float('-inf'))

        # Softmax over the restricted logits (only top-k survive)
        attn_weights = F.softmax(masked_logits, dim=1)

        # Weighted sum (now using only top-k distribution)
        output = torch.einsum("bnl,nm->bml", attn_weights, embeddings)

        # Codes for this block (take argmax among top-k)
        codes = torch.argmax(attn_weights, dim=1).detach() + offset

        return attn_weights, output, codes


    def forward(self, logits: torch.Tensor):
        """
        logits: (B, num_embeddings, seq_len)
        output: (B, embedding_dim, seq_len)
        """
        B, N, L = logits.shape
        q = self.quarter

        # Split logits into four parts
        logits_0 = logits[:, 0*q:1*q, :]
        logits_1 = logits[:, 1*q:2*q, :]
        logits_2 = logits[:, 2*q:3*q, :]
        logits_3 = logits[:, 3*q:4*q, :]


        if self.training and self.current_epoch<700:
            attn_0, out_0, codes_0 = self.process_block(logits_0, self.embeddings, offset=0*q)
            attn_1, out_1, codes_1 = self.process_block(logits_1, self.embeddings_r1, offset=1*q)
            attn_2, out_2, codes_2 = self.process_block(logits_2, self.embeddings_r2, offset=2*q)
            attn_3, out_3, codes_3 = self.process_block(logits_3, self.embeddings_r3, offset=3*q)
        elif self.training and self.current_epoch<1000:
            attn_0, out_0, codes_0 = self.process_block_max(logits_0, self.embeddings, offset=0*q, k=3)
            attn_1, out_1, codes_1 = self.process_block_max(logits_1, self.embeddings_r1, offset=1*q, k=3)
            attn_2, out_2, codes_2 = self.process_block_max(logits_2, self.embeddings_r2, offset=2*q, k=3)
            attn_3, out_3, codes_3 = self.process_block_max(logits_3, self.embeddings_r3, offset=3*q, k=3)
        else:
            attn_0, out_0, codes_0 = self.process_block_max(logits_0, self.embeddings, offset=0*q, k=1)
            attn_1, out_1, codes_1 = self.process_block_max(logits_1, self.embeddings_r1, offset=1*q, k=1)
            attn_2, out_2, codes_2 = self.process_block_max(logits_2, self.embeddings_r2, offset=2*q, k=1)
            attn_3, out_3, codes_3 = self.process_block_max(logits_3, self.embeddings_r3, offset=3*q, k=1)

        """
        elif self.training and self.current_epoch<800:
            attn_0, out_0, codes_0 = self.process_block_max(logits_0, self.embeddings, offset=0*q, k=3)
            attn_1, out_1, codes_1 = self.process_block_max(logits_1, self.embeddings_r1, offset=1*q, k=3)
            attn_2, out_2, codes_2 = self.process_block_max(logits_2, self.embeddings_r2, offset=2*q, k=3)
            attn_3, out_3, codes_3 = self.process_block_max(logits_3, self.embeddings_r3, offset=3*q, k=3)
        """

        # Combine outputs
        output = torch.cat([out_0, out_1, out_2, out_3], dim=1)

        # Save last attention/codes
        self.last_attn  = torch.cat([attn_0, attn_1, attn_2, attn_3], dim=1)
        self.last_codes = torch.stack([codes_0, codes_1, codes_2, codes_3], dim=1)

        # Update usage counts
        flat_codes = self.last_codes.reshape(-1)
        if flat_codes.numel() > 0:
            binc = torch.bincount(flat_codes, minlength=self.num_embeddings)
            self.usage_counts += binc.to(self.usage_counts.device, dtype=self.usage_counts.dtype)

        return output

    @torch.no_grad()
    def redistribute_embeddings(self, reset: bool = True, strength: float = 0.05, noise_scale: float = 1e-3):
        if self.usage_counts.sum() == 0:
            if reset:
                self.usage_counts.zero_()
            return

        # Example: only redistribute first quarter embeddings (can be extended similarly)
        q = self.quarter
        mf_idx = int(torch.argmax(self.usage_counts).item())
        lf_idx = int(torch.argmin(self.usage_counts).item())

        mf = self.embeddings.data[mf_idx % q].clone()
        lf = self.embeddings.data[lf_idx % q].clone()
        self.embeddings.data[lf_idx % q] = mf + strength * lf

        unused = (self.usage_counts == 0).nonzero(as_tuple=True)[0]
        used_counts = self.usage_counts.clone().float()

        if unused.numel() > 0 and used_counts.sum() > 0:
            probs = used_counts / used_counts.sum()
            sampled_indices = torch.multinomial(probs, num_samples=unused.numel(), replacement=True)
            for u, idx in zip(unused, sampled_indices):
                base = self.embeddings.data[idx % q].clone()
                noise = torch.randn(self.sub_dim, device=self.embeddings.device) * noise_scale
                self.embeddings.data[u % q] = base + noise

        if reset:
            self.usage_counts.zero_()

    @torch.no_grad()
    def reset_usage_counts(self):
        self.usage_counts.zero_()

    @torch.no_grad()
    def calc_entropy(self):
        if not hasattr(self, "last_codes"):
            return 0.0, 0.0
        eps = self.eps
        codes = self.last_codes.reshape(-1)
        one_hot = F.one_hot(codes, num_classes=self.num_embeddings).float()
        avg_counts = one_hot.mean(dim=0)
        avg_counts = avg_counts / (avg_counts.sum() + eps)
        raw_entropy = -(avg_counts * torch.log(avg_counts + eps)).sum()
        max_entropy = torch.log(torch.tensor(self.num_embeddings, device=raw_entropy.device, dtype=raw_entropy.dtype))
        normalized_entropy = raw_entropy / max_entropy
        return normalized_entropy

    def calc_entropy_loss(self, p=0.2) -> torch.Tensor:
        if not hasattr(self, "last_attn"):
            return torch.zeros((), device=next(self.parameters()).device)
        eps = self.eps
        avg_weight, soft_weight = (0.9, 0.1) if random.random() < p else (0.1, 0.9)
        #soft_weight = 0.015

        attn_weights = self.last_attn
        attn_weights_1,attn_weights_2,attn_weights_3,attn_weights_4 = attn_weights[:,:self.quarter,:], attn_weights[:,self.quarter:2*self.quarter,:],   attn_weights[:,2*self.quarter:3*self.quarter,:],attn_weights[:,3*self.quarter:,:]                                                
        attn_weights_1 = attn_weights_1 / (attn_weights_1.sum(dim=1, keepdim=True) + eps)
        attn_weights_2 = attn_weights_2 / (attn_weights_2.sum(dim=1, keepdim=True) + eps)
        attn_weights_3 = attn_weights_3 / (attn_weights_3.sum(dim=1, keepdim=True) + eps)
        attn_weights_4 = attn_weights_4 / (attn_weights_4.sum(dim=1, keepdim=True) + eps)

        soft_entropy_1 = -(attn_weights_1 * torch.log(attn_weights_1 + eps)).sum(dim=1)
        soft_entropy_2 = -(attn_weights_2 * torch.log(attn_weights_2 + eps)).sum(dim=1)
        soft_entropy_3 = -(attn_weights_3 * torch.log(attn_weights_3 + eps)).sum(dim=1)
        soft_entropy_4 = -(attn_weights_4 * torch.log(attn_weights_4 + eps)).sum(dim=1)

        soft_entropy = soft_entropy_1+soft_entropy_2+soft_entropy_3+soft_entropy_4
        soft_entropy = soft_entropy.mean()
        soft_loss = self.soft_weight * soft_entropy

        avg_dist = attn_weights.mean(dim=(0, 2))
        avg_dist = avg_dist / (avg_dist.sum() + eps)
        avg_entropy = -(avg_dist * torch.log(avg_dist + eps)).sum()
        max_entropy = torch.log(torch.tensor(self.num_embeddings, device=avg_entropy.device, dtype=avg_entropy.dtype))
        avg_loss = avg_weight * (max_entropy - avg_entropy)
        return soft_loss,soft_entropy #+ avg_loss


class ResNet1d(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size: int = 7,
        padding: str = 'valid',
        dilation: int = 1
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self._padding_size = (kernel_size // 2) * dilation
        self.conv0 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation)
        self.conv1 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=1)

    def forward(self, input):
        y = input
        x = self.conv0(input)
        x = F.elu(x)
        x = self.conv1(x)
        if self.padding == 'valid':
            y = y[:, :, self._padding_size:-self._padding_size]
        x += y
        x = F.elu(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResNet2d(nn.Module):
    def __init__(
        self,
        n_channels: int,
        factor: int,
        stride: Tuple[int, int]
    ) -> None:
        super().__init__()
        
        # FIX 1: Use explicit padding=1 instead of padding='same'.
        # For a 3x3 kernel with stride 1, padding=1 is mathematically identical 
        # to 'same', but perfectly safe for ONNX export and older PyTorch versions.
        self.conv0 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            padding=1)
        
        self.bn0 = nn.BatchNorm2d(n_channels)
        
        self.conv1 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride)
        
        self.bn1 = nn.BatchNorm2d(factor * n_channels)
        
        self.conv2 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=1,
            stride=stride)
        
        self.bn2 = nn.BatchNorm2d(factor * n_channels)
        
        # FIX 2: Store padding values instead of using nn.ReflectionPad2d module.
        # We will apply this dynamically in the forward pass.
        self.pad_vals = (
            (stride[1] + 1) // 2, # Left
            (stride[1] + 2) // 2, # Right
            (stride[0] + 1) // 2, # Top
            (stride[0] + 2) // 2, # Bottom
        )
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.activation(x)

        # FIX 2 (Continued): Apply safe padding
        # We check if the feature map is large enough to support reflection padding.
        # If the spatial dimensions are too small, we safely fall back to zero-padding.
        _, _, h, w = x.size()
        pad_l, pad_r, pad_t, pad_b = self.pad_vals
        
        if h > max(pad_t, pad_b) and w > max(pad_l, pad_r):
            x = F.pad(x, self.pad_vals, mode='reflect')
        else:
            x = F.pad(x, self.pad_vals, mode='constant', value=0.0)

        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut
        y = self.conv2(input)
        y = self.bn2(y)

        # FIX 3: Failsafe skip-connection shape alignment.
        # Mathematically, x and y should always match perfectly. However, 
        # odd-sized inputs combined with specific strides can occasionally 
        # produce a 1-pixel mismatch. This ensures they always add safely.
        if x.shape != y.shape:
            min_h = min(x.size(2), y.size(2))
            min_w = min(x.size(3), y.size(3))
            x = x[:, :, :min_h, :min_w]
            y = y[:, :, :min_h, :min_w]

        x = x + y
        x = self.activation(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            nn.Conv1d(
                n_channels // 2, n_channels,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                n_channels, n_channels // 2,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Encoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            EncoderBlock(2 * n_channels, padding=padding, stride=2),
            EncoderBlock(4 * n_channels, padding=padding, stride=4),
            EncoderBlock(8 * n_channels, padding=padding, stride=5),
            EncoderBlock(16 * n_channels, padding=padding, stride=8),
            nn.Conv1d(16 * n_channels, 64 * n_channels, kernel_size=3, padding=padding),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Decoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            DecoderBlock(16 * n_channels, padding=padding, stride=8),
            DecoderBlock(8 * n_channels, padding=padding, stride=5),
            DecoderBlock(4 * n_channels, padding=padding, stride=4),
            DecoderBlock(2 * n_channels, padding=padding, stride=2),
            nn.Conv1d(n_channels, 1, kernel_size=7, padding=padding),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)



class WaveDiscriminator(nn.Module):
    r"""MelGAN discriminator from https://arxiv.org/pdf/1910.06711.pdf
    """
    def __init__(self, resolution: int = 1, n_channels: int = 4) -> None:
        super().__init__()
        assert resolution >= 1
        if resolution == 1:
            self.avg_pool = nn.Identity()
        else:
            self.avg_pool = nn.AvgPool1d(resolution * 2, stride=resolution)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.layers = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, n_channels, kernel_size=15, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(n_channels, 4 * n_channels, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.utils.weight_norm(nn.Conv1d(4 * n_channels, 16 * n_channels, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(16 * n_channels, 64 * n_channels, kernel_size=41, stride=4, padding=20, groups=64)),
            nn.utils.weight_norm(nn.Conv1d(64 * n_channels, 256 * n_channels, kernel_size=41, stride=4, padding=20, groups=256)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 256 * n_channels, kernel_size=5, padding=2)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 1, kernel_size=3, padding=1)),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.avg_pool(x)
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
            x = self.activation(x)
        feats.append(self.layers[-1](x))
        return feats


class STFTDiscriminator(nn.Module):
    r"""STFT-based discriminator from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(
        self, n_fft: int = 512, hop_length: int = 256,
        n_channels: int = 32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n = n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=7, padding='same'),
            nn.LeakyReLU(0.3, inplace=True),
            ResNet2d(n_channels, 2, stride=(2, 1)),
            ResNet2d(2 * n_channels, 2, stride=(2, 2)),
            ResNet2d(4 * n_channels, 1, stride=(2, 1)),
            ResNet2d(4 * n_channels, 2, stride=(2, 2)),
            ResNet2d(8 * n_channels, 1, stride=(2, 1)),
            ResNet2d(8 * n_channels, 2, stride=(2, 2)),
            nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] == 1
        # input: [batch, channel, sequence]
        x = torch.squeeze(input, 1).to(torch.float32)  # torch.stft() doesn't accept float16
        x = torch.stft(x, self.n_fft, self.hop_length, normalized=True, center=False, onesided=True, return_complex=True)
        x = torch.abs(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.layers(x)
        return x


class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 10):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, center = False, win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, center = False, win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss / (10 - 6)


class ReconstructionLoss2(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, sample_rate, eps=1e-5):
        super().__init__()
        import torchaudio
        self.layers = nn.ModuleList()
        self.alpha = []
        self.eps = eps
        for i in range(6, 10):
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=int(2 ** i),
                win_length=int(2 ** i),
                hop_length=int(2 ** i / 4),
                n_mels=64)
            self.layers.append(melspec)
            self.alpha.append((2 ** i / 2) ** 0.5)

    def forward(self, input, target):
        loss = 0
        for alpha, melspec in zip(self.alpha, self.layers):
            x = melspec(input)
            y = melspec(target)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss


class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        channels: int = 32
    ) -> None:
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.weight_norm(nn.Conv2d(channels, channels * 2, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.weight_norm(nn.Conv2d(channels * 2, channels * 4, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            nn.utils.weight_norm(nn.Conv2d(channels * 4, channels * 8, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(channels * 8, 1, (3, 1), 1, padding=(1, 0)))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, 1, T]
        feats = []
        b, c, t = x.shape
        
        # Dynamic padding along time dimension so it divides evenly by period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect" if t > n_pad else "constant")
            t = x.shape[-1]
            
        x_2d = x.view(b, c, t // self.period, self.period)
        
        # Ensure temporal height is sufficient for kernel downsampling
        if x_2d.shape[2] < 5:
            x_2d = F.pad(x_2d, (0, 0, 0, 5 - x_2d.shape[2]), mode="constant")

        for layer in self.convs:
            x_2d = layer(x_2d)
            feats.append(x_2d)
            x_2d = self.activation(x_2d)

        out = self.conv_post(x_2d)
        feats.append(out)
        return feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        return [disc(x) for disc in self.discriminators]


class ComplexSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        n_channels: int = 32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        n = n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1
        self.final_freq_dim = n

        # 2 input channels: Real and Imaginary parts (carries phase + magnitude)
        self.conv_in = nn.Conv2d(2, n_channels, kernel_size=7, padding='same')
        self.activation = nn.LeakyReLU(0.3, inplace=True)

        self.res_blocks = nn.ModuleList([
            ResNet2d(n_channels, 2, stride=(2, 1)),
            ResNet2d(2 * n_channels, 2, stride=(2, 2)),
            ResNet2d(4 * n_channels, 1, stride=(2, 1)),
            ResNet2d(4 * n_channels, 2, stride=(2, 2)),
            ResNet2d(8 * n_channels, 1, stride=(2, 1)),
            ResNet2d(8 * n_channels, 2, stride=(2, 2)),
        ])
        self.conv_post = nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        assert input.shape[1] == 1
        x = torch.squeeze(input, 1).to(torch.float32)

        # Pad to avoid STFT failure on short segments
        if x.size(-1) < self.n_fft:
            pad_amt = self.n_fft - x.size(-1)
            x = F.pad(x, (0, pad_amt), mode='constant')

        stft = torch.stft(
            x, self.n_fft, self.hop_length,
            normalized=True, center=False, onesided=True, return_complex=True
        )

        # Shape: [B, 2, Freq, Time]
        feat = torch.cat([stft.real.unsqueeze(1), stft.imag.unsqueeze(1)], dim=1)

        feats = []
        feat = self.conv_in(feat)
        feats.append(feat)
        feat = self.activation(feat)

        for block in self.res_blocks:
            feat = block(feat)
            feats.append(feat)

        # Align frequency dimension safely before final projection
        if feat.size(2) > self.final_freq_dim:
            feat = feat[:, :, :self.final_freq_dim, :]
        elif feat.size(2) < self.final_freq_dim:
            feat = F.pad(feat, (0, 0, 0, self.final_freq_dim - feat.size(2)))

        out = self.conv_post(feat)
        feats.append(out)
        return feats


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        filters: Tuple[int, ...] = (256, 512, 1024),
        n_channels: int = 32
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([
            ComplexSTFTDiscriminator(n_fft=s, hop_length=s // 4, n_channels=n_channels)
            for s in filters
        ])

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        return [disc(x) for disc in self.discriminators]



class AudioTransformerDiscriminator(nn.Module):
    def __init__(
        self, 
        n_fft: int = 1024, 
        hop_length: int = 256, 
        d_model: int = 256, 
        n_heads: int = 4, 
        num_layers: int = 4
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_bins = n_fft // 2 + 1
        
        # Project raw frequency bins into the Transformer's hidden dimension
        self.input_proj = nn.Linear(n_bins, d_model)
        
        # Learnable positional embeddings (sized for segments up to ~32 seconds at 16kHz)
        self.pos_emb = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)
        
        # Stack layers in a ModuleList to easily extract intermediate features
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_heads, 
                dim_feedforward=d_model * 4,
                dropout=0.1, 
                activation="gelu", 
                batch_first=True, 
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        # input: [B, 1, T]
        x = torch.squeeze(input, 1).to(torch.float32)
        
        # Pad safely if the segment is unusually short
        if x.size(-1) < self.n_fft:
            x = F.pad(x, (0, self.n_fft - x.size(-1)))
            
        stft = torch.stft(
            x, self.n_fft, self.hop_length, 
            normalized=True, center=True, return_complex=True
        )
        mag = torch.abs(stft) # [B, Freq, Time]
        
        # Reshape to [B, Time, Freq] to treat time steps as sequence tokens
        x = mag.transpose(1, 2)
        x = self.input_proj(x)
        
        # Inject positional encoding
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
            
        x = self.final_norm(x)
        
        # Global average pooling across time
        x_pool = x.mean(dim=1)
        out = self.head(x_pool) # [B, 1]
        
        # Expand to [B, 1, 1] so its shape matches the CNN discriminators for the hinge loss
        out = out.unsqueeze(-1)
        feats.append(out)
        
        return feats




class StreamableModel(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 32,
        num_embeddings: int = 512,
        padding: str = "valid",
        batch_size: int = 32,
        sample_rate: int = 24_000,
        segment_length: int = 8000,
        window_shift: int = 300,
        include_tail: bool = True,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'librispeech',
        version: str = 'quadsingletrain',
        checkpoint: str = ''
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.encoder = Encoder(n_channels, padding)
        self.decoder = Decoder(n_channels, padding)

        # --- Replaced Discriminators ---
        self.mpd = MultiPeriodDiscriminator(periods=(2, 3, 5, 7, 11))
        self.ms_stft = MultiScaleSTFTDiscriminator(filters=(256, 512, 1024), n_channels=32)
        self.transformer_disc = AudioTransformerDiscriminator(d_model=256, num_layers=4)

        self.rec_loss = ReconstructionLoss()
        self.embed = Embedding(num_embeddings=512*4, embedding_dim=512, 
                               start_t=10.0, end_t=0.0, total_epochs=100, eps=1e-8)

        if checkpoint:
            self.load_state_dict(torch.load(checkpoint, self.device)['state_dict'])

        self.embed.soft_weight = 0.00001
        self._encoder_frozen = False

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.AdamW([
            {"params": self.encoder.parameters(), "weight_decay": 1e-5},
            {"params": self.decoder.parameters(), "weight_decay": 1e-5},
            {"params": self.embed.parameters(), "weight_decay": 0.0},
        ], lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.AdamW(
            [
                {"params": self.mpd.parameters(), "weight_decay": 0.0},
                {"params": self.ms_stft.parameters(), "weight_decay": 0.0},
                {"params": self.transformer_disc.parameters(), "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(b1, b2)
        )
        return [optimizer_g, optimizer_d], []

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        input = batch[:, None, :]  # [B, 1, T]

        all_discs = list(self.mpd.discriminators) + list(self.ms_stft.discriminators)

        # ----------------------------------------------------
        # 1. Forward Real Audio Once under no_grad for G step
        # ----------------------------------------------------
        # Extract real feature maps without tracking gradients
        with torch.no_grad():
            real_feats_list = [disc(input) for disc in all_discs]

        # ----------------------------------------------------
        # 2. Generator Step
        # ----------------------------------------------------
        self.toggle_optimizer(optimizer_g)
        output = self.forward(input)

        g_adv_loss = 0.0
        g_feat_loss = 0.0

        for disc, feats_real in zip(all_discs, real_feats_list):
            feats_fake = disc(output)

            # Adversarial hinge loss
            g_adv_loss += torch.mean(torch.relu(1.0 - feats_fake[-1]))

            # Feature matching against cached real features
            feat_dist = 0.0
            for f_real, f_fake in zip(feats_real[:-1], feats_fake[:-1]):
                feat_dist += torch.mean(torch.abs(f_real - f_fake))
            g_feat_loss += feat_dist / max(1, len(feats_real) - 1)

        g_adv_loss = g_adv_loss / len(all_discs)
        g_feat_loss = g_feat_loss / len(all_discs)

        g_rec_loss = self.rec_loss(output[:, 0, :], input[:, 0, :])
        codes_entropy = self.embed.calc_entropy()
        loss_entropy, soft_entropy = self.embed.calc_entropy_loss()

        g_loss = g_adv_loss + 100.0 * g_feat_loss + g_rec_loss

        if self.training and self.current_epoch > 700:
            g_loss = g_loss + loss_entropy
            g_loss_4save = g_loss
        else:
            g_loss_4save = g_loss + 50.0

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # ----------------------------------------------------
        # 3. Discriminator Step
        # ----------------------------------------------------
        self.toggle_optimizer(optimizer_d)
        output_det = output.detach()
        d_loss = 0.0

        for disc in all_discs:
            feats_real = disc(input)
            feats_fake = disc(output_det)

            d_real_loss = torch.mean(torch.relu(1.0 - feats_real[-1]))
            d_fake_loss = torch.mean(torch.relu(1.0 + feats_fake[-1]))
            d_loss += (d_real_loss + d_fake_loss)

        d_loss = d_loss / (2.0 * len(all_discs))

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Logging
        self.log("g_loss", g_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("d_loss", d_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("g_adv_loss", g_adv_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("g_feat_loss", g_feat_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("g_rec_loss", g_rec_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("codes_entropy", codes_entropy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("g_loss_4save", g_loss_4save, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


    def forward(self, input):
        # Extract original energy and normalize input
        energy = torch.sqrt(torch.mean(input**2, dim=-1, keepdim=True) + 1e-8)
        normalized_input = input / energy
        
        x = self.encoder(normalized_input)
        x = self.embed(x)   # quantized embeddings
        dec_out = self.decoder(x)
        
        # Normalize the decoder output before re-applying the original energy
        dec_energy = torch.sqrt(torch.mean(dec_out**2, dim=-1, keepdim=True) + 1e-8)
        normalized_dec_out = dec_out / dec_energy
        
        output = normalized_dec_out * energy
        return output


    def debugencode(self, input):
        energy = torch.sqrt(torch.mean(input**2, dim=-1, keepdim=True) + 1e-8)
        normalized_input = input / energy
        
        x = self.encoder(normalized_input)
        y = self.embed(x)  # quantized
        codes = self.embed.last_codes  # shape (B, 4, L)
        
        return input, x, y, codes, energy

    def justencode(self, x):
        return self.encoder(x)

    def encode(self, input):
        # Extract and save energy
        energy = torch.sqrt(torch.mean(input**2, dim=-1, keepdim=True) + 1e-8)
        normalized_input = input / energy
        
        x = self.encoder(normalized_input)
        _ = self.embed(x)   # forward to update last_codes
        codes = self.embed.last_codes  # (B, 4, L)
        
        return codes, energy

    def decode(self, codes, energy=None):
        x = self.embed.decode(codes)  
        dec_out = self.decoder(x)
        
        # Re-apply energy safely if provided during inference
        if energy is not None:
            dec_energy = torch.sqrt(torch.mean(dec_out**2, dim=-1, keepdim=True) + 1e-8)
            normalized_dec_out = dec_out / dec_energy
            output = normalized_dec_out * energy
            return output
            
        return dec_out


    def on_train_epoch_start(self):
        # Lightning tracks current_epoch automatically
        self.embed.update_epoch(self.current_epoch)
        if self.current_epoch>500:
            #self.embed.soft_weight = self.embed.soft_weight + 0.0004
            self.embed.soft_weight = 0.2
        if self.current_epoch >= 1000 and not self._encoder_frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self._encoder_frozen = True
            self.print(f"Encoder frozen at epoch {self.current_epoch}")
            
        #self.embed.redistribute_embeddings()

    #encode and decode functions are added.


    def train_dataloader(self):
        return self._make_dataloader(train=True)

    def val_dataloader(self):
        return self._make_dataloader(train=False)

    def _get_full_cached_dataset(self):
        """
        Build ALL sliding-window examples exactly once and keep them in RAM.

        The resulting dataset contains:
            0:590, 300:890, 600:1190, ...

        for every LibriSpeech train-clean-100 utterance.

        There is no random crop and no disk access from __getitem__ after
        this cache has been created.
        """
        import torchaudio
        import torch.nn.functional as F

        if hasattr(self, "_full_cached_dataset"):
            return self._full_cached_dataset

        if self.hparams.dataset == 'yesno':
            base_ds = torchaudio.datasets.YESNO("./data", download=True)
        elif self.hparams.dataset == 'librispeech-dev':
            base_ds = torchaudio.datasets.LIBRISPEECH("./data", url="dev-clean")
        elif DEBUG:
            # IMPORTANT:
            # Train and validation are NOT separate LibriSpeech splits here.
            # Both are created from the SAME generated window pool below.
            base_ds = torchaudio.datasets.LIBRISPEECH(
                "./dataDebug", url="train-clean-100"
            )
        elif self.hparams.dataset == 'librispeech':
            # IMPORTANT:
            # Train and validation are NOT separate LibriSpeech splits here.
            # Both are created from the SAME generated window pool below.
            base_ds = torchaudio.datasets.LIBRISPEECH(
                "./data", url="train-clean-100"
            )

        else:
            raise ValueError(f"Unsupported dataset: {self.hparams.dataset}")

        sample_rate = int(self.hparams.sample_rate)
        segment_length = int(self.hparams.segment_length)
        window_shift = int(getattr(self.hparams, "window_shift", 300))
        include_tail = bool(getattr(self.hparams, "include_tail", True))

        if window_shift <= 0:
            raise ValueError("window_shift must be > 0")

        class CachedSlidingWindowDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                dataset,
                sample_rate,
                segment_length,
                window_shift,
                include_tail=True,
            ):
                self.sample_rate = sample_rate
                self.segment_length = segment_length
                self.window_shift = window_shift
                self.include_tail = include_tail

                self.data = self._build_cache(dataset)

                ram_gb = (
                    self.data.numel() * self.data.element_size()
                    / 1024**3
                )

                print(
                    f"\nFULL RAM CACHE CREATED: "
                    f"{len(self.data):,} windows x {self.segment_length} samples "
                    f"= {self.data.numel():,} values "
                    f"({ram_gb:.2f} GB RAM)"
                )

            @torch.no_grad()
            def _build_cache(self, dataset):
                chunks = []

                for file_idx in range(len(dataset)):
                    x, source_rate, *_ = dataset[file_idx]
                    x = torch.squeeze(x).contiguous()

                    # Resample the complete utterance BEFORE windowing.
                    if source_rate != self.sample_rate:
                        x = torchaudio.functional.resample(
                            x, source_rate, self.sample_rate
                        )

                    n = x.numel()

                    if n == 0:
                        continue

                    # If an utterance is shorter than one window, create
                    # exactly one padded 590-sample example.
                    if n < self.segment_length:
                        x = F.pad(
                            x,
                            (0, self.segment_length - n),
                            mode="constant",
                            value=0.0,
                        )
                        n = self.segment_length

                    starts = list(
                        range(
                            0,
                            n - self.segment_length + 1,
                            self.window_shift,
                        )
                    )

                    # Optional final partial window.
                    if self.include_tail:
                        tail_start = starts[-1] + self.window_shift
                        if tail_start < n:
                            starts.append(tail_start)

                    for pos in starts:
                        window = x[pos:pos + self.segment_length]

                        if window.numel() < self.segment_length:
                            window = F.pad(
                                window,
                                (0, self.segment_length - window.numel()),
                                mode="constant",
                                value=0.0,
                            )

                        # Normalize once during cache creation.
                        max_val = torch.max(torch.abs(window))
                        if max_val > 0:
                            window = window * (0.95 / max_val)

                        chunks.append(window.contiguous())

                    if (file_idx + 1) % 100 == 0:
                        print(
                            f"Preparing FULL dataset: "
                            f"{file_idx + 1:,}/{len(dataset):,} files | "
                            f"windows: {len(chunks):,}"
                        )

                if not chunks:
                    raise RuntimeError("No audio windows were generated.")

                return torch.stack(chunks, dim=0)

            def __getitem__(self, index):
                # RAM only. No disk access.
                return self.data[index]

            def __len__(self):
                return self.data.shape[0]

        self._full_cached_dataset = CachedSlidingWindowDataset(
            base_ds,
            sample_rate=sample_rate,
            segment_length=segment_length,
            window_shift=window_shift,
            include_tail=include_tail,
        )

        # ------------------------------------------------------------
        # Split the GENERATED WINDOWS, not the original audio files.
        #
        # 90% -> training
        # 10% -> validation/evaluation
        #
        # The split is deterministic so the same examples remain in the
        # same split between runs.
        # ------------------------------------------------------------
        num_examples = len(self._full_cached_dataset)
        generator = torch.Generator()
        generator.manual_seed(42)

        permutation = torch.randperm(
            num_examples,
            generator=generator,
        )

        num_eval = max(1, int(num_examples * 0.10))
        num_train = num_examples - num_eval

        self._train_indices = permutation[:num_train]
        self._eval_indices = permutation[num_train:]

        print(
            f"\nWINDOW SPLIT:"
            f"\n  Total examples : {num_examples:,}"
            f"\n  Train (90%)    : {len(self._train_indices):,}"
            f"\n  Eval  (10%)    : {len(self._eval_indices):,}"
            f"\n  Split seed     : 42"
        )

        return self._full_cached_dataset

    def _make_dataloader(self, train: bool):
        def collate(examples):
            return torch.stack(examples)
        full_dataset = self._get_full_cached_dataset()

        if train:
            indices = self._train_indices
        else:
            indices = self._eval_indices

        # Subset points to the SAME RAM-resident tensor. No second copy of
        # the audio windows is made for evaluation.
        dataset = torch.utils.data.Subset(full_dataset, indices)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,       # shuffle only the 90% training examples
            collate_fn=collate,
            num_workers=0,       # important: avoid duplicating the huge RAM cache
            pin_memory=True,
        )

        return loader

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Evaluate the model on the fixed 10% window split.

        No optimizer step is performed. The reported metrics are:
          - val_g_loss
          - val_g_adv_loss
          - val_g_feat_loss
          - val_rec_loss
        """
        input = batch[:, None, :]

        all_discs = (
            list(self.mpd.discriminators)
            + list(self.ms_stft.discriminators) + [self.transformer_disc]
        )

        output = self.forward(input)

        g_adv_loss = 0.0
        g_feat_loss = 0.0

        for disc in all_discs:
            feats_real = disc(input)
            feats_fake = disc(output)

            g_adv_loss += torch.mean(
                torch.relu(1.0 - feats_fake[-1])
            )

            feat_dist = 0.0
            for f_real, f_fake in zip(
                feats_real[:-1],
                feats_fake[:-1],
            ):
                feat_dist += torch.mean(
                    torch.abs(f_real - f_fake)
                )

            g_feat_loss += feat_dist / max(
                1,
                len(feats_real) - 1,
            )

        g_adv_loss = g_adv_loss / len(all_discs)
        g_feat_loss = g_feat_loss / len(all_discs)

        g_rec_loss = self.rec_loss(
            output[:, 0, :],
            input[:, 0, :],
        )

        val_g_loss = (
            g_adv_loss
            + 100.0 * g_feat_loss
            + g_rec_loss
        )

        # Keep entropy out of the validation loss. The entropy term is
        # primarily used during the late training phase and is tied to
        # the trainable codebook state.
        self.log(
            "val_g_loss",
            val_g_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_g_adv_loss",
            g_adv_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_g_feat_loss",
            g_feat_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_rec_loss",
            g_rec_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return val_g_loss

def train():
    model = StreamableModel(
        batch_size=8192, #256,cc, 1024
        sample_rate=16_000, #16_000,
        segment_length=590, #32270,
        window_shift=300, #for dataset generation
        include_tail=True,
        padding='same',
        dataset='librispeech')

        # --- New checkpoint for best g_loss ---
    best_g_loss_ckpt = ModelCheckpoint(
        monitor='g_loss_4save',
        mode='min',
        save_top_k=1,
        filename='best-g_loss-{epoch:02d}-{g_loss_4save:.4f}',
        every_n_epochs=1,
    )

    # --- Trainer setup ---
    num_gpus = torch.cuda.device_count()
    
    trainer = pl.Trainer(
        max_epochs=30,
        log_every_n_steps=100,
        accelerator="gpu" if num_gpus > 0 else "auto",
        devices=num_gpus if num_gpus > 0 else "auto",
        precision='16-mixed',
        logger=pl.loggers.CSVLogger("."),
        strategy='ddp_find_unused_parameters_true' if num_gpus > 1 else 'auto',
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=1500),
            best_g_loss_ckpt,
        ],
        reload_dataloaders_every_n_epochs=0,
        check_val_every_n_epoch=1,

    )
    trainer.fit(
        model,
    )

    return model


if __name__ == "__main__":
    train()
