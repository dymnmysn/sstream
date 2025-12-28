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



class Embedding(nn.Module):
    def __init__(self, num_embeddings=2048, embedding_dim=512,
                 start_t=1, end_t=10, total_epochs=1000, eps=1e-8):
        super().__init__()
        assert num_embeddings % 4 == 0, "num_embeddings must be divisible by 4"
        quarter = num_embeddings // 4

        # Separate embedding tables for each quarter
        self.embeddings    = nn.Parameter(torch.randn(quarter, embedding_dim))
        self.embeddings_r1 = nn.Parameter(torch.randn(quarter, embedding_dim))
        self.embeddings_r2 = nn.Parameter(torch.randn(quarter, embedding_dim))
        self.embeddings_r3 = nn.Parameter(torch.randn(quarter, embedding_dim))

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
            # Randomly choose how many terms to sum
            k = torch.randint(low=0, high=4, size=(1,), device=decoded_0.device).item()
            if k == 0:
                decoded = decoded_0
            elif k == 1:
                decoded = decoded_0 + decoded_1
            elif k == 2:
                decoded = decoded_0 + decoded_1 + decoded_2
            else:
                decoded = decoded_0 + decoded_1 + decoded_2 + decoded_3
        else:
            # During eval, always use all terms
            decoded = decoded_0 + decoded_1 + decoded_2 + decoded_3

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
        else:
            attn_0, out_0, codes_0 = self.process_block_max(logits_0, self.embeddings, offset=0*q, k=3)
            attn_1, out_1, codes_1 = self.process_block_max(logits_1, self.embeddings_r1, offset=1*q, k=3)
            attn_2, out_2, codes_2 = self.process_block_max(logits_2, self.embeddings_r2, offset=2*q, k=3)
            attn_3, out_3, codes_3 = self.process_block_max(logits_3, self.embeddings_r3, offset=3*q, k=3)
        """
        elif self.training and self.current_epoch<800:
            attn_0, out_0, codes_0 = self.process_block_max(logits_0, self.embeddings, offset=0*q, k=3)
            attn_1, out_1, codes_1 = self.process_block_max(logits_1, self.embeddings_r1, offset=1*q, k=3)
            attn_2, out_2, codes_2 = self.process_block_max(logits_2, self.embeddings_r2, offset=2*q, k=3)
            attn_3, out_3, codes_3 = self.process_block_max(logits_3, self.embeddings_r3, offset=3*q, k=3)
        """

        

        # Combine outputs
        output = out_0 + out_1 + out_2 + out_3

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
                noise = torch.randn(self.embedding_dim, device=self.embeddings.device) * noise_scale
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


class ResNet2d(nn.Module):
    def __init__(
        self,
        n_channels: int,
        factor: int,
        stride: Tuple[int, int]
    ) -> None:
        # https://arxiv.org/pdf/2005.00341.pdf
        # The original paper uses layer normalization, but here
        # we use batch normalization.
        super().__init__()
        self.conv0 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            padding='same')
        self.bn0 = nn.BatchNorm2d(
            n_channels
        )
        self.conv1 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride)
        self.bn1 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.conv2 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=1,
            stride=stride)
        self.bn2 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.pad = nn.ReflectionPad2d([
            (stride[1] + 1) // 2,
            (stride[1] + 2) // 2,
            (stride[0] + 1) // 2,
            (stride[0] + 2) // 2,
        ])
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut
        y = self.conv2(input)
        y = self.bn2(y)

        x += y
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
        self, n_fft: int = 1024, hop_length: int = 256,
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
        x = torch.stft(x, self.n_fft, self.hop_length, normalized=True, onesided=True, return_complex=True)
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
        for i in range(6, 12):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss / (12 - 6)


class ReconstructionLoss2(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, sample_rate, eps=1e-5):
        super().__init__()
        import torchaudio
        self.layers = nn.ModuleList()
        self.alpha = []
        self.eps = eps
        for i in range(6, 12):
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


class StreamableModel(pl.LightningModule):
    def __init__(
        self,
        n_channels: int = 32,
        num_embeddings: int = 512, #1024
        padding: str = "valid",
        batch_size: int = 32,
        sample_rate: int = 24_000,
        segment_length: int = 8000, #32270,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'librispeech',
        version: str = 'quadsingletrain',
        checkpoint: str = ''#'/ari/users/ibaskaya/projeler/sstream/lightning_logs/version_55/checkpoints/last.ckpt'
    ) -> None:
        # https://arxiv.org/pdf/2009.02095.pdf
        # 2. Method
        # SEANet uses Adam with lr=1e-4, beta1=0.5, beta2=0.9
        # batch_size=16
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.encoder = Encoder(n_channels, padding)
        self.decoder = Decoder(n_channels, padding)

        self.wave_discriminators = nn.ModuleList([
            WaveDiscriminator(resolution=1),
            WaveDiscriminator(resolution=2),
            WaveDiscriminator(resolution=4)
        ])
        self.rec_loss = ReconstructionLoss()
        self.stft_discriminator = STFTDiscriminator()
        self.embed = Embedding(num_embeddings=512*4, embedding_dim=512, 
                 start_t=10.0, end_t=0.0, total_epochs=100, eps=1e-8)

        if checkpoint:
            self.load_state_dict(torch.load(checkpoint,self.device)['state_dict'])

        self.embed.soft_weight = 0.00001

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ),
            lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(
            chain(
                self.wave_discriminators.parameters(),
                self.stft_discriminator.parameters()
            ),
            lr=lr, betas=(b1, b2))#
        return [optimizer_g, optimizer_d], []


    def forward(self, input):
        x = self.encoder(input)
        x = self.embed(x)   # quantized embeddings
        x = self.decoder(x)
        return x

    def debugencode(self, input):
        x = self.encoder(input)
        y = self.embed(x)  # quantized
        codes = self.embed.last_codes  # shape (B, 2, L)
        return input, x, y, codes

    def justencode(self, x):
        return self.encoder(x)

    def encode(self, input):
        x = self.encoder(input)
        _ = self.embed(x)   # forward to update last_codes
        codes = self.embed.last_codes  # (B, 2, L)
        return codes

    def decode(self, codes):
        x = self.embed.decode(codes)  # handle (B, 2, L) internally
        x = self.decoder(x)
        return x

    def on_train_epoch_start(self):
        # Lightning tracks current_epoch automatically
        self.embed.update_epoch(self.current_epoch)
        if self.current_epoch>500:
            #self.embed.soft_weight = self.embed.soft_weight + 0.0004
            self.embed.soft_weight = 0.2
        #self.embed.redistribute_embeddings()

    #encode and decode functions are added.

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        input = batch[:, None, :]
        # input: [batch, channel, sequence]

        # train generator
        self.toggle_optimizer(optimizer_g)
        output = self.forward(input)
        # output: [batch, channel, sequence]
        # print(input.shape, output.shape)

        stft_out = self.stft_discriminator(output)
        g_stft_loss = torch.mean(torch.relu(1 - stft_out))
        self.log("g_stft_loss", g_stft_loss, on_step=False, on_epoch=True)

        g_wave_loss = 0
        g_feat_loss = 0
        for i in range(3):
            feats1 = self.wave_discriminators[i](input)
            feats2 = self.wave_discriminators[i](output)
            assert len(feats1) == len(feats2)
            g_wave_loss += torch.mean(torch.relu(1 - feats2[-1]))
            g_feat_loss += sum(torch.mean(
                torch.abs(f1 - f2))
                for f1, f2 in zip(feats1[:-1], feats2[:-1])) / (len(feats1) - 1)
        self.log("g_wave_loss", g_wave_loss / 3, on_step=False, on_epoch=True)
        self.log("g_feat_loss", g_feat_loss / 3, on_step=False, on_epoch=True)

        g_rec_loss = self.rec_loss(output[:, 0, :], input[:, 0, :])
        self.log("g_rec_loss", g_rec_loss, prog_bar=True, on_step=False, on_epoch=True)

        codes_entropy = self.embed.calc_entropy()

        loss_entropy, soft_entropy = self.embed.calc_entropy_loss()

        g_feat_loss = g_feat_loss / 3
        g_adv_loss = (g_stft_loss + g_wave_loss) / 4
        g_loss = g_adv_loss + 100 * g_feat_loss + g_rec_loss 

        #if not self.current_epoch%2:
        #    g_loss = g_loss + loss_entropy
        ##
        if self.training and self.current_epoch>700:
            g_loss = g_loss + loss_entropy
            g_loss_4save = g_loss
        else:
            g_loss_4save = g_loss + 50
        ##
        
        self.log('g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('g_loss_4save', g_loss_4save, prog_bar=True, on_step=False, on_epoch=True)
        self.log("codes_entropy", codes_entropy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_entropy", loss_entropy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("soft_entropy", soft_entropy, prog_bar=True, on_step=False, on_epoch=True)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)


        # train discriminator
        self.toggle_optimizer(optimizer_d)
        output = self.forward(input)

        stft_out = self.stft_discriminator(input)
        d_stft_loss = torch.mean(torch.relu(1 - stft_out))
        stft_out = self.stft_discriminator(output)
        d_stft_loss += torch.mean(torch.relu(1 + stft_out))

        d_wave_loss = 0
        for i in range(3):
            feats = self.wave_discriminators[i](input)
            d_wave_loss += torch.mean(torch.relu(1 - feats[-1]))
            feats = self.wave_discriminators[i](output)
            d_wave_loss += torch.mean(torch.relu(1 + feats[-1]))

        d_loss = (d_stft_loss + d_wave_loss) / 4

        self.log("d_stft_loss", d_stft_loss)
        self.log("d_wave_loss", d_wave_loss / 3)

        d_loss = (d_stft_loss + d_wave_loss) / 4
        self.log("d_loss", d_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def train_dataloader(self):
        return self._make_dataloader(True)

    def _make_dataloader(self, train: bool):
        import torchaudio

        def collate(examples):
            return torch.stack(examples)

        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, sample_rate, segment_length):
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                import random
                x, sample_rate, *_ = self._dataset[index]
                x = torchaudio.functional.resample(x, sample_rate, self._sample_rate)
                assert x.shape[0] == 1
                x = torch.squeeze(x)
                x *= 0.95 / torch.max(x)
                assert x.dim() == 1
                if x.shape[0] < self._segment_length:
                    x = F.pad(x, [0, self._segment_length - x.shape[0]], "constant")
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                return x

            def __len__(self):
                return len(self._dataset)

        if self.hparams.dataset == 'yesno':
            ds = torchaudio.datasets.YESNO("./data", download=True)
        elif self.hparams.dataset == 'librispeech-dev':
            ds = torchaudio.datasets.LIBRISPEECH("./data", url="dev-clean")
        elif self.hparams.dataset == 'librispeech':
            url = "train-clean-100" if train else "dev-clean"
            ds = torchaudio.datasets.LIBRISPEECH("./data", url=url)
        else:
            raise ValueError()
        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        numberof_workers = min(12, os.cpu_count() // torch.cuda.device_count())
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=True,
            collate_fn=collate,
            num_workers=numberof_workers,         #  add this
            pin_memory=True       #  optional, helps with GPU transfers
            )
        return loader


def train():
    model = StreamableModel(
        batch_size=512, #256,cc
        sample_rate=24_000, #16_000,
        segment_length=11790, #32270,
        padding='same',
        dataset='librispeech')

        # --- New checkpoint for best g_loss ---
    best_g_loss_ckpt = ModelCheckpoint(
        monitor='g_loss_4save',
        mode='min',
        save_top_k=1,
        filename='best-g_loss-{epoch:02d}-{g_loss_4save:.4f}',
    )

    # --- Trainer setup ---
    trainer = pl.Trainer(
        max_epochs=1000,  # Bura benden: 10000 idi
        log_every_n_steps=2,
        devices=4,  # Bura benden
        precision='16-mixed',
        logger=pl.loggers.CSVLogger("."),
        # logger=pl.loggers.TensorBoardLogger("lightning_logs", name="soundstream"),
        strategy='ddp_find_unused_parameters_true',  # Bura benden
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=1500),  # mevcut
            best_g_loss_ckpt,  # 
        ],
    )
    trainer.fit(
        model,
    )

    return model


if __name__ == "__main__":
    train()
