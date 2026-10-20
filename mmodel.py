# Copyright (C) 2023 Katsuya Iida. All rights reserved.
# Modified to Direct VQ architecture (No Decoder, 320-word direct output, Local/Global Discriminators)

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import random
import os
import pytorch_lightning as pl
from itertools import chain
from pytorch_lightning.callbacks import ModelCheckpoint


DEBUG = True

class DirectEncoder(nn.Module):
    """
    Downsamples the input waveform by a factor of 320.
    Input: (B, 1, L)
    Output: (B, 320, L/320) 
    """
    def __init__(self, in_channels=1, out_channels=320):
        super().__init__()
        # Downsampling factor = 5 * 4 * 4 * 4 = 320
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=5, padding=5),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=9, stride=4, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, out_channels, kernel_size=9, stride=4, padding=4),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class DirectVQEmbedding(nn.Module):
    """
    Single Codebook VQ layer. 
    Selects vectors of length 320 which act DIRECTLY as the output waveform signal.
    """
    def __init__(self, num_embeddings=4096, embedding_dim=320, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z: torch.Tensor):
        # z shape: (B, 320, T)
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances: ||z - e||^2
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(z_flat, self.codebook.weight.t()))
        
        # Find closest codebook vectors
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantized vectors: (B*T, 320) -> (B, 320, T)
        z_q = self.codebook(encoding_indices).view(z.shape[0], z.shape[2], self.embedding_dim).permute(0, 2, 1)
        
        # VQ Loss (Codebook loss + Commitment loss)
        vq_loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(z_q, z.detach())
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, vq_loss, encoding_indices

    def decode(self, indices: torch.Tensor):
        # indices shape: (B, T)
        z_q = self.codebook(indices).permute(0, 2, 1) # (B, 320, T)
        return z_q


class LocalWaveDiscriminator(nn.Module):
    """
    Evaluates individual 320-length waveform chunks directly.
    """
    def __init__(self, n_channels=16):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.layers = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, n_channels, kernel_size=15, stride=2, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(n_channels, 4 * n_channels, kernel_size=11, stride=4, padding=5)),
            nn.utils.weight_norm(nn.Conv1d(4 * n_channels, 16 * n_channels, kernel_size=9, stride=4, padding=4)),
            nn.utils.weight_norm(nn.Conv1d(16 * n_channels, 1, kernel_size=3, padding=1)),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Input expected: (B * T, 1, 320)
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
            x = self.activation(x)
        feats.append(self.layers[-1](x))
        return feats


class GlobalWaveDiscriminator(nn.Module):
    """
    Evaluates the full contiguous 0.3s audio segment.
    """
    def __init__(self, resolution=1, n_channels=32):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(resolution * 2, stride=resolution) if resolution > 1 else nn.Identity()
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


class StreamableModel(pl.LightningModule):
    def __init__(
        self,
        num_embeddings: int = 4096,
        embedding_dim: int = 320,
        batch_size: int = 32,
        sample_rate: int = 24_000,
        segment_length: int = 7680,  # ~0.32 seconds (Multiple of 320)
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'librispeech',
        checkpoint: str = ''
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.encoder = DirectEncoder(in_channels=1, out_channels=embedding_dim)
        self.vq_embed = DirectVQEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.local_discriminator = LocalWaveDiscriminator()
        self.global_discriminators = nn.ModuleList([
            GlobalWaveDiscriminator(resolution=1),
            GlobalWaveDiscriminator(resolution=2)
        ])

        if checkpoint:
            self.load_state_dict(torch.load(checkpoint, map_location=self.device)['state_dict'])

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            chain(self.encoder.parameters(), self.vq_embed.parameters()),
            lr=lr, betas=(b1, b2)
        )
        optimizer_d = torch.optim.Adam(
            chain(self.local_discriminator.parameters(), self.global_discriminators.parameters()),
            lr=lr, betas=(b1, b2)
        )
        return [optimizer_g, optimizer_d], []

    def forward(self, input):
        # input: (B, 1, L)
        encoded = self.encoder(input) # (B, 320, T)
        z_q, vq_loss, _ = self.vq_embed(encoded)
        
        # Direct Reshape mapping codebook features continuously to output waveform
        # (B, 320, T) -> (B, T, 320) -> (B, 1, T * 320)
        B, D, T = z_q.shape
        output_waveform = z_q.permute(0, 2, 1).contiguous().view(B, 1, T * D)
        
        return output_waveform, vq_loss

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        input = batch[:, None, :] # [batch, 1, sequence]

        # Ensure signal aligns with 320 chunks perfectly
        B, _, L = input.shape
        assert L % 320 == 0, f"Segment length {L} must be divisible by 320."

        # Train Generator
        self.toggle_optimizer(optimizer_g)
        output, vq_loss = self.forward(input)

        # 1. Global Discriminator Loss & Feature Loss
        g_wave_loss = 0
        g_feat_loss = 0
        for i in range(len(self.global_discriminators)):
            feats_real_g = self.global_discriminators[i](input)
            feats_fake_g = self.global_discriminators[i](output)
            g_wave_loss += torch.mean(torch.relu(1 - feats_fake_g[-1]))
            g_feat_loss += sum(torch.mean(torch.abs(f1 - f2)) for f1, f2 in zip(feats_real_g[:-1], feats_fake_g[:-1])) / (len(feats_real_g) - 1)

        # 2. Local Discriminator Loss & Feature Loss (Operating on 320-length chunks)
        # Reshape (B, 1, L) -> (B * T, 1, 320)
        input_chunks = input.view(B * (L // 320), 1, 320)
        output_chunks = output.view(B * (L // 320), 1, 320)
        
        feats_real_l = self.local_discriminator(input_chunks)
        feats_fake_l = self.local_discriminator(output_chunks)
        
        g_local_loss = torch.mean(torch.relu(1 - feats_fake_l[-1]))
        g_local_feat_loss = sum(torch.mean(torch.abs(f1 - f2)) for f1, f2 in zip(feats_real_l[:-1], feats_fake_l[:-1])) / (len(feats_real_l) - 1)

        # 3. Direct L1 Reconstruction Loss on continuous wave
        g_rec_loss = F.l1_loss(output, input)

        g_adv_loss = (g_wave_loss / len(self.global_discriminators)) + g_local_loss
        total_feat_loss = (g_feat_loss / len(self.global_discriminators)) + g_local_feat_loss
        
        g_loss = g_adv_loss + (100 * total_feat_loss) + (10 * g_rec_loss) + vq_loss

        self.log('g_loss', g_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('vq_loss', vq_loss, prog_bar=True, on_step=False, on_epoch=True)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Train Discriminator
        self.toggle_optimizer(optimizer_d)
        
        # 1. Global D step
        d_global_loss = 0
        for i in range(len(self.global_discriminators)):
            feats_real_g = self.global_discriminators[i](input)
            feats_fake_g = self.global_discriminators[i](output.detach())
            d_global_loss += torch.mean(torch.relu(1 - feats_real_g[-1])) + torch.mean(torch.relu(1 + feats_fake_g[-1]))

        # 2. Local D step
        feats_real_l = self.local_discriminator(input_chunks)
        feats_fake_l = self.local_discriminator(output_chunks.detach())
        d_local_loss = torch.mean(torch.relu(1 - feats_real_l[-1])) + torch.mean(torch.relu(1 + feats_fake_l[-1]))

        d_loss = (d_global_loss / len(self.global_discriminators)) + d_local_loss

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
                x = torch.squeeze(x)
                x *= 0.95 / torch.max(x)
                if x.shape[0] < self._segment_length:
                    x = F.pad(x, [0, self._segment_length - x.shape[0]], "constant")
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                return x

            def __len__(self):
                return len(self._dataset)

        url = "train-clean-100" if train else "dev-clean"
        ds = torchaudio.datasets.LIBRISPEECH("./data", url=url)
        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        num_workers = min(12, os.cpu_count() // (torch.cuda.device_count() or 1))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams.batch_size, shuffle=True,
            collate_fn=collate, num_workers=num_workers, pin_memory=True
        )
        return loader


def train():
    # segment_length must be perfectly divisible by 320 to reshape seamlessly
    # 7680 / 24000 = 0.32 seconds.
    model = StreamableModel(
        num_embeddings=4096,
        embedding_dim=320,
        batch_size=256,
        sample_rate=24_000,
        segment_length=7680, 
        dataset='librispeech'
    )

    best_g_loss_ckpt = ModelCheckpoint(
        monitor='g_loss',
        mode='min',
        save_top_k=1,
        filename='best-g_loss-{epoch:02d}-{g_loss:.4f}',
    )

    trainer = pl.Trainer(
        max_epochs=1000,
        log_every_n_steps=2,
        devices=4 if torch.cuda.is_available() else 1,
        precision='16-mixed',
        logger=pl.loggers.CSVLogger("."),
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=1500),
            best_g_loss_ckpt,
        ],
    )
    trainer.fit(model)

    return model

if __name__ == "__main__":
    train()
