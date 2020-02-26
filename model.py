import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VQEmbeddingEMA(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.zeros(latent_dim, num_embeddings, embedding_dim)
        embedding.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(latent_dim, num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x):
        B, C, L = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, L).permute(1, 0, 3, 2)  # N, B, L, D
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.cdist(x_flat, self.embedding)
        indices = torch.argmin(distances, dim=-1)

        encodings = F.one_hot(indices, M).float()
        quantized = torch.gather(self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        return quantized.permute(1, 0, 3, 2).reshape(B, C, L), loss, perplexity.sum()


class ChannelNorm(nn.Module):
    def __init__(self,
                 n_features,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(
                torch.Tensor(1, n_features, 1),
                requires_grad=True
            )
            self.bias = nn.parameter.Parameter(
                torch.Tensor(1, n_features, 1),
                requires_grad=True
            )
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cum_mean = x.mean(dim=1, keepdim=True)
        cum_var = x.var(dim=1, keepdim=True)
        x = (x - cum_mean)*torch.rsqrt(cum_var + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, encoder_channels, z_dim, c_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, encoder_channels, 3, 1, 1),
            ChannelNorm(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 1),
            ChannelNorm(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 1),
            ChannelNorm(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 1),
            ChannelNorm(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 1),
            ChannelNorm(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 1),
            ChannelNorm(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, encoder_channels, 1),
            ChannelNorm(encoder_channels),
            nn.ReLU(True),
            nn.Conv1d(encoder_channels, z_dim, 1),
        )
        self.codebook = VQEmbeddingEMA(2, 128, 128)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)

    def forward(self, mels):
        z = self.encoder(mels)
        z, loss, perplexity = self.codebook(z)
        z = z.transpose(1, 2)
        c, _ = self.rnn(z)
        return z, c, loss, perplexity


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz, device):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class CPCLoss(nn.Module):
    def __init__(self, n_speakers, n_prediction_steps, n_utterances_per_speaker, n_negatives, z_dim, c_dim):
        super(CPCLoss, self).__init__()
        self.n_speakers = n_speakers
        self.n_prediction_steps = n_prediction_steps
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.positional = PositionalEncoding(z_dim, max_len=200)
        self.mask = None
        self.predictors = nn.ModuleList([
            nn.TransformerEncoderLayer(z_dim, 8) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c):
        length = z.size(1) - self.n_prediction_steps

        z = z.reshape(
            self.n_speakers,
            self.n_utterances_per_speaker,
            -1,
            self.z_dim
        )
        c = c[:, :-self.n_prediction_steps, :].transpose(1, 0)
        c = self.positional(c)
        if self.mask is None:
            self.mask = generate_square_subsequent_mask(c.size(0), c.device)

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, :, k:length + k, :]

            Wc = self.predictors[k-1](c, src_mask=self.mask).transpose(1, 0)
            Wc = Wc.view(
                self.n_speakers,
                self.n_utterances_per_speaker,
                -1,
                self.z_dim
            )

            batch_index = torch.randint(
                0, self.n_utterances_per_speaker,
                size=(
                    self.n_utterances_per_speaker,
                    self.n_negatives
                ),
                device=z.device
            )
            batch_index = batch_index.view(
                1, self.n_utterances_per_speaker, self.n_negatives, 1
            )

            seq_index = torch.randint(
                1, length,
                size=(
                    self.n_speakers,
                    self.n_utterances_per_speaker,
                    self.n_negatives,
                    length
                ),
                device=z.device
            )
            seq_index += torch.arange(length, device=z.device)
            seq_index = torch.remainder(seq_index, length)

            speaker_index = torch.arange(self.n_speakers, device=z.device)
            speaker_index = speaker_index.view(-1, 1, 1, 1)

            z_negatives = z_shift[speaker_index, batch_index, seq_index, :]

            zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2)

            f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1)
            f = f.view(
                self.n_speakers * self.n_utterances_per_speaker,
                self.n_negatives + 1,
                -1
            )

            labels = torch.zeros(
                self.n_speakers * self.n_utterances_per_speaker, length,
                dtype=int, device=z.device
            )

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies