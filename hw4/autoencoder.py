import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        hidden_layers = [64, 128, out_channels]
        for i in range(len(hidden_layers)):
          if i == 0:
              modules.append(nn.Conv2d(in_channels, hidden_layers[0], (5, 5), padding=2, stride=2))
          else:
              modules.append(nn.Conv2d(hidden_layers[i-1], hidden_layers[i], (5, 5), padding=2, stride=2))
          modules.append(nn.BatchNorm2d(hidden_layers[i]))
          modules.append(nn.ReLU())
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        hidden_layers = [256, 128, 32]
        for i in range(len(hidden_layers)):
          if i == 0:
              modules.append(nn.ConvTranspose2d(in_channels, hidden_layers[0], (5, 5), padding=2, stride=2,
                                                output_padding=1))
          else:
              modules.append(nn.ConvTranspose2d(hidden_layers[i - 1], hidden_layers[i], (5, 5), padding=2, stride=2,
                                                output_padding=1))
          modules.append(nn.BatchNorm2d(hidden_layers[i]))
          modules.append(nn.ReLU())
        modules.append(nn.Conv2d(32, out_channels, kernel_size=5, padding=2, stride=1))
        modules.append(nn.Tanh())
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return self.cnn(h) #torch.tanh(self.cnn(h))

class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim
        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        temp_h_dim = 1024
        self.x_h = nn.Linear(n_features, temp_h_dim, bias=False)
        self.h_mu = nn.Linear(temp_h_dim, z_dim)
        self.h_sigma = nn.Linear(temp_h_dim, z_dim)
        self.z_hwave = nn.Linear(z_dim, n_features, bias=False)
        self.encoder_batchnorm = nn.BatchNorm1d(temp_h_dim, momentum=0.9)
        self.decoder_batchnorm = nn.BatchNorm1d(n_features, momentum=0.9)
        self.eval()
        self.add_module(f'x_h', self.x_h)
        self.add_module(f'h_mu', self.h_mu)
        self.add_module(f'h_sigma', self.h_sigma)
        self.add_module(f'z_hwave', self.z_hwave)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        # Extract image features
        encoded = self.features_encoder(x)
        encoded_flat = torch.flatten(encoded, start_dim=1)

        # Transform into mu and log_sigma2
        h_temp = self.x_h(encoded_flat)
        h_norm = self.encoder_batchnorm(h_temp)
        h = nn.ReLU(True)(h_norm)
        mu = self.h_mu(h)
        log_sigma2 = self.h_sigma(h)

        # Reparametrization trick: Sample u from an isotropic gaussian and
        # use it to create z.
        device = next(self.parameters()).device
        u = torch.randn(x.shape[0], self.z_dim).to(device)
        z = mu + (torch.exp(log_sigma2) ** 0.5) * u
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h_temp = self.z_hwave(z)
        h_norm = self.decoder_batchnorm(h_temp)
        h_wave = nn.ReLU(True)(h_norm)
        size = tuple([h_wave.shape[0]]) + self.features_shape
        h_wave = h_wave.reshape(size)
        x_rec = self.features_decoder(h_wave)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            samples = self.decode(torch.randn(size=(n, self.z_dim), device=device))
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    batch_size = x.shape[0]

    sigma = torch.exp(z_log_sigma2)

    x_flat = torch.flatten(x, start_dim=1)
    xr_flat = torch.flatten(xr, start_dim=1)

    data_loss = torch.norm(x_flat - xr_flat) ** 2
    data_loss = data_loss / (x_sigma2 * x_flat.shape[1])

    kldiv_loss = torch.sum(sigma)

    mu2 = torch.norm(z_mu) ** 2
    mu2 = torch.sum(mu2)

    z_dim = z_log_sigma2.shape[1]

    logdet = torch.log(sigma)
    logdet = torch.sum(logdet)

    loss = (data_loss + kldiv_loss + mu2 - logdet) / batch_size - z_dim
    data_loss = data_loss / batch_size
    kldiv_loss = kldiv_loss / batch_size  
    # ========================

    return loss, data_loss, kldiv_loss
