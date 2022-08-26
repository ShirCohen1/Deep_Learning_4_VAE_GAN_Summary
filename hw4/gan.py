import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from . import autoencoder

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        for channel_in, channel_out in [(in_size[0], 64), (64, 128), (128, 256), (256, 512)]:
            modules.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                                     kernel_size=4, padding=1, stride=2, bias=False))
            if channel_in != in_size:
                modules.append(nn.BatchNorm2d(num_features=channel_out))
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        #Flatten with convolusion layer
        modules.append(nn.Conv2d(in_channels=512, out_channels=1,
                                 kernel_size=4, padding=0, stride=1, bias=False))
        modules.append(nn.Sigmoid())
        self.conv = nn.Sequential(*modules)
        self.eval()
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.conv(x)
        y = torch.squeeze(y, 3)
        y = torch.squeeze(y, 2)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        h_size = 64
        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        #hint (you dont have to use....)
        # from .autoencoder import DecoderCNN
        # from . import autoencoder

        self.decoder = nn.Sequential(
        # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, h_size * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(h_size * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 8, h_size * 4, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 4, h_size * 2, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size * 2, h_size, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(h_size),
            nn.ReLU(),
            nn.ConvTranspose2d(h_size, out_channels, featuremap_size, 2, 1, bias=False),
            nn.Tanh()
        )
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        #Create n random noise samples of dim z to input into generator
        n_samples = torch.randn(size = (n, self.z_dim), device = device)
        
        if with_grad:
            samples = self.forward(n_samples)
        else:
            with torch.no_grad():
              samples = self.forward(n_samples)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.unsqueeze(2)
        z = z.unsqueeze(3)
        x = self.decoder(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    BC_loss = nn.BCEWithLogitsLoss()
    r1 = data_label - (label_noise / 2)
    r2 =  data_label + (label_noise / 2)
    labels_rdata = torch.distributions.uniform.Uniform(r1, r2).sample(y_data.shape)
    loss_data = BC_loss(y_data, labels_rdata.to(device=y_data.device))
    r11 =  (1-data_label) - (label_noise / 2)
    r21 =  (1-data_label) + (label_noise / 2)
    labels_gdata = torch.distributions.uniform.Uniform(r11, r21).sample(y_data.shape)
    loss_generated = BC_loss(y_generated, labels_gdata.to(device=y_generated.device))
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    labels = torch.full_like(y_generated, data_label, device=y_generated.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(y_generated, labels)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    #1. show dsc real and generated data
    dsc_optimizer.zero_grad()
    #generated data
    gdata = gen_model.sample(x_data.shape[0], with_grad=False)
    gscore = dsc_model(gdata)
    #x_data is real data

    #2. Dsc loss
    rscore = dsc_model(x_data)
    dsc_loss = dsc_loss_fn(rscore, gscore)
    
    #3. Update params
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    #1. show dsc generated data
    gen_optimizer.zero_grad()
    dsc_data = dsc_model(gen_model.sample(x_data.shape[0], with_grad=True))
    
    #2. generator loss
    gen_loss = gen_loss_fn(dsc_data)
    
    #3. Update params
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    torch.save(gen_model, checkpoint_file)
    saved = True
    # ========================

    return saved