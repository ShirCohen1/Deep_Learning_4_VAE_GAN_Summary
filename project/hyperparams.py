def gan_hyperparams():
    hypers = dict(
        batch_size=36,
        z_dim=16,
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=.0005,
            betas=(0.3, 0.999),

        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=.0002,
            betas=(0.1, 0.999)
            # You an add extra args for the optimizer here
        ),
        n_critic=1,
        c=0.0
    )
    return hypers


def vanilla_gan_hyperparams():
    hypers = gan_hyperparams()
    new_hypers = dict(
    batch_size=36,
    z_dim=10,
    discriminator_optimizer=dict(
        type="Adam",  # Any name in nn.optim like SGD, Adam
        lr=.0002,
        betas=(0.3, 0.999),

    ),
    generator_optimizer=dict(
        type="Adam",  # Any name in nn.optim like SGD, Adam
        lr=.0002,
        betas=(0.1, 0.999)
        # You an add extra args for the optimizer here
    )
    )
    hypers.update(new_hypers)
    return hypers


def sngan_hyperparams():
    hypers = gan_hyperparams()
    new_hypers = dict(
    batch_size = 36,
    z_dim=10,
    discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=.0002,
            betas=(0.2, 0.999),

        )

    )
    hypers.update(new_hypers)
    return hypers

def wgan_hyperparams():
    hypers = gan_hyperparams()
    new_hypers = dict(
        discriminator_optimizer=dict(
            type="RMSprop",
            lr=5e-4,
        ),
        generator_optimizer=dict(
            type="RMSprop",
            lr=5e-4,
        ),
        n_critic=5,
        c=0.01
    )
    hypers.update(new_hypers)
    return hypers

def w_sn_gan_hyperparams():
    hypers = gan_hyperparams()
    new_hypers = dict(
        discriminator_optimizer=dict(
            type="RMSprop",
            lr=2e-4,
        ),
        generator_optimizer=dict(
            type="RMSprop",
            lr=2e-4,
        ),
        n_critic=5,
        c=0.01
    )
    hypers.update(new_hypers)
    return hypers