r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None

def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
        )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size = 40, h_dim = 256, z_dim = 128, x_sigma2 = 0.001, learn_rate = 0.0003, betas = (0.9, 0.999),
        )
    # ========================
    return hypers

part2_q1 = r"""
$\sigma^2$ represents the variance of the likelihood distribution. 
The function is defined by: $$\mathcal{N}(\Psi_b(z),\sigma^2 I) $$

A higher variance means a wider normal distribution when decoding back to the original space.
Specifically when the decoder function is applied on z,  vector from the latent space, it is sampling from the normal distribution with mean $ \Psi_b(z)$ 
and standard deviation $\sigma^2$. This will yield to more variability in the output values. 
A model with high variance may yield better overall results in more varied datasets but there will be higher bias error in the predicting power. This is because the model can take into account
variability in the distribution. 

A lower variance will lead to less variability, and can lead to decoded samples that are more similar to original instance space. 
This can lead to lower bias since the model will be better trained on the dataset. However, this can lead to high variance in the models. If the model cannot account for variability in the datset, then any skewed/bias data will lead to 
biased decoded output results. 
"""

part2_q2 = r"""
VAE Loss: $\mathcal{L}(\vec{\alpha},\vec{\beta})  = E_x[E_{z\sim q_a}[-logP_b(X|Z)] + D_{KL}(q_a(Z|x) || p(Z)] $

1. The VAE loss term is made up of a reconstruction loss and the KL divergence loss. 
The goal is to maximize the expected log-likelihood (the reconstruction part) while minimizing the information gained by using the posterior q(Z|X) instead of the prior distribution p(Z) (the KL part).
When minimizing the KL-loss, we're finding a Gaussian distribution $q(Z)$ that is as close as possible to $p(Z)$.
Thus making the lower bound tighter. In addition, the KL-loss term forces $q$ to be a distribution rather than a 
point-mass (that would minimize the reconstruction loss). 

2. This causes the prior latent space distribution to stay close to the parametric posterior gaussian distribution (defining the encoder). 

3. The benefit is to control the variance in the latent space distribution. 

"""

part2_q3 = r"""
The evidence distribution p(X) is the distribution of the instance space due to the generative process.
This distribution is unknown. Thus we cannot compute the expectation of log(p(X)) either. Instead we will find the greatest lower bound (infimum) so we can best approximate its value.
In order to find the distribution we will perform maximum likelihood estimate given the dataset. 
Here we maximize the evident distribution. 

In general, we don't know what is the evidence distribution $p(X)$  (very complex and infeasible to compute).
Our goal is to learn an estimation of it in order be able to generate new data-points.
We do so by maximizing a lower bound (as explained in q_2) and aim to get it as tight as possible so that we'd get 
an estimation of $p(X)$ that's as accurate as possible.
"""

part2_q4 = r"""
We model the log of the latent-space variance corresponding to an input,  $\sigma^2_{\alpha}$ ,
instead of directly modelling the variance to make the training process more numerically stable.
While $\sigma^2_{\alpha}$ 's values are by definition positive real numbers $[0, \infty]$ (usually close to zero). 
Using the log transformation(and then using the exponent of the log) not only enforces $\sigma^2_{\alpha}$'s 
values to be positive ( $log(\sigma^2_{\alpha})$ 's values can be any number in $[-\infty, \infty]$) but also 'expands' the 
region close to 0, which gives the model numerical stability (as the representation of very small positive values as 
floating points might not be accurate enough). 
In addition, the log function is differentiable in all the range it's defined upon, which allows using the derivative easily 
(Giving another aspect in which using this transformation increases the numerical stability.)
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size= 35, #128, #32 
        z_dim=9, #124 #64
        data_label=1,
        label_noise=0.1,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas = (0.3, 0.999)
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas =  (0.1, 0.999)
            # You an add extra args for the optimizer here
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
For the GAN model we need to train two models simultaneously, each with it's own separate loss function and optimizer
The goal in training the generator is to generate 'realistic' 
data that would fool the discriminator into classifying it as real.
The goal in training the discriminator is to decide whether a 
given datapoint is real or not (generated).

When training the generator, we sample a datapoint and generate an output is fed to the discriminator. We wll maintain the gradient (update the gradient) in this secion based on the generator loss so that we can update the weights in back-propogration.

When training the discriminator, we use the generator generated output and calculate it the discriminator loss.
Here we do not need to maintain the gradient because we are not training the gradient and do not need to update generator weights.



Training the GAN model is done by optimizing the losses of each part, one after another for each batch.
The discriminator is trained by sampling a datapoint using the generator and checking what is the discriminator's 
classification and updating the discriminator based on the discriminator-loss. In this part we do not maintain the 
gradients (with_grad=False) as we don't need to update the generator's weights during backpropagation.  
The generator is trained by sampling a datapoint (generating it) and then showing it to the discriminator and updating 
the generator based on the generator-loss. In this part we do maintain the gradients (with_grad=True) as we do want the 
generator's weights to be updated during backpropagation.  
"""

part3_q2 = r"""
1. No in training GAN's it is important to look at generator loss in addition to discrimiator loss. Based on GANs are designed,
the losses are inversely propertional. So as one loss goes up the other goes. Therefore, it is not enough to stop training based on generator loss alone. 

2.  If the discriminator loss remains at a constant value while the generator loss decreases, 
this means that the model is failing to converge. In this case, the genrator is getting better and better at fooling the discriminator and the discriminator is not 
learning from its discrimination mistakes. 

1. No. In GANs, the evaluation of the generator loss is done using the discriminator.
 Assume the discriminator performs poorly, meaning it is easily fooled. In such scenario,  
 the generator loss can increase even though it is not generating good samples. (Note: the other direction holds as well
 meaning that the evaluation of the discriminator depends on the generator's performance)
2. If the discriminator loss remains at constant value while the generator loss decreases, 
it means that the model is failing to converge. 
As this means the generator is able to fool the discriminator more ofter as we train 
but the discriminator is not getting better at identifying non-real data. And as both parts' evaluation depends 
on one another, this means the discriminator is not improving and not good enough to perform well.
"""

part3_q3 = r"""
Comparing the VAE results with the GAN results we can see that the VAE results are more blurry with less sharp edges, 
while the GAN results are sharper and more detailed.
We think this is the expected result as the VAE loss has a term of reconstruction loss,
forcing the output to be similar to the input in terms of MSE loss which results in blurry images and smooth edges
and making the generated images more similar to each other. On the other hand in GAN, the generator does not have
'direct access' to real images but learns how those should look through the decisions of the discriminator, 
forcing it's predictions to be more realistic (because otherwise it would be easy for the discriminator to 
detect generated images)
"""

# ==============
