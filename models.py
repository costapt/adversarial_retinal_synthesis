"""The model definitions."""
from keras import objectives
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, merge, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Reshape, Dropout
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
import numpy as np


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Convolution2D(f, k, k, border_mode=border_mode, subsample=(s, s),
                         **kwargs)


def Deconvolution(f, output_shape, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    return Deconvolution2D(f, k, k, output_shape=output_shape,
                           subsample=(s, s), **kwargs)


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    return BatchNormalization(mode=mode, axis=axis, **kwargs)


def g_unet(in_ch, out_ch, nf, is_binary=False, name='unet'):
    """Define a U-Net.

    Input has shape in_ch x 256 x 256
    Parameters:
    - in_ch: the number of input channels;
    - out_ch: the number of output channels;
    - nf: the number of filters of the first layer;
    - is_binary: if is_binary is true, the last layer is followed by a sigmoid
    activation function, otherwise, a tanh is used.
    """
    merge_params = {
        'mode': 'concat',
        'concat_axis': 1
    }

    i = Input(shape=(in_ch, 256, 256))

    # in_ch x 256 x 256
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 16 x 16

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 8 x 8

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 4 x 4

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 2 x 2

    conv8 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 1 x 1

    dconv1 = Deconvolution(nf * 8, (None, nf*8, 2, 2), k=2, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    dconv1 = Dropout(0.5)(dconv1)
    x = merge([dconv1, conv7], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(nf * 8, (None, nf*8, 4, 4))(x)
    dconv2 = BatchNorm()(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = merge([dconv2, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(nf * 8, (None, nf*8, 8, 8))(x)
    dconv3 = BatchNorm()(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    x = merge([dconv3, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(nf * 8, (None, nf*8, 16, 16))(x)
    dconv4 = BatchNorm()(dconv4)
    x = merge([dconv4, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(nf * 4, (None, nf*4, 32, 32))(x)
    dconv5 = BatchNorm()(dconv5)
    x = merge([dconv5, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(4 + 4) x 32 x 32

    dconv6 = Deconvolution(nf * 2, (None, nf*2, 64, 64))(x)
    dconv6 = BatchNorm()(dconv6)
    x = merge([dconv6, conv2], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(2 + 2) x 64 x 64

    dconv7 = Deconvolution(nf, (None, nf, 128, 128))(x)
    dconv7 = BatchNorm()(dconv7)
    x = merge([dconv7, conv1], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(1 + 1) x 128 x 128

    dconv8 = Deconvolution(out_ch, (None, out_ch, 256, 256))(x)

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(dconv8)

    unet = Model(i, out, name=name)

    return unet


def g_vae(in_ch, out_ch, nf, latent_dim, is_binary=False, name='vae'):
    """
    Define a Variational Auto Encoder.

    Params:
    - in_ch: number of input channels.
    - out_ch: number of output channels.
    - nf: number of filters of the first layer.
    - latent_dim: the number of latent factors to use.
    - is_binary: whether the output is binary or not.
    """
    i = Input(shape=(in_ch, 256, 256))

    # in_ch x 256 x 256
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    conv1 = Convolution(nf, s=1)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    conv2 = Convolution(nf * 2, s=1)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    conv3 = Convolution(nf * 4, s=1)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    conv4 = Convolution(nf * 8, s=1)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 16 x 16

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    x = Dropout(0.5)(x)
    conv5 = Convolution(nf * 8, s=1)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    x = Dropout(0.5)(x)
    # nf*8 x 8 x 8

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    x = Dropout(0.5)(x)
    conv6 = Convolution(nf * 8, s=1)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    x = Dropout(0.5)(x)
    # nf*8 x 4 x 4

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    x = Dropout(0.5)(x)
    conv7 = Convolution(nf * 8, s=1)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    x = Dropout(0.5)(x)
    # nf*8 x 2 x 2

    conv8 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 1 x 1

    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch_size = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., std=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(i, z, name='{0}_encoder'.format(name))

    ig = Input(shape=(latent_dim,))

    x = Dense(nf * 8)(ig)
    x = Reshape((nf * 8, 1, 1))(x)

    x = UpSampling2D(size=(2, 2))(x)
    dconv1 = Convolution(nf * 8, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    x = LeakyReLU(0.2)(dconv1)
    # nf*8 x 2 x 2

    x = UpSampling2D(size=(2, 2))(x)
    dconv2 = Convolution(nf * 8, s=1)(x)
    dconv2 = BatchNorm()(dconv2)
    x = LeakyReLU(0.2)(dconv2)
    x = Dropout(0.5)(x)
    # nf*8 x 4 x 4

    x = UpSampling2D(size=(2, 2))(x)
    dconv3 = Convolution(nf * 8, s=1)(x)
    dconv3 = BatchNorm()(dconv3)
    x = LeakyReLU(0.2)(dconv3)
    x = Dropout(0.5)(x)
    # nf*8 x 8 x 8

    x = UpSampling2D(size=(2, 2))(x)
    dconv4 = Convolution(nf * 8, s=1)(x)
    dconv4 = BatchNorm()(dconv4)
    x = LeakyReLU(0.2)(dconv4)
    x = Dropout(0.5)(x)
    # nf*8 x 16 x 16

    x = UpSampling2D(size=(2, 2))(x)
    dconv5 = Convolution(nf * 4, s=1)(x)
    dconv5 = BatchNorm()(dconv5)
    x = LeakyReLU(0.2)(dconv5)
    dconv5 = Convolution(nf * 4, s=1)(x)
    dconv5 = BatchNorm()(dconv5)
    x = LeakyReLU(0.2)(dconv5)
    # nf*4 x 32 x 32

    x = UpSampling2D(size=(2, 2))(x)
    dconv6 = Convolution(nf * 2, s=1)(x)
    dconv6 = BatchNorm()(dconv6)
    x = LeakyReLU(0.2)(dconv6)
    dconv6 = Convolution(nf * 2, s=1)(x)
    dconv6 = BatchNorm()(dconv6)
    x = LeakyReLU(0.2)(dconv6)
    # nf*2 x 64 x 64

    x = UpSampling2D(size=(2, 2))(x)
    dconv7 = Convolution(nf, s=1)(x)
    dconv7 = BatchNorm()(dconv7)
    x = LeakyReLU(0.2)(dconv7)
    dconv7 = Convolution(nf, s=1)(x)
    dconv7 = BatchNorm()(dconv7)
    x = LeakyReLU(0.2)(dconv7)
    # nf x 128 x 128

    x = UpSampling2D(size=(2, 2))(x)
    dconv8 = Convolution(nf, s=1)(x)
    dconv8 = BatchNorm()(dconv8)
    x = LeakyReLU(0.2)(dconv8)
    dconv8 = Convolution(nf, s=1)(x)
    dconv8 = BatchNorm()(dconv8)
    x = LeakyReLU(0.2)(dconv8)
    x = Convolution(out_ch, k=1, s=1)(x)

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(x)

    decoder = Model(ig, out, name='{0}_decoder'.format(name))

    def vae_loss(a, ap):
        a_flat = K.batch_flatten(a)
        ap_flat = K.batch_flatten(ap)

        L_atoa = objectives.binary_crossentropy(a_flat, ap_flat)
        return 100 * L_atoa

    vae = Model(i, decoder(encoder(i)), name=name)
    vaeopt = Adam(lr=1e-4)
    vae.compile(optimizer=vaeopt, loss=vae_loss)

    return vae


def discriminator(a_ch, b_ch, nf, opt=Adam(lr=2e-4, beta_1=0.5), name='d'):
    """Define the discriminator network.

    Parameters:
    - a_ch: the number of channels of the first image;
    - b_ch: the number of channels of the second image;
    - nf: the number of filters of the first layer.
    """
    i = Input(shape=(a_ch + b_ch, 256, 256))

    # (a_ch + b_ch) x 256 x 256
    conv1 = Convolution(nf)(i)
    x = LeakyReLU(0.2)(conv1)
    # nf x 128 x 128

    conv2 = Convolution(nf*2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 64 x 64

    conv3 = Convolution(nf * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 32 x 32

    conv4 = Convolution(1)(x)
    out = Activation('sigmoid')(conv4)
    # 1 x 16 x 16

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=opt, loss=d_loss)
    return d


def code_discriminator(latent_dim, nf, opt=Adam(), name='code_d'):
    """Define the discriminator that validates the latent code."""
    z = Input(shape=(latent_dim,))

    h1 = Dense(nf)(z)
    x = LeakyReLU(0.2)(h1)

    h2 = Dense(1)(x)
    out = Activation('sigmoid')(h2)

    d = Model(z, out)

    d.compile(optimizer=opt, loss='binary_crossentropy')
    return d


def pix2pix(atob, d, a_ch, b_ch, alpha=100, is_a_binary=False,
            is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5), name='pix2pix'):
    """Define the pix2pix network."""
    a = Input(shape=(a_ch, 256, 256))
    b = Input(shape=(b_ch, 256, 256))

    # A -> B'
    bp = atob(a)

    # Discriminator receives the pair of images
    d_in = merge([a, bp], mode='concat', concat_axis=1)

    pix2pix = Model([a, b], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):
        # Flatten the output of the discriminator. For some reason, applying
        # the loss direcly on the tensors was not working
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = objectives.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        return L_adv + alpha*L_atob

    # This network is used to train the generator. Freeze the discriminator
    # part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)

    return pix2pix


def pix2pix2pix(vae, atob, d, code_d, a_ch, b_ch, alpha=100, beta=100, is_a_binary=False,
                is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5),
                name='pix2pix2pix'):
    """
    Define the pix2pix2pix network.

    Generator converts A -> A' -> B' and discriminator checks if A'/B' is a
    valid pair.

    The A -> A' transofrmation is performed by a VAE to make sure that
    the bottleneck can be sampled by a gaussian distribution.

    Then, it is possible to sample from z -> A' -> B' to get generated A'/B'
    pairs from z.

    Parameters:
    - vae: a model for a variational auto encoder. Needs to bee composed of
    3 models: a 'vae_encoder' that maps an image to the parameters of the
    distribution (mean, var); a 'vae_sampler' that samples from the previous
    distribution; and a 'vae_decoder' that maps a sample from a distribution
    to an image.
    - atob: a standard auto encoder.
    - d: the discriminator model. Must have the name 'd'.
    - alpha: the weight of the reconstruction term of the atob model in relation
    to the adversarial term. See the pix2pix paper.
    - beta: the weight of the reconstruction term of the atoa model in relation
    to the adversarial term.
    """
    a = Input(shape=(a_ch, 256, 256))
    b = Input(shape=(b_ch, 256, 256))

    # A -> A'
    encoder = vae.get_layer('vae_encoder')
    decoder = vae.get_layer('vae_decoder')

    z = encoder(a)
    ap = decoder(z)

    # A' -> B'
    bp = atob(ap)

    # Discriminator receives the two generated images
    d_in = merge([ap, bp], mode='concat', concat_axis=1)

    gan = Model([a, b], d(d_in), name=name)

    def gan_loss(y_true, y_pred):
        # Flatten the output of the discriminator. For some reason, applying
        # the loss direcly on the tensors was not working
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to A loss
        a_flat = K.batch_flatten(a)
        ap_flat = K.batch_flatten(ap)
        if is_a_binary:
            L_atoa = objectives.binary_crossentropy(a_flat, ap_flat)
        else:
            L_atoa = K.mean(K.abs(a_flat - ap_flat))

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = objectives.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        L_code = objectives.binary_crossentropy(np.asarray(1).astype('float32').reshape((-1, 1)), code_d(z))

        return L_adv + beta*L_atoa + alpha*L_atob + L_code

    # This network is used to train the generator. Freeze the discriminator
    # part
    gan.get_layer('d').trainable = False

    gan.compile(optimizer=opt, loss=gan_loss)

    return gan


def conditional_generator(atoa, atob, a_ch):
    """Merge the two models into one generator model that goes from a to b."""
    i = Input(shape=(a_ch, 256, 256))
    g = Model(i, atob(atoa(i)))

    return g


def generator(vae, atob, latent_dim):
    """Create a model that generates a pair of images."""
    i = Input(shape=(latent_dim,))

    decoder = vae.get_layer('vae_decoder')

    ap = decoder(i)
    bp = atob(ap)

    g = Model(i, [ap, bp])

    return g


def generator_from_conditional_generator(g, latent_dim):
    """Create a generator from a conditional generator."""
    vae = g.layers[1]
    atob = g.layers[2]

    return generator(vae, atob, latent_dim)
