"""The script used to train the model."""
import os
import sys
import getopt

import numpy as np
import models as m

from tqdm import tqdm
from keras.optimizers import Adam
from util.data import TwoImageIterator
from util.util import MyDict, log, save_weights, load_weights, load_losses, create_expt_dir
from keras import backend as K


def print_help():
    """Print how to use this script."""
    print "Usage:"
    print "train.py [--help] [--nfd] [--nfatoa] [--nfatob] [--a_ch] [--b_ch] " \
          "[--is_a_binary] [--is_b_binary] [--is_a_grayscale] [--is_b_grayscale] " \
          "[--log_dir] [--base_dir] [--train_dir] [--val_dir] [--load_to_memory] " \
          "[--epochs] [--batch_size] [--latent_dim] [--alpha] [--beta] [--save_every] " \
          "[--lr] [--beta_1] [--continue_train] [--train_samples] [--val_samples] " \
          "[--rotation_range] [--height_shift_range] [--width_shift_range] " \
          "[--horizontal_flip] [--vertical_flip] [--zoom_range] [--pix2pix] " \
          "[--expt_name] [--target_size]"
    print "--nfd: Number of filters of the first layer of the discriminator."
    print "--nfatoa: Number of filters of the first layer of the AtoA model."
    print "--nfatob: Number of filters of the first layer of the AtoB model."
    print "--a_ch: Number of channels of images A."
    print "--b_ch: Number of channels of images B."
    print "--is_a_binary: If A is binary, the last layer of the atoa model is " \
          "followed by a sigmoid. Otherwise, a tanh is used. When the sigmoid is " \
          "used, the binary crossentropy loss is used. For the tanh, the L1 is used."
    print "--is_b_binary: If B is binary, the last layer of the atob model is " \
          "followed by a sigmoid. Otherwise, a tanh is used. When the sigmoid is " \
          "used, the binary crossentropy loss is used. For the tanh, the L1 is used."
    print "--is_a_grayscale: If A images are grayscale."
    print "--is_b_grayscale: If B images are grayscale."
    print "--log_dir: The directory to place the logs."
    print "--base_dir: Directory that contains the data."
    print "--train_dir: Directory inside base_dir that contains training data. " \
          "Must contain an A and B folder."
    print "--val_dir: Directory inside base_dir that contains validation data. " \
          "Must contain an A and B folder."
    print "--load_to_memory: Whether to load images into memory or read from the filesystem."
    print "--epochs: Number of epochs to train the model."
    print "--batch_size: The number of samples to train each model on each iteration."
    print "--latent_dim: The dimension of the latent space. Necessary when training the VAE."
    print "--alpha: The weight of the reconstruction loss of the atob model."
    print "--beta: The weight of the reconstruction loss of the atoa model."
    print "--save_evey: Save results every 'save_every' epochs on the log folder."
    print "--lr: The learning rate to train the models."
    print "--beta_1: The beta_1 value of the Adam optimizer."
    print "--continue_train: If it should continue the training from the last checkpoint."
    print "--train_samples: The number of training samples. Set -1 to be the same as training examples."
    print "--val_samples: The number of validation samples. Set -1 to be the same as validation examples."
    print "--rotation_range: The range to rotate training images for dataset augmentation."
    print "--height_shift_range: Percentage of height of the image to translate for dataset augmentation."
    print "--width_shift_range: Percentage of width of the image to translate for dataset augmentation."
    print "--horizontal_flip: If true performs random horizontal flips on the train set."
    print "--vertical_flip: If true performs random vertical flips on the train set."
    print "--zoom_range: Defines the range to scale the image for dataset augmentation."
    print "--pix2pix: If true only trains a pix2pix. Otherwise it trains the pix2pix2pix."
    print "--expt_name: The name of the experiment. Saves the logs into a folder with this name."
    print "--target_size: The size of the images loaded by the iterator. THIS DOES NOT CHANGE THE MODELS. " \
          "If you want to accept images of different sizes you will need to update the models.py files."


def discriminator_generator(it, g, dout_size=(16, 16)):
    """Generate batches for the discriminator."""
    # Sample fake and real pairs
    for a, b in it:
        a_fake = a
        b_fake = g.predict(a_fake)

        a_real, b_real = next(it)

        # Concatenate the channels. Images become (ch_a + ch_b) x 256 x 256
        fake = np.concatenate((a_fake, b_fake), axis=1)
        real = np.concatenate((a_real, b_real), axis=1)

        # Concatenate fake and real pairs into a single batch
        batch_x = np.concatenate((fake, real), axis=0)

        batch_y = np.zeros((batch_x.shape[0], 1) + dout_size)
        batch_y[fake.shape[0]:] = 1

        yield batch_x, batch_y


def train_discriminator(d, it, batch_size=20):
    """
    Train the discriminator network.

    Parameters:
    - d: the discriminator to be trained;
    - it: an iterator that returns batches of pairs of images.
    """
    return d.fit_generator(it, samples_per_epoch=batch_size*2, nb_epoch=1, verbose=False)


def code_discriminator_generator(it, encoder, dout_size=(16, 16)):
    """Define a generator that produces data for the full generator network."""
    for a, _ in it:
        z_fake = encoder.predict(a)

        z_real = np.random.normal(loc=0., scale=1., size=z_fake.shape)

        # concatenate fake and real pairs
        batch_x = np.concatenate((z_fake, z_real), axis=0)

        # 0 is fake, 1 is real
        y = np.zeros((batch_x.shape[0], 1))
        y[z_fake.shape[0]:] = 1

        yield batch_x, y


def generator_generator(it, dout_size=(16, 16)):
    """Define the generator that produces data for the generator network."""
    for a, b in it:
        # 0 is fake, 1 is real
        y = np.ones((a.shape[0], 1) + dout_size)
        yield [a, b], y


def train_generator(gan, it, batch_size=20):
    """Train the generator network."""
    return gan.fit_generator(it, nb_epoch=1, samples_per_epoch=batch_size, verbose=False)


def evaluate(models, generators, losses, val_samples=192, verbose=True):
    """Evaluate and display the losses of the models."""
    # Get necessary generators
    d_gen = generators.d_gen_val
    p2p_train_gen = generators.p2p_gen
    p2p_gen = generators.p2p_gen_val
    p2p2p_gen = generators.p2p2p_gen_val
    code_d_gen = generators.code_d_gen_val

    # Get necessary models
    d = models.d
    p2p = models.p2p
    code_d = models.code_d
    p2p2p = models.p2p2p

    d_loss = d.evaluate_generator(d_gen, val_samples)
    p2p_loss = p2p.evaluate_generator(p2p_gen, val_samples)
    p2p_train_loss = p2p.evaluate_generator(p2p_train_gen, val_samples)

    if p2p2p is not None and p2p2p_gen is None:
        print 'WARNING: There is a Pix2Pix2Pix model to evaluate but no generator.'

    p2p2p_loss = None
    if p2p2p is not None:
        p2p2p_loss = p2p2p.evaluate_generator(p2p2p_gen, val_samples)

    if code_d is not None and code_d_gen is None:
        print 'WARNING: There is a latent discriminator model to evaluate but no generator.'

    code_d_loss = None
    if code_d is not None:
        code_d_loss = code_d.evaluate_generator(code_d_gen, val_samples)

    losses['d_val'].append(d_loss)
    losses['p2p_val'].append(p2p_loss)
    losses['p2p'].append(p2p_train_loss)
    if p2p2p is not None:
        losses['p2p2p_val'].append(p2p2p_loss)
        losses['code_d_val'].append(code_d_loss)

    if not verbose:
        return d_loss, p2p_loss, p2p2p_loss, code_d_loss

    if p2p2p is None:
        print ''
        print ('Train Losses of (D={0} / P2P={1});\n'
               'Validation Losses of (D={2} / P2P={3})'.format(
                    losses['d'][-1], losses['p2p'][-1], d_loss, p2p_loss))
    else:
        print ''
        print ('Train Losses of (D={0} / P2P={1} / P2P2P={2} / CODE_D={3});\n'
               'Validation Losses of (D={4} / P2P={5} / P2P2P={6} / CODE_D={7})'.format(
                    losses['d'][-1], losses['p2p'][-1], losses['p2p2p'][-1], losses['code_d'][-1],
                    d_loss, p2p_loss, p2p2p_loss, code_d_loss))

    return d_loss, p2p_loss, p2p2p_loss, code_d_loss


def model_creation(d, atob, params, atoa=None):
    """Create all the necessary models."""
    # Define the necessary models.
    opt = Adam(lr=params.lr, beta_1=params.beta_1)
    p2p = m.pix2pix(atob, d, params.a_ch, params.b_ch, alpha=params.alpha, opt=opt,
                    is_a_binary=params.is_a_binary, is_b_binary=params.is_b_binary)

    # When atoa is None only use the pix2pix model. Otherwise, use the pix2pix2pix also.
    code_d = None
    g = atob
    sampler = None
    p2p2p = None
    if atoa is not None:
        code_d = m.code_discriminator(params.latent_dim, params.nfd)
        g = m.conditional_generator(atoa, atob, params.a_ch)
        sampler = m.generator_from_conditional_generator(g, params.latent_dim)
        p2p2p = m.pix2pix2pix(atoa, atob, d, code_d, params.a_ch, params.b_ch, alpha=params.alpha,
                              beta=params.beta, is_a_binary=params.is_a_binary,
                              is_b_binary=params.is_b_binary,
                              opt=Adam(lr=params.lr, beta_1=params.beta_1))

    models = MyDict({
        'atoa': atoa,
        'atob': atob,
        'd': d,
        'code_d': code_d,
        'p2p': p2p,
        'g': g,
        'sampler': sampler,
        'p2p2p': p2p2p,
    })

    return models


def generators_creation(it_train, it_val, models, dout_size=(16, 16)):
    """Create all the necessary generators."""
    d_gen = discriminator_generator(it_train, models.g, dout_size=dout_size)
    d_gen_val = discriminator_generator(it_val, models.g, dout_size=dout_size)

    p2p_gen = generator_generator(it_train, dout_size=dout_size)
    p2p_gen_val = generator_generator(it_val, dout_size=dout_size)

    p2p2p_gen = generator_generator(it_train, dout_size=dout_size)
    p2p2p_gen_val = generator_generator(it_val, dout_size=dout_size)

    encoder = models.atoa.get_layer('vae_encoder')
    code_d_gen = code_discriminator_generator(it_train, encoder, dout_size=dout_size)
    code_d_gen_val = code_discriminator_generator(it_val, encoder, dout_size=dout_size)

    generators = MyDict({
        'd_gen': d_gen,
        'd_gen_val': d_gen_val,
        'p2p_gen': p2p_gen,
        'p2p_gen_val': p2p_gen_val,
        'p2p2p_gen': p2p2p_gen,
        'p2p2p_gen_val': p2p2p_gen_val,
        'code_d_gen': code_d_gen,
        'code_d_gen_val': code_d_gen_val,
    })

    return generators


def train_iteration(models, generators, losses, params):
    """Perform a train iteration."""
    # Get necessary generators
    d_gen = generators.d_gen
    p2p2p_gen = generators.p2p2p_gen
    code_d_gen = generators.code_d_gen

    # Get necessary models
    d = models.d
    code_d = models.code_d
    p2p2p = models.p2p2p

    # Train discriminator and atob models
    dhist = train_discriminator(d, d_gen, batch_size=params.batch_size)
    losses['d'].extend(dhist.history['loss'])

    # Train the code discriminator, atoa and atob models
    cdhist = train_discriminator(code_d, code_d_gen, batch_size=params.batch_size)
    p2p2phist = train_generator(p2p2p, p2p2p_gen, batch_size=params.batch_size)

    losses['code_d'].extend(cdhist.history['loss'])
    losses['p2p2p'].extend(p2p2phist.history['loss'])


def train(models, it_train, it_val, params):
    """
    Train the model.

    Parameters:
    - models: a dictionary with all the models.
        - atoa: a model that goes from A to A. Must be a VAE or None. When this
        is None it is the same as the Pix2Pix model.
        - atob: a model that goes from A to B.
        - d: the discriminator model.
        - p2p: a Pix2Pix model.
        - g: a conditional generator. Goes from A to B. When atoa is None this
        is the same as atob.
        - sampler: a model that samples new A/B pairs from a random vector.
        When atoa is None this is also None.
        - p2p2p: a Pix2Pix2Pix model. When atoa is None this is also None.
    - it_train: the iterator of the training data.
    - it_val: the iterator of the validation data.
    - params: parameters of the training procedure. Must define:
        - batch_size
        - train_samples
        - val_samples
        - epochs
        - save_every
        - continue_train
    - dout_size: the size of the output of the discriminator model.
    """
    # Create the experiment folder and save the parameters
    create_expt_dir(params)

    d = models.d
    atob = models.atob
    sampler = models.sampler

    # Get the output shape of the discriminator
    dout_size = d.output_shape[-2:]
    # Define the data generators
    generators = generators_creation(it_train, it_val, models, dout_size=dout_size)

    # Define the number of samples to use on each training epoch
    train_samples = params.train_samples
    if params.train_samples == -1:
        train_samples = it_train.N
    batches_per_epoch = train_samples // params.batch_size

    # Define the number of samples to use for validation
    val_samples = params.val_samples
    if val_samples == -1:
        val_samples = it_val.N

    losses = {'p2p': [], 'p2p2p': [], 'd': [], 'p2p_val': [], 'p2p2p_val': [],
              'd_val': [], 'code_d': [], 'code_d_val': []}
    if params.continue_train:
        losses = load_losses(log_dir=params.log_dir, expt_name=params.expt_name)

    for e in tqdm(range(params.epochs)):

        for b in range(batches_per_epoch):
            train_iteration(models, generators, losses, params)

        # Evaluate how the models is doing on the validation set.
        evaluate(models, generators, losses, val_samples=val_samples)

        if (e + 1) % params.save_every == 0:
            save_weights(models, log_dir=params.log_dir, expt_name=params.expt_name)
            log(losses, atob, params, it_val, sampler=sampler, latent_dim=params.latent_dim,
                log_dir=params.log_dir, expt_name=params.expt_name)

if __name__ == '__main__':
    
    global graph1
    global graph2
    # https://github.com/keras-team/keras/issues/2397
    
    a = sys.argv[1:]

    params = MyDict({
        'nfd': 32,  # Number of filters of the first layer of the discriminator
        'nfatoa': 32,  # Number of filters of the first layer of the AtoA model
        'nfatob': 32,  # Number of filters of the first layer of the AtoB model
        'a_ch': 1,  # Number of channels of images A
        'b_ch': 3,  # Number of channels of images B
        'is_a_binary': True,  # If A is binary, the last layer of the atoa model is followed by a sigmoid
        'is_b_binary': False,  # If B is binary, the last layer of the atob model is followed by a sigmoid
        'is_a_grayscale': True,  # If A is grayscale
        'is_b_grayscale': False,  # If B is grayscale
        'log_dir': 'log',  # Directory to log
        'base_dir': 'data/unet_segmentations_binary',  # Directory that contains the data
        'train_dir': 'train',  # Directory inside base_dir that contains training data
        'val_dir': 'val',  # Directory inside base_dir that contains validation data
        'load_to_memory': True,  # Whether to load the images into memory
        'epochs': 100,  # Number of epochs to train the model
        'batch_size': 20,  # The number of samples to train each model on each iteration
        'latent_dim': 16,  # The dimension of the latent space. Necessary when training the VAE
        'alpha': 100,  # The weight of the reconstruction loss of the atob model
        'beta': 100,  # The weight of the reconstruction loss of the atoa model
        'save_every': 10,  # Save results every 'save_every' epochs on the log folder
        'lr': 2e-4,  # The learning rate to train the models
        'beta_1': 0.5,  # The beta_1 value of the Adam optimizer
        'continue_train': False,  # If it should continue the training from the last checkpoint
        'train_samples': -1,  # The number of training samples. Set -1 to be the same as training examples
        'val_samples': -1,  # The number of validation samples. Set -1 to be the same as validation examples
        'rotation_range': 0.,  # The range to rotate training images for dataset augmentation
        'height_shift_range': 0.,  # Percentage of height of the image to translate for dataset augmentation
        'width_shift_range': 0.,  # Percentage of width of the image to translate for dataset augmentation
        'horizontal_flip': False,  # If true performs random horizontal flips on the train set
        'vertical_flip': False,  # If true performs random vertical flips on the train set
        'zoom_range': 0.,  # Defines the range to scale the image for dataset augmentation
        'pix2pix': False,  # If true only trains a pix2pix. Otherwise it trains the pix2pix2pix
        'expt_name': None,  # The name of the experiment. Saves the logs into a folder with this name
        'target_size': 256,  # The size of the images loaded by the iterator. DOES NOT CHANGE THE MODELS
    })

    param_names = [k + '=' for k in params.keys()] + ['help']

    try:
        opts, args = getopt.getopt(a, '', param_names)
    except getopt.GetoptError:
        print_help()
        sys.exit()

    for opt, arg in opts:
        if opt == '--help':
            print_help()
            sys.exit()
        elif opt in ('--nfatoa', '--nfatob' '--nfd', '--a_ch', '--b_ch',
                     '--epochs', '--batch_size', '--latent_dim', '--alpha',
                     '--save_every', '--train_samples', '--val_samples',
                     '--target_size'):
            params[opt[2:]] = int(arg)
        elif opt in ('--lr', '--beta_1', '--rotation_range', '--height_shift_range',
                     '--width_shift_range', '--zoom_range', '--alpha', '--beta'):
            params[opt[2:]] = float(arg)
        elif opt in ('--is_a_binary', '--is_b_binary', '--is_a_grayscale', '--is_b_grayscale',
                     '--continue_train', '--horizontal_flip', '--vertical_flip', '--pix2pix',
                     '--load_to_memory'):
            params[opt[2:]] = True if arg == 'True' else False
        elif opt in ('--base_dir', '--train_dir', '--val_dir', '--expt_name',
                     '--log_dir'):
            params[opt[2:]] = arg

    dopt = Adam(lr=params.lr, beta_1=params.beta_1)

    # Define the U-Net generator
    unet = m.g_unet(params.a_ch, params.b_ch, params.nfatob, is_binary=params.is_b_binary)
    vae = None
    if not params.pix2pix:
        vae = m.g_vae(params.a_ch, params.a_ch, params.nfatoa, params.latent_dim,
                      is_binary=params.is_a_binary)
    graph1 = K.get_session().graph

    # Define the discriminator
    d = m.discriminator(params.a_ch, params.b_ch, params.nfd, opt=dopt)
    graph2 = K.get_session().graph

    if params.continue_train:
        load_weights(vae, unet, d, log_dir=params.log_dir, expt_name=params.expt_name)

    ts = params.target_size
    train_dir = os.path.join(params.base_dir, params.train_dir)
    it_train = TwoImageIterator(train_dir,  is_a_binary=params.is_a_binary,
                                is_a_grayscale=params.is_a_grayscale,
                                is_b_grayscale=params.is_b_grayscale,
                                is_b_binary=params.is_b_binary, batch_size=1,
                                load_to_memory=params.load_to_memory,
                                rotation_range=params.rotation_range,
                                height_shift_range=params.height_shift_range,
                                width_shift_range=params.height_shift_range,
                                zoom_range=params.zoom_range,
                                horizontal_flip=params.horizontal_flip,
                                vertical_flip=params.vertical_flip,
                                target_size=(ts, ts))
    val_dir = os.path.join(params.base_dir, params.val_dir)
    it_val = TwoImageIterator(val_dir,  is_a_binary=params.is_a_binary,
                              is_b_binary=params.is_b_binary,
                              is_a_grayscale=params.is_a_grayscale,
                              is_b_grayscale=params.is_b_grayscale, batch_size=1,
                              load_to_memory=params.load_to_memory,
                              target_size=(ts, ts))

    models = model_creation(d, unet, params, atoa=vae)
    train(models, it_train, it_val, params)
