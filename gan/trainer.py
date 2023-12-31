import torch
import os
import tqdm
import numpy as np
import glob
from collections import defaultdict
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from .losses import wasserstein_distance, gradient_penalty


class Trainer:
    '''
        Trainer module which contains both the full training driver
        which calls the train_batch method
    '''

    gradient_weight = 25

    neptune_config = None

    def __init__(self, generator, discriminator, savefolder, device='cuda'):
        '''
            Store the generator and discriminator info
        '''

        generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        if savefolder[-1] != '/':
            savefolder += '/'

        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        self.start = 1

    def batch(self, x, train=False):
        '''
            Train the generator and discriminator on a single batch
        '''

        if not isinstance(x, torch.Tensor):
            input_tensor = torch.as_tensor(x, dtype=torch.float).to(self.device)
        else:
            input_tensor = x.to(self.device, non_blocking=True)

        z_samp = torch.randn((x.shape[0], self.generator.n_z), device=self.device)
        x_gen = self.generator(z_samp)

        disc_fake = self.discriminator(x_gen)
        gen_loss = -torch.mean(disc_fake)

        if train:
            self.generator.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

        disc_real = self.discriminator(input_tensor)

        disc_fake = self.discriminator(x_gen.detach())
        w_dist = wasserstein_distance(disc_real, disc_fake)
        d_regularizer = gradient_penalty(self.discriminator, input_tensor, x_gen) * self.gradient_weight

        disc_loss = w_dist + d_regularizer

        if train:
            self.discriminator.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

        keys = ['gen', 'disc', 'w_dist', 'grad_penalty']
        mean_loss_i = [gen_loss.item(), disc_loss.item(), w_dist.item(), d_regularizer.item()]

        loss = {key: val for key, val in zip(keys, mean_loss_i)}

        return loss

    def train(self, train_data, val_data, epochs, dsc_learning_rate=1.e-3,
              gen_learning_rate=1.e-3, save_freq=10, lr_decay=None, decay_freq=5):
        '''
            Training driver which loads the optimizer and calls the
            `train_batch` method. Also handles checkpoint saving
            Inputs
            ------
            train_data : DataLoader object
                Training data that is mapped using the DataLoader or
                MmapDataLoader object defined in io.py
            val_data : DataLoader object
                Validation data loaded in using the DataLoader or
                MmapDataLoader object
            epochs : int
                Number of epochs to run the model
            dsc_learning_rate : float [default: 1e-4]
                Initial learning rate for the discriminator
            gen_learning_rate : float [default: 1e-3]
                Initial learning rate for the generator
            save_freq : int [default: 10]
                Frequency at which to save checkpoints to the save folder
            lr_decay : float [default: None]
                Learning rate decay rate (ratio of new learning rate
                to previous). A value of 0.95, for example, would set the
                new LR to 95% of the previous value
            decay_freq : int [default: 5]
                Frequency at which to decay the learning rate. For example,
                a value of for decay_freq and 0.95 for lr_decay would decay
                the learning to 95% of the current value every 5 epochs.
            Outputs
            -------
            G_loss_plot : numpy.ndarray
                Generator loss history as a function of the epochs
            D_loss_plot : numpy.ndarray
                Discriminator loss history as a function of the epochs
        '''

        if (lr_decay is not None):
            gen_lr = gen_learning_rate * (lr_decay)**((self.start - 1) / (decay_freq))
            dsc_lr = dsc_learning_rate * (lr_decay)**((self.start - 1) / (decay_freq))
        else:
            gen_lr = gen_learning_rate
            dsc_lr = dsc_learning_rate

        if self.neptune_config is not None:
            self.neptune_config['model/parameters/gen_learning_rate'] = gen_lr
            self.neptune_config['model/parameters/dsc_learning_rate'] = dsc_lr
            self.neptune_config['model/parameters/start'] = self.start
            self.neptune_config['model/parameters/n_epochs'] = epochs

        # create the Adam optimzers
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), lr=gen_lr)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=dsc_lr)

        # set up the learning rate scheduler with exponential lr decay
        if lr_decay is not None:
            gen_scheduler = ExponentialLR(self.gen_optimizer, gamma=lr_decay)
            dsc_scheduler = ExponentialLR(self.disc_optimizer, gamma=lr_decay)
            if self.neptune_config is not None:
                self.neptune_config['model/parameters/scheduler'] = 'ExponentialLR'
                self.neptune_config['model/parameters/decay_freq'] = decay_freq
                self.neptune_config['model/parameters/lr_decay'] = lr_decay
        else:
            gen_scheduler = None
            dsc_scheduler = None

        # empty lists for storing epoch loss data
        D_loss_ep, G_loss_ep = [], []
        for epoch in range(self.start, epochs + 1):
            if isinstance(gen_scheduler, ExponentialLR):
                gen_lr = gen_scheduler.get_last_lr()[0]
                dsc_lr = dsc_scheduler.get_last_lr()[0]
            else:
                gen_lr = gen_learning_rate
                dsc_lr = dsc_learning_rate

            print(f"Epoch {epoch} -- lr: {gen_lr:5.3e}, {dsc_lr:5.3e}")
            print("-------------------------------------------------------")

            # batch loss data
            pbar = tqdm.tqdm(train_data, desc='Training: ', dynamic_ncols=True)

            if hasattr(train_data, 'shuffle'):
                train_data.shuffle()

            # set to training mode
            self.generator.train()
            self.discriminator.train()

            losses = defaultdict(list)
            # loop through the training data
            for i, input_img in enumerate(pbar):
                if isinstance(input_img, (list, tuple)):
                    input_img = input_img[0]

                # train on this batch
                batch_loss = self.batch(input_img, train=True)

                # append the current batch loss
                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key], axis=0)

                loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])

                pbar.set_postfix_str(loss_str)

            # update the epoch loss
            D_loss_ep.append(loss_mean['disc'])
            G_loss_ep.append(loss_mean['gen'])

            if self.neptune_config is not None:
                self.neptune_config['train/gen_loss'].append(loss_mean['gen'])
                self.neptune_config['train/disc_loss'].append(loss_mean['disc'])

            # validate every `validation_freq` epochs
            self.discriminator.eval()
            self.generator.eval()
            pbar = tqdm.tqdm(val_data, desc='Validation: ')

            if hasattr(val_data, 'shuffle'):
                val_data.shuffle()

            losses = defaultdict(list)
            # loop through the training data
            for i, input_img in enumerate(pbar):
                if isinstance(input_img, (list, tuple)):
                    input_img = input_img[0]

                # train on this batch
                batch_loss = self.batch(input_img, train=False)

                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key], axis=0)

                loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])

                pbar.set_postfix_str(loss_str)

            if self.neptune_config is not None:
                self.neptune_config['eval/gen_loss'].append(loss_mean['gen'])
                self.neptune_config['eval/disc_loss'].append(loss_mean['disc'])

            # apply learning rate decay
            if (gen_scheduler is not None) & (dsc_scheduler is not None):
                if isinstance(gen_scheduler, ExponentialLR):
                    if epoch % decay_freq == 0:
                        gen_scheduler.step()
                        dsc_scheduler.step()

            # save checkpoints
            if epoch % save_freq == 0:
                self.save(epoch)

        return G_loss_ep, D_loss_ep

    def save(self, epoch):
        gen_savefile = f'{self.savefolder}/generator_ep_{epoch:03d}.pth'
        disc_savefile = f'{self.savefolder}/discriminator_ep_{epoch:03d}.pth'

        print(f"Saving to {gen_savefile} and {disc_savefile}")
        torch.save(self.generator.state_dict(), gen_savefile)
        torch.save(self.discriminator.state_dict(), disc_savefile)

    def load_last_checkpoint(self):
        gen_checkpoints = sorted(
            glob.glob(self.savefolder + "generator_ep*.pth"))
        disc_checkpoints = sorted(
            glob.glob(self.savefolder + "discriminator_ep*.pth"))

        gen_epochs = set([int(ch.split(
            '/')[-1].replace('generator_ep_', '')[:-4]) for
            ch in gen_checkpoints])
        dsc_epochs = set([int(ch.split(
            '/')[-1].replace('discriminator_ep_', '')[:-4]) for
            ch in disc_checkpoints])

        try:
            assert len(gen_epochs) > 0, "No checkpoints found!"

            start = max(gen_epochs.union(dsc_epochs))
            self.load(f"{self.savefolder}/generator_ep_{start:03d}.pth",
                      f"{self.savefolder}/discriminator_ep_{start:03d}.pth")
            self.start = start + 1
        except Exception as e:
            print(e)
            print("Checkpoints not loaded")

    def load(self, generator_save, discriminator_save):
        print(generator_save, discriminator_save)
        self.generator.load_state_dict(torch.load(generator_save))
        self.discriminator.load_state_dict(torch.load(discriminator_save))

        gfname = generator_save.split('/')[-1]
        dfname = discriminator_save.split('/')[-1]
        print(
            f"Loaded checkpoints from {gfname} and {dfname}")


def weights_init(net):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('InstanceNorm') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data, 1.0)
            torch.nn.init.constant_(m.bias.data, 0.0)
