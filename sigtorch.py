# Author: Kyle W. McClintick <kyle.mcclintick@ll.mit.edu>.

"""pytorch tools and common tasks library

This module is a library that:

    - loads, transforms, and parses datasets using dataset and dataloader objects
    - builds common pytorch models
    - trains those models
    - predicts with these models and evaluates their score

The following is a simple usage example. This library is designed for 3-line usage that initializes a model, trains it, and tests it, with evaluation built-in to testing if y is given::

    import sigtorch as st
    deep_SVDD = st.OneClass()
    deep_SVDD.train(x, y, n_epochs = 50, task_path='market_forcasting_anomalys', model_name='SVDD_50epochs')
    predictions, metrics = deep_SVDD.test(x, y)

The module contains the following public classes:

    - PytorchDataset -- A dataset object that can function for unsupervisded data (x only), transform data with signal transforms (e.g., fft) or with random noise for regularization/augmentation
    - ModelHandler -- All nn.Module objects need some common helper functions, this class handles it. Each child class saves its nn.Module object as self.model
    - CNN -- a set of resnet state-of-the-art algorithms for regression and multi-class classification
    - OneClass -- a state-of-the-art algorithm for one-class classification

All other classes in this module are considered implementation details.
"""

__version__ = '1.0'

import math
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchaudio.transforms as T
import logging
import logging.config
import numpy as np
from alive_progress import alive_bar
from sklearn.metrics import roc_auc_score, f1_score
from torchvision.transforms import Compose, ToTensor, Lambda
import torchaudio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGING_CONFIG = {
     'version': 1,
     'disable_existing_loggers': True,
     'formatters': {
         'standard': {
             'format': '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s'
         },
     },
     'handlers': {
         'default': {
             'level': 'DEBUG',
             'formatter': 'standard',
             'class': 'logging.StreamHandler',
             'stream': 'ext://sys.stderr',  # Default is stderr
         },
     },
     'loggers': {
         '': {  # root logger
             'handlers': ['default'],
             'level': 'DEBUG',
             'propagate': False
         },
     }
 }
logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

class PytorchDataset(Dataset):
    """
    This generalized format allows for transforms and unsupervised data
    """
    def __init__(self,x,y=None, return_idx=False, transform = None):
        '''
        @param x <tensor>: input data to model
        @param y <tensor>: output data to model (optional)
        @param transform <func>: transforms on x, e.g. to cast a complex tensor as stacked real/imag: Compose([torch.from_numpy,Lambda(lambda t: torch.stack((t.real, t.imag), axis = 0))])
        @param return_idx <bool>: will add another return value if true, the index. This is needed for certain metrics (e.g., AUC)
        '''
        self.x = x
        self.y = y
        self.transform = transform
        self.return_idx = return_idx

    def __len__(self):
        '''
        @return <int>: samples of input data
        '''
        return len(self.x)

    def __getitem__(self, idx):
        '''
        @param idx <int>: index of dataset to grab
        @return <tensor>: input data, output data sample, index of sample
        '''
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        if not self.y is None:
            if not self.return_idx:
                return x.to(DEVICE), self.y[idx].to(DEVICE)
            else:
                return x.to(DEVICE), self.y[idx].to(DEVICE), idx
        else:
            if not self.return_idx:
                return x.to(DEVICE)
            else:
                return x.to(DEVICE), idx

    def get_transformed_shape(self):
        '''
        Returns the size() of x with all transforms applied
        '''
        x = self.x
        if self.transform:
            x = self.transform(x)
        return x.size()

def standardize():
    '''
    Applies a zero mean, unit standard deviation transform to data
    '''
    return Compose([Lambda(lambda t: t - t.mean(dim=1, keepdim=True)), Lambda(lambda t: t / t.std(dim=1, keepdim=True))])

def fft(n = 512):
    '''
    Applies a fast-fourier transform to achieve a constant-size frequency domain represtation of time series

    @param <int> n: size of FFT
    '''
    return Lambda(lambda t: torch.abs(torch.fft.fft(t, dim=-1, n=n)))

def spectrogram(n_fft = 255, win_length = 128):
    '''
    Applies a spectrogram transform to acheive a constant-size 2D time-frequency representation of time series

    @param n_fft <int>: Size of FFT, controls the number of frequency bins on the returned spectrogram
    @param win_length <int>: window size, controls the number of time bins on the returned spectrogram
    '''
    return torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length)

def IQ():
    '''
    Stacks the in-phase (real) and quadrature (imag) components of a complex-valued time series
    '''
    return Lambda(lambda t: torch.stack((t.real, t.imag), axis = 1).squeeze())

def MP():
    '''
    Stacks the magnitude and phase components of a complex-valued time series
    '''
    return Lambda(lambda t: torch.stack((torch.abs(t), torch.angle(t)), axis = 1).squeeze())

def AWGN(sigma = 0.1):
    '''
    Applies additive white gaussian noise to a signal, which has been shown to be equivalent to l2 regularization

    @param sigma <float>: variance of Gaussian distribution, higher will make more noise
    '''
    return Lambda(lambda t: t + sigma * torch.randn(t.size(), dtype=t.dtype))

def rotatePhase():
    '''
    Applies a random phase rotation to a complex-valued signal
    '''
    return Lambda(lambda t: t * torch.exp(1j*torch.rand(t.size(0))*2*np.pi))

def resample(sample_rate = 2, low = 1, high = 4):
    '''
    Changes the sample rate of the signal from sample_rate to a frequency uniform random in [low, high], good to use if you don't know the sample rate of test-stage data

    @param sample_rate <int>: input data's sample rate in Hz
    @Param low <int>: inclusive lower bound for resampled frequency
    @Param high <int>: inclusive upper bound for resampled frequency
    '''
    resample_rate = torch.randint(low, high+1, (1,))
    return T.Resample(sample_rate, resample_rate)

def fgsm_attack(input, epsilon_low=0.0, epsilon_high=0.1):
    '''
    Creates a perturbed input for adversarial training. As suggested by the state-of-the-art, epsilon is randomized to avoid overfitting

    @param <tensor> input: the signal being perturbed
    @param epsilon_low <float>: perturbation lower limit, inclusive. Must be greater than zero
    @param epislon_high <float>: perturbation upper limit, inclusive. Must be greater than zero and than low
    '''
    epsilon = (epsilon_low - epsilon_high) * torch.rand(1) + epsilon_high
    data_grad = input.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_input = input + epsilon.to(DEVICE) * sign_data_grad
    perturbed_input = torch.clamp(perturbed_input, torch.min(input), torch.max(input))
    return perturbed_input

def pgd_attack(input, labels, model, epsilon_low=0.01, epsilon_high=0.1, alpha_low=0.0, alpha_high=0.01, iters_low=10, iters_high=40, criterion = nn.CrossEntropyLoss()):
    '''
    Creates a perturbed input for adversarial training. Stronger than FGSM but more computational cost to create. As suggested by the state-of-the-art, epsilon, iters, and alpha are randomized to avoid overfitting

    @param model <nn.Module>: the model must be passed to compute iterative forward passes
    @param input <tensor>: the input being attacked
    @param epsilon_low <float>: perturbation lower limit, inclusive. Must be greater than zero
    @param epislon_high <float>: perturbation upper limit, inclusive. Must be greater than zero and than low
    @param alpha_low <float>: step size lower limit, inclusive. Must be greater than zero
    @param alpha_high <float>: step size upper limit, inclusive. Must be greater than zero and than low
    @param iters_low <int>: number of steps lower limit, inclusive. Must be greater than zero
    @param iters_high <int>: number of steps upper limit, inclusive. Must be greater than zero and than low
    @param criterion <nn>: pytorch loss function object
    '''
    epsilon = (epsilon_low - epsilon_high) * torch.rand(1) + epsilon_high
    epsilon = epsilon.to(DEVICE)
    alpha = (alpha_low - alpha_high) * torch.rand(1) + alpha_high
    alpha = alpha.to(DEVICE)
    iters = torch.randint(iters_low, iters_high+1, (1,))

    ori_input = input.data
    for i in range(iters) :
        input.requires_grad = True
        outputs = model(input)
        model.zero_grad()
        cost = criterion(outputs, labels).to(DEVICE)
        cost.backward()
        adv_input = input + alpha*input.grad.sign()
        eta = torch.clamp(adv_input - ori_input, min=-epsilon, max=epsilon)
        input = torch.clamp(ori_input + eta, min=torch.min(ori_input), max=torch.max(ori_input)).detach_()
    return input

class ModelHandler(object):
    """
    Parent class that all models will use
    """
    def __init__(self):
        '''
        Common architecture initialization, messages, saving and loading
        '''
        logger.info(f"SigTorch v{__version__}")
        logger.info("""

                  .:.
                .:++.                   ███████╗██╗ ██████╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗
              .:+++=.   .               ██╔════╝██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║
            ..=+++.   -++-              ███████╗██║██║  ███╗  ██║   ██║   ██║██████╔╝██║     ███████║
           .-+++:.    ..:.              ╚════██║██║██║   ██║  ██║   ██║   ██║██╔══██╗██║     ██╔══██║
          :+++-.          .=+:.         ███████║██║╚██████╔╝  ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║
        .:+++.            .=++-.        ╚══════╝╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
        :+++.              .+++:
       .-++.                .++=.
       .-++.                .++=.                           ......                             .
       .-++-                :++-.                          ..     ..            ..: .       ..  ...
        .+++.              .+++.              .           .         . ... .. .:..     .   .        ..
         .+++-            -+++.         :.   ....      ....          .                  ..          .
          .=+++-..    ..-+++=.           .. ...   ..... .
           ..++++++++++++++..              .         ..
              ..-======-..

        """)
        logger.info(f"Torch device is {DEVICE}")
        self.model = None

    def save_if_best(self, model_name, task_dir, loss):
        '''
        Saves pytorch model to .pt file if its loss (training or validation) is lower than the .json (if it exists) of its task

        @param model_name <string>: name of the model, user can decide what's important to include here regarding hyperparameters (e.g., resnet18_lr001 vs resnet18_lr01)
        @param task_dir <string>: directory string of the task we're trying to find the optimal model for
        @param loss <float>: loss we're comparing to what's in the .json file
        '''
        metadata_file = f"{task_dir}/ckpt.json"
        chkpt_file    = f"{task_dir}/ckpt.pt"
        logger.debug("Converting loss to a float if it isn't already")
        try:
            loss = loss.cpu().numpy().tolist()
        except AttributeError:
            pass
        logger.debug("Creating the task directory if it doesn't exist already")
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        logger.debug("Creating the checkpoint and its json metadatafile if they don't exist already")
        if not os.path.exists(metadata_file) or not os.path.exists(chkpt_file):
            torch.save(self.model.state_dict(), chkpt_file)
            logger.info(f"Checkpoint doesn't exist, creating {metadata_file} and {chkpt_file}")
            dictionary = {
                "model_name": model_name,
                "loss": loss
            }
            with open(metadata_file, "w") as outfile:
                json.dump(dictionary, outfile)
        else:
            logger.debug("Open the existing metadata file to see if the current checkpoint has a lower loss")
            with open(metadata_file, 'r') as openfile:
                json_object = json.load(openfile)
                if json_object['loss'] > loss:
                    logger.debug("Overwrite existing checkpoint and metadata file with model name of the best model")
                    logger.info(f"Best checkpoint yet, saving with name {model_name} to {metadata_file} and {chkpt_file}")
                    json_object["loss"] = loss
                    json_object["model_name"] = model_name
                    with open(metadata_file, "w") as jsonFile:
                        json.dump(json_object, jsonFile)
                    torch.save(self.model.state_dict(), chkpt_file)

    def load(self, path):
        '''
        Loads pytorch model from .pt file, and enables eval mode for testing

        @param path <string>: .pt file
        '''
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class CIFAR10_LeNet_ELU(nn.Module):
    """
    State of the art one-class architecture
    """
    def __init__(self, num_channel_in=3, onedim=True, scale=[32,32]):
        '''
        @param num_channel_in <int>: the number of data channels (e.g., 3 for RGB images)
        @param onedim <bool>: if data is 1D or 2D (i.e., time series vs images)
        @param W <int>: data sample width (i.e., pixels, signal samples), must be base 2
        '''
        super(CIFAR10_LeNet_ELU, self).__init__()
        self.rep_dim = 128
        if onedim:
            scale = int(max(scale) / 32)
            self.pool = nn.MaxPool1d(2, 2)
            self.conv1 = nn.Conv1d(num_channel_in, 32, 5, bias=False, padding=2)
            self.bn2d1 = nn.BatchNorm1d(32, eps=1e-04, affine=False)
            self.conv2 = nn.Conv1d(32, 64, 5, bias=False, padding=2)
            self.bn2d2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
            self.conv3 = nn.Conv1d(64, 128, 5, bias=False, padding=2)
            self.bn2d3 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
            self.fc1 = nn.Linear(128 * 4 * scale, self.rep_dim, bias=False)
        else:
            scale = int((scale[0] / 32) * (scale[1] / 32))
            self.pool = nn.MaxPool2d(2, 2)
            self.conv1 = nn.Conv2d(num_channel_in, 32, 5, bias=False, padding=2)
            self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
            self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
            self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
            self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
            self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
            self.fc1 = nn.Linear(128 * 4 * 4 * scale, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class OneClass(ModelHandler):
    """
    Framework for one-class tasks (i.e., anomaly detection)
    """
    def __init__(self):
        '''
        Inits DeepSVDD
        '''
        super(OneClass, self).__init__()
        logger.info("""

         ██████╗ ███╗   ██╗███████╗     ██████╗██╗      █████╗ ███████╗███████╗
        ██╔═══██╗████╗  ██║██╔════╝    ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝
        ██║   ██║██╔██╗ ██║█████╗█████╗██║     ██║     ███████║███████╗███████╗
        ██║   ██║██║╚██╗██║██╔══╝╚════╝██║     ██║     ██╔══██║╚════██║╚════██║
        ╚██████╔╝██║ ╚████║███████╗    ╚██████╗███████╗██║  ██║███████║███████║
         ╚═════╝ ╚═╝  ╚═══╝╚══════╝     ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝
                                        :#%+         +%%:                             XXXX
                XXXX          .=%-                           .#*.                     XXXX
                XXXX        ..                      ...            :.                 XXXX
                XXXX      :#-                    .=*-=*.          .*-
                        ...                       .#.  +-             .-.
                       .=+.         .=*+:          .+**-               :#.
                        .:++=.     .#: .+=                                .
                     .#.:+. .*     .=+:-*....                     ....   .+=.
                    .-. .*-.++       ...:*:.=*.                 .*-.:*:   .:.
                    .     ...           -+  :#.:#::#:           .#. .*-     :.
                   -+                    .=+:  +.  .+     :**:   .:+=.      :%.
                             .=##:.            .+##+.    *-  =+.    ..::..
                  ..         *.  --                      -+..*-     :+. -+.:--*.
                  +:         =+:-#.         ...                     :+..-+#. .*:
                                           =*+*=                     .-=..*:..*:
                 .-.             .:***:   .*  .*    -#**:.                 .-.*-
        XXXX     .*:             .*  .+    -*+*-    +.  #.  :=+-.            .*:
        XXXX                     .+*=++             ++=*=. -=. .+
        XXXX      .+               ...               ...   :*:.+=            .+.  XXXX
                  .#:                           :*::*-  **+#....             +-   XXXX
                                  .*#*.    =*-*=*-  -*.==. .#.                    XXXX
                    +=           .*  .+   .*   * -##:  .+#*#.               #.
                    .=.          .++.=+    -#*#-                    .+##*. ..
                      ..                        ...                 =-  .*:.               XXXX
                      .+=.                    .*=:*-.               .*=-*#:                XXXX
                                              =+  .#.                                      XXXX
                         .*+.                  -**+.                 :#-.
                             .+#:                        ...    .+#:              XXXX
                               .. :-                   :*..=+.=. ..               XXXX
                                  ..:. ..              =+-.:#..                   XXXX
                                       ...  :%%+  #%#. ..=+:.
        """)
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

    def train(self, x, y, lr: float = 0.001, n_epochs: int = 50, lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, n_jobs_dataloader: int = 0, task_path=None, model_name=None, transform=None, adv_train=None):
        '''
        Trains the Deep SVDD model on the training data.

        @param x <tensor>: training data inputs, must be of shape (Number of samples, Channels, Height of sample, Width of sample)
        @param y <tensor>: trainin data outputs, must of of shape (Number of samples), values must be integers 0 (anomaly) or 1 (in-class)
        @param n_epochs <int>: number of epochs to train for before giving up
        @param lr_milestones <int>: set of epoch milestones to increment learning rate
        @param batch_size <int>: the number of samples to include in each mini-batch
        @param lr <float>: the learning rate to use
        @param model_name <string>: what to call this training session
        @param task_path <string>: what to call the directory of this learning task
        @param adv_train <string>: if None, train as normal, 'fgsm' or 'pgd' or adversarial training
        '''
        N, Channels, self.W, self.H = x.size()
        logger.debug("Initializing model based on data dimensions")
        if not np.log2(self.W).is_integer() or not np.log2(self.H).is_integer():
            raise ValueError("Sample width (dim 2) and height (dim 3) must both be powers of 2")
        if not model_name or not task_path:
            raise ValueError("must select a unique string for model_name and task_path or you will overwrite checkpoints you don't want to")
        logger.info(f"Training on inputs of shape: {N} samples, {Channels} channels/sample, {self.W} sample width, {self.H} sample height")
        logger.info(f"Training on outputs of shape: {N} samples in range, where y=1 are anomalies and y=0 are in-class")
        if self.W==1 or self.H==1:
            logger.info("1D mode enabled")
            self.onedim = True
            if self.W==1:
                x = x[:,:,0,:]
            else:
                x = x[:,:,:,0]
        train_set = PytorchDataset(x, y, transform=transform)
        self.transform = transform
        if transform:
            try:
                N, Channels, self.W, self.H = train_set.get_transformed_shape()
            except ValueError:
                if self.W==1:
                    N, Channels, self.H = train_set.get_transformed_shape()
                else:
                    N, Channels, self.W = train_set.get_transformed_shape()
            logger.info(f"Transformed inputs shape: {N} samples, {Channels} channels/sample, {self.W} sample width, {self.H} sample height")
            if self.W != 1 and self.H != 1:
                logger.info("1D mode disabled")
                self.onedim = False
        if adv_train == 'fgsm':
            logger.info("FGSM adversarial training enabled")
        elif adv_train == 'pgd':
            logger.info("PGD adversarial training enabled")
        self.model = CIFAR10_LeNet_ELU(num_channel_in=Channels, scale=[self.W,self.H], onedim=self.onedim).to(DEVICE)
        logger.debug("Defining variables")
        self.lr = lr
        self.n_epochs = n_epochs+1
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.n_jobs_dataloader = n_jobs_dataloader
        self.R = torch.tensor(self.R, device=DEVICE)
        self.c = torch.tensor(self.c, device=DEVICE) if self.c is not None else None
        logger.debug("Init train data loader")
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs_dataloader)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        logger.debug("Initialize hypersphere center c (if c not loaded)")
        if self.c is None:
            n_samples = 0
            eps = 0.1
            self.c = torch.zeros(self.model.rep_dim, device=DEVICE)
            self.model.eval()
            with torch.no_grad():
                for data in train_loader:
                    inputs, _ = data
                    outputs = self.model(inputs)
                    n_samples += outputs.shape[0]
                    self.c += torch.sum(outputs, dim=0)
            self.c /= n_samples
            logger.debug("If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.")
            self.c[(abs(self.c) < eps) & (self.c < 0)] = -eps
            self.c[(abs(self.c) < eps) & (self.c > 0)] = eps

        logger.debug("Training loop")
        old_loss_epoch = np.infty
        for epoch in range(self.n_epochs):
          with alive_bar(len(train_loader), theme='classic') as bar:
            bar.title('Training, Epoch: [{0:d} / {1:d}]'.format(epoch, self.n_epochs))
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            loss_epoch = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(DEVICE)
                if adv_train:
                    optimizer.zero_grad()
                    inputs.requires_grad = True
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    self.model.zero_grad()
                    loss.backward()
                    data_grad = inputs.grad.data
                    if adv_train == 'fgsm':
                        perturbed_data = fgsm_attack(inputs)
                    elif adv_train == 'pgd':
                        perturbed_data = pgd_attack(inputs, labels, self.model, criterion=self.criterion)
                    else:
                        raise ValueError("adv_train must be None, fgsm, or pgd")
                    outputs = self.model(perturbed_data)
                else:
                    outputs = self.model(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                loss = torch.mean(dist)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_batches += 1
                bar.title('Training, Epoch: [{0:d} / {1:d}], Quadratic Loss={2:3.3f}'.format(epoch, self.n_epochs, loss_epoch / n_batches))
                bar()
            scheduler.step()
            if old_loss_epoch < loss_epoch:
                logger.warning("Loss is increasing over previous epoch!")
            elif old_loss_epoch - loss_epoch < old_loss_epoch * 0.001:
                logger.warning("Loss decreased by less than 0.1% of previous epoch!")
            old_loss_epoch = loss_epoch
            self.save_if_best(model_name, task_path, loss_epoch / (n_batches))

    def test(self, x, y=None):
        '''
        Tests the given input data, prediciting if each sample in x is an anomoly or not, given the trained model. If ground truth is given, AUC metric is computed and returned

        @param x <tensor>: training data inputs, must be of shape (Number of samples, Channels, Height of sample, Width of sample)
        @param y <tensor>: trainin data outputs, must of of shape (Number of samples), values must be integers 0 (anomaly) or 1 (in-class)
        '''
        logger.debug("Squeezes out sample width or height if equal to 1")
        if self.onedim:
            if self.W==1:
                x = x[:,:,0,:]
            else:
                x = x[:,:,:,0]
        logger.debug("Init data loaders")
        test_set = PytorchDataset(x, y, return_idx=True, transform=self.transform)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        self.model.eval()
        logger.debug("Main test loop")
        with torch.no_grad():
            for data in test_loader:
                if y is not None:
                    inputs, labels, idx = data
                else:
                    inputs, idx = data
                inputs = inputs.to(DEVICE)
                outputs = self.model(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                scores = dist
                if y is not None:
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                else:
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)
        logger.debug("Compute metric if labels given, otherwise just return predictions")
        if y is not None:
            _, labels, scores = zip(*idx_label_score)
            labels = np.array(labels)
            scores = np.array(scores)
            self.test_auc = roc_auc_score(labels, scores)
            logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
        else:
            _, scores = zip(*idx_label_score)
            scores = np.array(scores)
            self.test_auc = None
        logger.info('Finished testing.')
        return scores, self.test_auc


class CNN(ModelHandler):
    """
    Framework for convolutional neural networks used for regression or multi-class classification tasks
    """
    def __init__(self):
        '''
        A framework for pytorch convolutional neural networks (cnns) with a focus on 1D data, generalization, hyperparameter search, and optimization
        '''
        super(CNN, self).__init__()
        logger.info("""

         ██████╗███╗   ██╗███╗   ██╗
        ██╔════╝████╗  ██║████╗  ██║
        ██║     ██╔██╗ ██║██╔██╗ ██║
        ██║     ██║╚██╗██║██║╚██╗██║
        ╚██████╗██║ ╚████║██║ ╚████║
         ╚═════╝╚═╝  ╚═══╝╚═╝  ╚═══╝
                                                  -*+**************+:..                               #+++++++++=-.       ..............                 :.:.
                                                  -*+-................:                               #=**********=:.     :*+++++++-                     :-=.    HORSE
                                                  -*+-.+****************+         :*************.     #=*=+********#=:    :**=*******-.                  :.:.
        .=*-+:            .-.                    -*+-.=+*===============--       :**..............   #=*=+#**********-.  :**==*++++++=:.                :-=.
        :==#.:+.     .=*--+..=.    ....          -*+-.=++.:===============--.    :**.*++++++++++++*..#=*=+#***********+:.:**==*=*******+..
        .=. --.+.    .=..--  .+..++....:*..      -*+-.=++.:*+*+++++++++++++*:..  :**.**-............:.-*=+#***-++++++++++.:-==*=+=#======--.             .       DOG
          .=+.  .=*-.#.   .*-*=.       . .=.     -*+-.=++.:**=................+  :**.**-:*+***********-.:+#***-*=========== .=*=++#=+++++++=:.           .
             -..     -:.   ..+:..    .:#..-=:..  -*+-.=++.:**=.+++++++++++++++++*:**.**-:*#.::::::::::::::+***-*=*=----------..:++#=-*+-------:          .
              -..    .==.     .:+         :++:   -*+-.=++.:**=.++*:::::::::::::::::*.**-:*#.=+***********#::+*-*=***----------:..+#=-*++--------:.       .
              .--.    ..+----=-.         .=.       .:.=++.:**=.++*................:-.**-:*#.=*=..............-:*=***=+=++++++++=:.:=-*++=*=======+:      .       CAT
               .+.                      =:            =++.:**=.++*................: .**-:*#.=*=....:::::....  .+=***=+-**********-. -*++=*+=======++.   .:..
                ..=:..                .+.             ..:.:**=.++*.........=--=::.: ...::*#.=*=...:=--:..::.   .:***=+-***########*-..-+=*+=++......::  :-=.
                    .:=-.           ..+:       ......   .::**=.++*.........=--=...:.....:*#.=*=....:::::::........+*=+-**+:..----:.:.   -*+=++.......:  :.:.    ------
                     ...::++++*++-:...         ............++=.++*................:.......=-+*+:::..................=*-**+:..=--=..:......+=++.......:  :-=.====|BIRD|
                       .-+=...=.                           ..=:***:...............:        .=*=.............        ..=#*+:........::.......*+:-:....:  ...     ------
                       ..+..=#.
                             ..
        """)

    def train(self, x, y, val_split=None, epochs=10, batch_size=16, opt='adam', lr=0.001, validation_period=5, model_name=None, task_path=None, task='classification', depth=18, transform=None, adv_train=None):
        '''
        @param x <tensor>: training data inputs, must be of shape (Number of samples, Channels, Height of sample, Width of sample)
        @param y <tensor>: trainin data outputs, must of of shape (Number of samples) with values in range [0, n_classes] for classification tasks, or of shape (Number of samples, number of regression outputs) for regression tasks
        @param val_split <float>: percentage in range [0.0, 1.0] of training data that should be parsed for validation usage
        @param epochs <int>: number of epochs to train for before giving up
        @param batch_size <int>: the number of samples to include in each mini-batch
        @param opt <string>: the optimizer to use. I generally find only sgd or adam are useful for CNNs
        @param lr <float>: the learning rate to use
        @param validation_period <int>: how often youd like to validate data, in epochs
        @param model_name <string>: what to call this training session
        @param task_path <string>: what to call the directory of this learning task
        @param depth <int>: the depth of the resnet model in number of layers
        @param task <string>: the purpose of the CNN, either to classify or regress
        @param adv_train <string>: if None, train as normal, 'fgsm' or 'pgd' or adversarial training
        '''
        N, Channels, self.W, self.H = x.size()
        logger.info(f"Training on inputs of shape: {N} samples, {Channels} channels/sample, {self.W} sample width, {self.H} sample height")
        logger.info(f"Training on outputs of shape: {y.size()}")
        logger.debug("Select a downloadable model (not pretrained) given depth options")
        if depth == 18:
            self.model = models.resnet18()
        elif depth == 34:
            self.model = models.resnet34()
        elif depth == 50:
            self.model = models.resnet50()
        elif depth == 101:
            self.model = models.resnet101()
        elif depth == 152:
            self.model = models.resnet152()
        else:
            raise ValueError("model depth must be an integer: 18, 34, 50, 101, or 152")
        logger.debug("Initialize network model based on dataset dimensions and task")
        num_channel_in = x.size(1)
        if self.W==1 or self.H==1:
            logger.info("1D mode enabled")
            self.onedim = True
            if self.W==1:
                x = x[:,:,0,:]
            else:
                x = x[:,:,:,0]
        if adv_train == 'fgsm':
            logger.info("FGSM adversarial training enabled")
        elif adv_train == 'pgd':
            logger.info("PGD adversarial training enabled")
        train_set = PytorchDataset(x, y, transform=transform)
        self.transform = transform
        if transform:
            try:
                N, Channels, self.W, self.H = train_set.get_transformed_shape()
            except ValueError:
                if self.W==1:
                    N, Channels, self.H = train_set.get_transformed_shape()
                else:
                    N, Channels, self.W = train_set.get_transformed_shape()
            num_channel_in = Channels
            logger.info(f"Transformed inputs shape: {N} samples, {Channels} channels/sample, {self.W} sample width, {self.H} sample height")


        self.model.conv1 = nn.Conv2d(num_channel_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.task = task
        if self.task == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            num_ftrs = self.model.fc.in_features
            n_regression_outputs = y.size(1)
            self.model.fc = nn.Linear(num_ftrs, n_regression_outputs)
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("Task selection must be string regression or classification")
        self.model.to(DEVICE)
        if not model_name or not task_path:
            raise ValueError("must select a unique string for model_name and task_path or you will overwrite checkpoints you don't want to")
        logger.debug("Splitting dataset into training and validation sets, if desired. Validation brings better generalization as long as it has enough samples")
        if val_split:
            train_set, val_set, = torch.utils.data.random_split(train_set, [1-val_split, val_split])
            validationloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        logger.debug("SGD can be more stable but adam can be faster and deeper learning")
        if opt == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif opt == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer selection must be string sgd or adam")

        logger.debug("Main training loop")
        old_loss = np.infty
        old_vloss = np.infty
        for epoch in range(epochs+1):
            epoch_loss = 0
            with alive_bar(len(trainloader), theme='classic') as bar:
                bar.title('Training, Epoch: [{0:d} / {1:d}]'.format(epoch, epochs))
                for i, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    if adv_train:
                        optimizer.zero_grad()
                        inputs.requires_grad = True
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        self.model.zero_grad()
                        loss.backward()
                        data_grad = inputs.grad.data
                        if adv_train == 'fgsm':
                             perturbed_data = fgsm_attack(inputs)
                        elif adv_train == 'pgd':
                             perturbed_data = pgd_attack(inputs, labels, self.model, criterion=self.criterion)
                        else:
                             raise ValueError("adv_train must be None, fgsm, or pgd")
                        try:
                            outputs = self.model(perturbed_data)
                        except RuntimeError:
                            outputs = self.model(perturbed_data.unsqueeze(-1))
                    else:
                        try:
                            outputs = self.model(inputs)
                        except RuntimeError:
                            outputs = self.model(inputs.unsqueeze(-1))
                    loss = self.criterion(outputs, labels)
                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                    bar.title('Training, Epoch: [{0:d} / {1:d}], {2:s}={3:3.3f}'.format(epoch, epochs, str(self.criterion), epoch_loss / (i+1)))
                    loss.backward()
                    optimizer.step()
                    bar()
            if old_loss < epoch_loss:
                logger.warning("Training loss is increasing over previous epoch!")
            elif old_loss - epoch_loss < old_loss * 0.001:
                logger.warning("Training loss decreased by less than 0.1% of previous epoch!")
            old_loss = epoch_loss
            if not val_split:
                self.save_if_best(model_name, task_path, epoch_loss / (i+1))

            logger.debug("Main validation loop")
            if val_split:
                if epoch % validation_period == 0:
                    epoch_vloss = 0
                    with alive_bar(len(validationloader), bar='circles', spinner='arrows') as vbar:
                        vbar.title('Validation, Epoch: [{0:d} / {1:d}]'.format(epoch, epochs))
                        with torch.no_grad():
                            for j, vdata in enumerate(validationloader):
                                vinputs, vlabels = vdata
                                try:
                                    voutputs = self.model(vinputs)
                                except RuntimeError:
                                    voutputs = self.model(vinputs.unsqueeze(-1))
                                vloss = self.criterion(voutputs, vlabels)
                                epoch_vloss += vloss
                                vbar.title('Validation, Epoch: [{0:d} / {1:d}], {2:s}={3:3.3f}'.format(epoch, epochs, str(self.criterion), epoch_vloss / (j+1)))
                                vbar()
                    self.save_if_best(model_name, task_path, epoch_vloss / (j+1))
                    if epoch_vloss < epoch_loss:
                        logger.warning("Validation loss is less than training loss!")
                    if old_vloss < epoch_vloss:
                        logger.warning("Validation loss is increasing over previous epoch!")
                    elif old_vloss - epoch_vloss < old_vloss * 0.001:
                        logger.warning("Validation loss decreased by less than 0.1% of previous epoch!")
                    old_vloss = epoch_vloss

    def test(self, x, y=None, batch_size=1):
        '''
        Performs CNN inference, producing a set of class predictions or regression estimates. If labels (y) are given, will compute F1 score or Mean squared error, depending on task

        @param x <tensor>: testing data to either classify or perform regression on
        @param y <tensor>: testing data labels for evaluation (optional)
        @param batch_size <int>: to predict in batches

        @return scores <array>: Class predictions <int> ranging in values from 0 to N_class-1, or regression estimates <float>
        @return self.test_metric <float>: F1 score for classification or MSE for regression
        '''
        test_set = PytorchDataset(x, y, return_idx=True, transform=self.transform)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        idx_label_score = []
        self.model.eval()
        logger.info('Starting testing...')
        start_time = time.time()
        with torch.no_grad():
            for data in test_loader:
                if y is not None:
                    inputs, labels, idx = data
                else:
                    inputs, idx = data
                try:
                    outputs = self.model(inputs)
                except RuntimeError:
                    outputs = self.model(inputs.unsqueeze(-1))
                if self.task == 'classification':
                    _, scores = torch.max(outputs, 1)
                else:
                    scores = outputs
                if y is not None:
                    idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                else:
                     idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)
        if y is not None:
            _, labels, scores = zip(*idx_label_score)
            labels = np.array(labels)
            scores = np.array(scores)
            if self.task == 'classification':
                self.test_metric = f1_score(labels, scores, average='weighted')
                logger.info('Test set F1 score: {:.5f}%'.format(self.test_metric))
            else:
                self.test_metric = np.mean((labels-scores)**2)
                logger.info('Test set MSE: {:.5f}'.format(self.test_metric))
        else:
            _, scores = zip(*idx_label_score)
            scores = np.array(scores)
            self.test_metric = None
        logger.info('Finished testing')
        return scores, self.test_metric
