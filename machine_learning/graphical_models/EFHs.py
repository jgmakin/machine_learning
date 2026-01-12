# standard libraries
import pdb
import os
import math
import time

# third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils
# import torch.distributions
# from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# local libraries
from neural_models.probabilistic_population_codes import TorchMultisensoryData
from utils_jgm.machine_compatibility_utils import MachineCompatibilityUtils
from utils_jgm.tikz_pgf_helpers import tpl_save
from utils_jgm.toolbox import tau, print_fixed
MCUs = MachineCompatibilityUtils()

tau = torch.tensor(tau)
torch.manual_seed(0)


'''
(Generalized) exponential-family harmoniums

 :Author: J.G. Makin (except where otherwise noted)

Last modified:  01/07/2025
Created:        12/08/2025


'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)


class EFH(nn.Module):
    def __init__(
        self,
        N_visible=784,
        N_hidden=256,
        emission_family='Bernoulli',
        posterior_family='Bernoulli',
        N_trials=4,  # for multinomial sampler
    ):
        super().__init__()
        self.W = nn.Parameter(torch.randn(N_hidden, N_visible) * 0.01)
        self.b_vis = nn.Parameter(torch.zeros(N_visible))
        self.b_hid = nn.Parameter(torch.zeros(N_hidden))

        self.emission_family = emission_family
        self.posterior_family = posterior_family
        self.N_trials = N_trials

    def moments_to_samples(self, moments, distribution, **kwargs):
        match distribution:
            case 'Bernoulli':
                return torch.distributions.Bernoulli(probs=moments, **kwargs).sample()
            case 'Poisson':
                return torch.distributions.Poisson(rate=moments, **kwargs).sample()
            case 'multinomial':
                samples = multinomial_gemini(moments/self.N_trials, self.N_trials)
                
                # assume probs sum in dimension 1!!!
                # NB: moments should be divided by N_trials to be probs, but
                #   PyTorch will automatically normalize
                # counts = torch.distributions.Multinomial(
                #     probs=moments.permute([0, 2, 3, 1]),
                #     total_count=self.N_trials
                # ).sample()
                # samples = counts.permute([0, 3, 1, 2])

                # mean-field
                # samples = mu

                return samples
            case _:
                raise NotImplementedError(
                    'Sampling not implemented for %s distribution' % distribution
                )

    def inverse_link(self, natural_parameters, distribution):
        match distribution:
            case 'Bernoulli':
                return torch.sigmoid(natural_parameters)
            case 'Poisson':
                return torch.exp(natural_parameters)
            case 'multinomial':
                # assume probs sum in dimension 1!!!
                # E[X] = N*p
                return self.N_trials*F.softmax(natural_parameters, dim=1)
            case _:
                raise NotImplementedError(
                    'Inverse link not implemented for %s distribution' % distribution
                )

    def conditional(self, X, weights, biases, family):
        eta = F.linear(X, weights, biases)
        mu = self.inverse_link(eta, family)
        samples = self.moments_to_samples(mu, family)
        return mu, samples

    def infer(self, vis):
        return self.conditional(vis, self.W, self.b_hid, self.posterior_family)

    def emit(self, hid):
        return self.conditional(hid, self.W.T, self.b_vis, self.emission_family)

    def forward(self, hid, N_steps):
        # block Gibbs sample
        for _ in range(N_steps):
            mu_vis, vis = self.emit(hid)
            mu_hid, hid = self.infer(vis)

        return mu_vis, vis, mu_hid, hid

    def generate(self, N_CD_steps, num_examples):

        # assume that generation starts from standard normal noise
        V0 = torch.randn((num_examples, self.W.shape[1]))

        # _, H0 = self.infer(V0)
        # mu_VN, _, _, _ = self.forward(H0, N_CD_steps)

        mu_H0, _ = self.infer(V0)
        mu_VN, _, _, _ = self.forward(mu_H0, N_CD_steps)
    
        return mu_VN


class GEFH(EFH):
    def __init__(
        self,
        C_visible=2,
        C_hidden=20,
        kernel_width=10,
        emission_family='Bernoulli',
        posterior_family='multinomial',
        N_trials=4,
        initialization='other',
        # initialization='Glorot',
    ):

        super().__init__()
        # For convolutional networks, it doesn't really make much sense to
        #  specify the total number of hidden units, because bigger images
        #  will drive more units (etc.).  Instead, we simply specify the
        #  number of *channels*.
        self.b_vis = nn.Parameter(torch.zeros(C_visible))
        self.b_hid = nn.Parameter(torch.zeros(C_hidden))
        self.W = nn.Parameter(
            torch.empty(C_hidden, C_visible, kernel_width, kernel_width)
        )
        match initialization:
            case 'Glorot':
                # use Xavier/Glorot for Sigmoid/Softmax families:
                nn.init.xavier_uniform_(self.W) 
            case _:
                # OR use a very small normal distribution (Common RBM trick):
                # nn.init.normal_(self.W, mean=0, std=0.01)
                nn.init.normal_(self.W, mean=0, std=0.001)
        self.stride = 1

        self.emission_family = emission_family
        self.posterior_family = posterior_family
        self.N_trials = N_trials

    def infer(self, vis):
        eta = F.conv2d(vis, self.W, self.b_hid, stride=self.stride, padding=1)
        mu = self.inverse_link(eta, self.posterior_family)
        samples = self.moments_to_samples(mu, self.posterior_family)

        return mu, samples

    def emit(self, hid):
        eta = F.conv_transpose2d(
            hid, self.W, self.b_vis, stride=self.stride, padding=1,
            output_padding=0
        )
        mu = self.inverse_link(eta, self.emission_family)
        samples = self.moments_to_samples(mu, self.emission_family)
        return mu, samples

    def generate(self, N_CD_steps, num_examples, H, W):

        data_shape = (num_examples, self.W.shape[1], H, W)
        
        # assume that generation starts from standard normal noise
        V0 = torch.randn(data_shape)

        _, H0 = self.infer(V0)
        mu_VN, _, _, _ = self.forward(H0, N_CD_steps)
    
        return mu_VN


# Training loop
def train_EFH(
    training_loader,
    validator,
    N_CD_steps=1,
    N_steps=1000,
    N_steps_print=None,
    CONV=False,
    data_init_fraction=0,  # 0.04
    **EFH_kwargs,
):
    
    if N_steps_print is None:
        N_steps_print = N_steps//5

    efh = GEFH(**EFH_kwargs) if CONV else EFH(**EFH_kwargs)
    optimizer = torch.optim.Adam(efh.parameters(), lr=0.001)

    start = time.time()
    for step, (V0, _) in enumerate(training_loader):
        total_loss = 0
        N_examples_per_batch = V0.shape[0]
            
        # "positive" phase
        _, H0 = efh.infer(V0)

        # Abdulsalam 2025
        if data_init_fraction == 0: 
            H0_neg = H0
        else:
            # this could be made much more efficient....
            N_data_init_examples = round(N_examples_per_batch*data_init_fraction)
            inds = np.random.choice(N_examples_per_batch, N_data_init_examples)
            V_random = torch.randn((N_data_init_examples, *V0.shape[1:]))
            _, H_random = efh.infer(V_random)

            H0_neg = H0
            H0_neg[inds] = H_random

        # negative phase
        # mu_VN, VN, mu_HN, HN = efh(H0_neg, N_CD_steps)
        mu_VN, VN, mu_HN, HN = efh(H0, N_CD_steps)
        ##########
        # standard GEH hack
        HN = mu_HN
        ##########

        # CD-N
        if CONV:
            grad_W_pos = nn.grad.conv2d_weight(V0, efh.W.shape, H0, stride=efh.stride, padding=1)
            grad_W_neg = nn.grad.conv2d_weight(VN, efh.W.shape, HN, stride=efh.stride, padding=1)
            efh.W.grad = (grad_W_neg - grad_W_pos) / N_examples_per_batch
            efh.b_vis.grad = -(V0 - VN).sum(dim=(0, 2, 3)) / N_examples_per_batch
            efh.b_hid.grad = -(H0 - HN).sum(dim=(0, 2, 3)) / N_examples_per_batch
        else:
            efh.W.grad = - (H0.T @ V0 - HN.T @ VN) / N_examples_per_batch
            efh.b_vis.grad = -(V0 - VN).sum(dim=0) / N_examples_per_batch
            efh.b_hid.grad = -(H0 - HN).sum(dim=0) / N_examples_per_batch

        optimizer.step()
        optimizer.zero_grad()

        loss = torch.mean((V0 - mu_VN)**2)
        total_loss += loss.item()

        if (step+1) % N_steps_print == 0:
            torch.cuda.synchronize()
            
            print_fixed("step", step, 6, 0, 0, end='')
            print_fixed("infer time", time.time() - start, 8, 3, 0, end='')
            # print_fixed("loss", total_loss, 6, 2, 0, end='')
            print_fixed("loss", total_loss, 6, 2, -2, end='')

            efh.eval()
            validator(efh, step)
            efh.train()
            
            print()
            start = time.time()

        if step == N_steps:
            break

    return efh


# Training loop
def train_DBN(
    training_loader,
    validator,
    N_EFH=1,
    N_CD_steps=1,
    N_steps=1000,
    N_steps_print=None,
    CONV=False,
    data_init_fraction=0,  # 0.04
    **EFH_kwargs,
):
    
    if N_steps_print is None:
        N_steps_print = N_steps//5

    efh_list = []

    for iEFH in range(N_EFH):
        efh = GEFH(**EFH_kwargs) if CONV else EFH(**EFH_kwargs)
        efh_list.append(efh)
        
        optimizer = torch.optim.Adam(efh.parameters(), lr=0.001)

        start = time.time()
        for step, (mu_Y, _) in enumerate(training_loader):
            total_loss = 0
            N_examples_per_batch = mu_Y.shape[0]

            # push data up through stack of EFHs
            for efh in efh_list:
                mu_Y, V0 = efh.infer(mu_Y)
                
            # "positive" phase
            _, H0 = efh.infer(V0)

            # Abdulsalam 2025
            if data_init_fraction == 0: 
                H0_neg = H0
            else:
                ### is this the most efficient way?
                N_data_init_examples = round(N_examples_per_batch*data_init_fraction)
                inds = np.random.choice(N_examples_per_batch, N_data_init_examples)
                V_random = torch.randn((N_data_init_examples, *V0.shape[1:]))
                _, H_random = efh.infer(V_random)

                H0_neg = H0
                H0_neg[inds] = H_random

            # negative phase
            # mu_VN, VN, mu_HN, HN = efh(H0_neg, N_CD_steps)
            mu_VN, VN, mu_HN, HN = efh(H0, N_CD_steps)
            ##########
            # standard GEH hack
            HN = mu_HN
            ##########

            # CD-N
            if CONV:
                grad_W_pos = nn.grad.conv2d_weight(V0, efh.W.shape, H0, stride=efh.stride, padding=1)
                grad_W_neg = nn.grad.conv2d_weight(VN, efh.W.shape, HN, stride=efh.stride, padding=1)
                efh.W.grad = (grad_W_neg - grad_W_pos) / N_examples_per_batch
                efh.b_vis.grad = -(V0 - VN).sum(dim=(0, 2, 3)) / N_examples_per_batch
                efh.b_hid.grad = -(H0 - HN).sum(dim=(0, 2, 3)) / N_examples_per_batch
            else:
                efh.W.grad = - (H0.T @ V0 - HN.T @ VN) / N_examples_per_batch
                efh.b_vis.grad = -(V0 - VN).sum(dim=0) / N_examples_per_batch
                efh.b_hid.grad = -(H0 - HN).sum(dim=0) / N_examples_per_batch

            optimizer.step()
            optimizer.zero_grad()

            loss = torch.mean((V0 - mu_VN)**2)
            total_loss += loss.item()

            if (step+1) % N_steps_print == 0:
                torch.cuda.synchronize()
                
                print_fixed("step", step, 6, 0, 0, end='')
                print_fixed("infer time", time.time() - start, 8, 3, 0, end='')
                # print_fixed("loss", total_loss, 6, 2, 0, end='')
                print_fixed("loss", total_loss, 6, 2, -2, end='')

                efh.eval()
                validator(efh, step)
                efh.train()
                
                print()
                start = time.time()

            if step == N_steps:
                break

        print('------------------')
        print('trained EFH %i' % iEFH)
        print('------------------')

    return efh_list

#-----------------------------------------------------------------------------#
# Fast multinomial samplers (courtesy of Gemini 3.0)
#-----------------------------------------------------------------------------#
def multinomial_gemini(probs, total_count):
    N, C, H, W = probs.shape
    device = probs.device
    
    # 1. Compute CDF and flatten: (N*H*W, C)
    # We permute so that the Channel dimension is last
    cdf_flat = torch.cumsum(probs, dim=1).permute(0, 2, 3, 1).reshape(-1, C)
    
    # 2. Generate random numbers: (N*H*W, total_count)
    # We transpose the random numbers so the batch dim (N*H*W) comes first
    rr = torch.rand((N * H * W, total_count), device=device)
    
    # 3. Searchsorted: O(log C)
    # indices shape: (N*H*W, total_count)
    indices = torch.searchsorted(cdf_flat, rr.contiguous())
    
    # 4. Tally the counts
    # We use scatter_add_ to avoid a Python loop. 
    # We need a 'src' tensor of ones to add up.
    counts_flat = torch.zeros_like(cdf_flat)
    ones = torch.ones_like(indices, dtype=probs.dtype)
    
    # scatter_add_(dim, index, src)
    counts_flat.scatter_add_(1, indices.clamp(max=C-1), ones)

    # 5. Reshape back to (N, C, H, W)
    return counts_flat.view(N, H, W, C).permute(0, 3, 1, 2)


def ultra_fast_4d_scatter(probs, total_count):
    '''
    Useful when total_count is quite small.  Then the memory savings outweight
    the cost of the loop.
    '''

    N, C, H, W = probs.shape
    device = probs.device
    
    # Pre-allocate output to avoid repeated allocation overhead
    counts = torch.zeros_like(probs) 
    
    # Compute CDF once
    cdf = torch.cumsum(probs, dim=1)
    
    for _ in range(total_count):
        # Generate 1 random set at a time to keep memory footprint tiny
        rr = torch.rand((N, 1, H, W), device=device)
        # Find indices (N, H, W)
        indices = (rr > cdf).sum(dim=1, keepdim=True).clamp(max=C-1)
        # Add 1 to the counts at these indices
        counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=probs.dtype))
        
    return counts.float()


def ultra_fast_4d_counts(probs, total_count):
    # probs: (N, 20, H, W)
    N, C, H, W = probs.shape
    probs_half = probs.half()
    
    # 1. Generate all random numbers at once 
    # Shape: (total_count, N, 1, H, W)
    rr = torch.rand((total_count, N, 1, H, W), dtype=torch.half)

    # 2. Compute Cumulative Distribution (CDF) 
    # Shape: (N, 20, H, W)
    cdf = torch.cumsum(probs_half, dim=1)
    
    # 3. Compare random numbers to CDF
    # (total_count, N, 1, H, W) > (1, N, 20, H, W) -> (total_count, N, 20, H, W)
    # The sum along the C dimension tells us how many boundaries were crossed
    # but we want the specific index. 'r < cdf' gives a mask.
    # Logic: how many categories are "to the left" of our random number?
    samples = (rr > cdf.unsqueeze(0)).sum(dim=2)  # Shape: (total_count, N, H, W)
    samples = torch.clamp(samples, max=C-1)
    
    # 4. Use bincount-like logic via one-hot to get final counts
    counts = F.one_hot(samples, num_classes=C).sum(dim=0).float()
    # Result: (N, H, W, C)
    
    return counts.permute([0, 3, 1, 2])  # Back to (N, C, H, W)


def fast_multinomial_sample(probs, total_count):

    N, C, H, W = probs.shape
    probs_flat = probs.permute([0, 2, 3, 1]).reshape(-1, C)
    indices = torch.multinomial(probs_flat, num_samples=total_count, replacement=True)
    one_hots = F.one_hot(indices, num_classes=C)
    counts_flat = one_hots.sum(dim=1).to(probs.dtype)
    counts = counts_flat.view(N, H, W, C).permute(0, 3, 1, 2)

    return counts.float()


def fast_multinomial_sample_old(natural_params, total_count):
    # natural_params shape: (Batch, Channels, H, W)
    # We want to sample across the Channel dimension (dim=1)
    
    # 1. Generate Gumbel noise
    # G = -log(-log(U)) where U ~ Uniform(0, 1)
    uniform_samples = torch.rand(total_count, *natural_params.shape, device=natural_params.device)
    centered_gumbel_samples = -torch.log(-torch.log(uniform_samples + 1e-10) + 1e-10)
    
    # 2. Add noise to logits and find the argmax for each of the 'total_count' draws
    # perturbed_logits shape: (total_count, Batch, Channels, H, W)
    gumbel_samples = natural_params.unsqueeze(0) + centered_gumbel_samples
    
    # 3. Get the indices of the max for each draw
    indices = gumbel_samples.argmax(dim=2)  # shape: (total_count, Batch, H, W)
    
    # 4. Convert indices to one-hot and sum them up to get the final counts
    # (Doing this via scatter_add is very fast on GPU)
    counts = torch.zeros_like(natural_params)
    # Reshape for scatter:
    # We essentially "brush" the indices into the count bins
    for i in range(total_count):
        counts.scatter_add_(
            1, indices[i:i+1],
            torch.ones_like(indices[i:i+1], dtype=natural_params.dtype)
        )
        
    return counts


@torch.no_grad()
def plot_reconstruction_components(efh, vis_image):
    # 1. Infer hidden states
    # vis_image shape: (1, C_vis, H, W)
    mu_h, h_samples = efh.infer(vis_image)
    
    # 2. Get the top active channels
    # We'll look at the mean-field mu_h to see which filters are "active"
    # sum over spatial dimensions to find the strongest channels
    channel_strength = mu_h.sum(dim=(2, 3)).squeeze() 
    top_channels = torch.topk(channel_strength, k=4).indices
    
    # 3. Compute individual contributions to the log-rate (eta)
    # eta = conv_transpose(H) + bias
    # We'll isolate specific channels
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    
    # Show Original
    axes[0].imshow(vis_image[0, 0].cpu(), cmap='gray')
    axes[0].set_title("Original V0")
    
    # Show Total Reconstruction mu
    mu_v, _ = efh.emit(mu_h)
    axes[1].imshow(mu_v[0, 0].cpu(), cmap='magma')
    axes[1].set_title("Total Rec (mu_V)")
    
    # Show Top 4 Contributing Components
    for i, ch in enumerate(top_channels):
        # Create a "one-hot" hidden volume for just this channel
        h_single = torch.zeros_like(mu_h)
        h_single[:, ch, :, :] = mu_h[:, ch, :, :]
        
        # Emit just that one channel (without bias to see pure filter contribution)
        eta_single = F.conv_transpose2d(h_single, efh.W, stride=efh.stride, padding=1)
        comp = torch.exp(eta_single) # The Poisson rate contribution
        
        axes[i+2].imshow(comp[0, 0].cpu(), cmap='magma')
        axes[i+2].set_title(f"Comp: Ch {ch}")
        
    for ax in axes:
        ax.axis('off')
        plt.show()


def generate_and_plot(efh, CONV, N_CD_steps, C, H, W, N_rows=5, N_cols=5):
    # This functon makes sense for CIFAR and MNIST, but not multisensory data:
    #  (1) FLAT multisensory data are not equivalent to flattened NCHW data,
    #      because you want to allow for unequally sized populations.  So the
    #      reshaping for (CONV=False) networks is wrong.
    #  (2) In any case, imshow doesn't know what to do with 2 channels and
    #      fails (it can only handle 1, 3, or 4 channel data).

    if CONV:
        VF = efh.generate(N_CD_steps, N_rows*N_cols, H, W)
    else:
        VF = efh.generate(N_CD_steps, N_rows*N_cols)
        VF = VF.reshape((N_rows*N_cols, C, H, W))

    # imshow wants channels *last*
    VF = VF.permute(0, 2, 3, 1)

    fig, AXES = plt.subplots(N_rows, N_cols)
    for i, axes in enumerate(AXES):
        for k, ax in enumerate(axes):
            if C in [1, 3, 4]:
                # imshow can handle 1, 3, or 4 channels
                ax.imshow(VF[k + i*N_cols].cpu().detach())
            else:
                # just plot the first channel
                ax.imshow(VF[k + i*N_cols].cpu().detach())
