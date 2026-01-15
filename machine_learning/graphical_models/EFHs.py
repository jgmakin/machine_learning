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
from torch.utils.data import IterableDataset 

import numpy as np
import matplotlib.pyplot as plt

# local libraries
from utils_jgm.machine_compatibility_utils import MachineCompatibilityUtils
from utils_jgm.tikz_pgf_helpers import tpl_save
from utils_jgm.toolbox import tau, print_fixed, auto_attribute
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
# assume probs sum in dimension 1!!!
# NB: moments should be divided by N_trials to be probs, but
#   PyTorch will automatically normalize
# counts = torch.distributions.Multinomial(
#     probs=moments.permute([0, 2, 3, 1]),
#     total_count=self.N_trials
# ).sample()
# samples = counts.permute([0, 3, 1, 2])


class EFH(nn.Module):
    def __init__(
        self,
        C_in=2,
        C_out=20,
        kernel_width=1,
        stride=1,
        CONV=False,
        emission_family='Bernoulli',
        posterior_family='multinomial',
        N_trials=4,
        initialization='other',
        # initialization='Glorot',
    ):

        super().__init__()
        self.b_vis = nn.Parameter(torch.zeros(C_in))
        self.b_hid = nn.Parameter(torch.zeros(C_out))
        self.W = nn.Parameter(
            torch.empty(C_out, C_in, kernel_width, kernel_width)
        )
        match initialization:
            case 'Glorot':
                # use Xavier/Glorot for Sigmoid/Softmax families:
                nn.init.xavier_uniform_(self.W) 
            case _:
                # OR use a very small normal distribution (Common RBM trick):
                # nn.init.normal_(self.W, mean=0, std=0.01)
                nn.init.normal_(self.W, mean=0, std=0.001)
        self.W.data = self.W.data.to(memory_format=torch.channels_last)

        self.stride = stride
        self.emission_family = emission_family
        self.posterior_family = posterior_family
        self.N_trials = N_trials

    def infer(self, vis):
        # assert vis.dim() > 2

        eta = F.conv2d(vis, self.W, self.b_hid, stride=self.stride, padding=0)
        mu = self.inverse_link(eta, self.posterior_family)
        samples = self.moments_to_samples(mu, self.posterior_family)

        return mu, samples

    def emit(self, hid):
        eta = F.conv_transpose2d(
            hid, self.W, self.b_vis, stride=self.stride, padding=0,
            output_padding=0
        )
        mu = self.inverse_link(eta, self.emission_family)
        samples = self.moments_to_samples(mu, self.emission_family)
        return mu, samples

    def forward(self, hid, N_CD_steps):
        # block Gibbs sample
        for _ in range(N_CD_steps):
            mu_vis, vis = self.emit(hid)
            mu_hid, hid = self.infer(vis)

        return mu_vis, vis, mu_hid, hid

    @torch.no_grad()
    def updown(self, vis, USE_MEANS=True):
        mu_H, H = self.infer(vis)
        hid = mu_H if USE_MEANS else H
        mu_V, V = self.emit(hid)
        vis = mu_V if USE_MEANS else V
        return vis

    @torch.no_grad()
    def generate(self, N_CD_steps, num_examples, H, W, USE_MEANS=True):

        # ...
        data_shape = (num_examples, self.W.shape[1], H, W)
        
        # assume that generation starts from standard normal noise
        V0 = torch.randn(data_shape)

        mu_H0, H0 = self.infer(V0)
        hid = mu_H0 if USE_MEANS else H0
        mu_VN, _, _, _ = self.forward(hid, N_CD_steps)
    
        return mu_VN

    @torch.no_grad()
    def _gradient_update(self, V0, H0, VN, HN):

        N = V0.shape[0]
        w_shape = self.W.shape

        # this is secretly just a 1x1 conv (perhaps a linear layer)
        if w_shape[2] == 1 and w_shape[3] == 1:
            H0_flat = H0.permute(0, 2, 3, 1).reshape(N, -1)
            V0_flat = V0.permute(0, 2, 3, 1).reshape(N, -1)
            grad_W_pos = (H0_flat.T @ V0_flat).view(w_shape)

            HN_flat = HN.permute(0, 2, 3, 1).reshape(N, -1)
            VN_flat = VN.permute(0, 2, 3, 1).reshape(N, -1)
            grad_W_neg = (HN_flat.T @ VN_flat).view(w_shape)
        else:
            grad_W_pos = nn.grad.conv2d_weight(
                V0, w_shape, H0, stride=self.stride, padding=0
            )
            grad_W_neg = nn.grad.conv2d_weight(
                VN, w_shape, HN, stride=self.stride, padding=0
            )
        self.W.grad = (grad_W_neg - grad_W_pos)/N
        self.b_vis.grad = -(V0 - VN).sum(dim=(0, 2, 3))/N
        self.b_hid.grad = -(H0 - HN).sum(dim=(0, 2, 3))/N

    @torch.no_grad()
    def moments_to_samples(self, moments, distribution, **kwargs):
        match distribution:
            case 'Bernoulli':
                return torch.distributions.Bernoulli(probs=moments, **kwargs).sample()
            case 'Poisson':
                return torch.distributions.Poisson(rate=moments, **kwargs).sample()
            case 'multinomial':
                samples = multinomial_gemini(moments/self.N_trials, self.N_trials)
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
                ######
                # clamped exp or softplus?
                return torch.exp(natural_parameters)
                ######
            case 'multinomial':
                # assume probs sum in dimension 1!!!
                # E[X] = N*p
                return self.N_trials*F.softmax(natural_parameters, dim=1)
            case _:
                raise NotImplementedError(
                    'Inverse link not implemented for %s distribution' % distribution
                )


class EFHtrainer():
    @auto_attribute
    def __init__(
        self,
        efh,
        training_loader,
        validator,
        N_CD_steps=1,
        N_steps=1000,
        N_steps_print=None,
        data_init_fraction=0,  # 0.04
    ):
        self.optimizer = torch.optim.Adam(self.efh.parameters(), lr=0.001)
        self.N_steps_print = N_steps//5 if N_steps_print is None else N_steps_print

    @torch.no_grad()
    def __call__(self):
        start = time.time()
        for step, (V0, _) in enumerate(self.training_loader):
            self.optimizer.zero_grad()

            # collect stats from Gibbs sampling and update params
            with torch.no_grad():
                H0, VN, HN, recon_error = self._step(V0)
            
            self.efh._gradient_update(V0, H0, VN, HN)
            self.optimizer.step()

            # validate
            if (step+1) % self.N_steps_print == 0:
                start = self._validate(start, step, recon_error)
                
            # break condition
            if step == self.N_steps:
                break

    @torch.no_grad()
    def _step(self, V0):

        # "positive" phase
        _, H0 = self.efh.infer(V0)

        # Abdulsalam 2025
        if self.data_init_fraction > 0:
            H0 = self.noise_init(H0, V0.shape)

        # negative phase
        mu_VN, VN, mu_HN, HN = self.efh(H0, self.N_CD_steps)
        recon_error = torch.mean((V0 - mu_VN)**2)
        
        # standard GEH hack
        return H0, VN, mu_HN, recon_error
        ### return H0, VN, HN, recon_error

    @torch.no_grad()
    def noise_init(self, H0, V0_shape):
        # Abdulsalam 2025

        # Ns
        N_examples_per_batch = V0_shape[0]
        N_data_init_examples = round(N_examples_per_batch*self.data_init_fraction)

        # pass (a small number of) noise samples up through the network
        V_random = torch.randn((N_data_init_examples, *V0_shape[1:]))
        _, H_random = self.efh.infer(V_random)

        # replace random subset of inferences from data with inferences from noise
        inds = np.random.choice(N_examples_per_batch, N_data_init_examples)
        H0[inds] = H_random

        return H0

    @torch.no_grad()
    def _validate(self, start, step, recon_error):
        torch.cuda.synchronize()
        
        print_fixed("step", step, 6, 0, 0, end='')
        print_fixed("infer time", time.time() - start, 8, 3, 0, end='')
        print_fixed("loss", recon_error, 6, 2, -2, end='')

        self.efh.eval()
        self.validator(self.efh, step)
        self.efh.train()
        
        print()
        return time.time()


class DBN(nn.Module):
    def __init__(
        self, layer_specs=None, layers=None, **kwargs
    ):
        super().__init__()
      
        if layers is not None:
            self.layers = nn.ModuleList(layers) 
        elif layer_specs is not None:
            self.layers = nn.ModuleList()

            C_in = layer_specs[0]['channels']
            emission_family = layer_specs[0]['family']

            for layer_spec in layer_specs[1:]:
                C_out = layer_spec['channels']
                posterior_family = layer_spec['family']
                kernel_width = layer_spec['kernel width']
                N_trials = layer_spec.get('N_trials', None)
                # create an EFH and append it to the layer list
                new_efh = EFH(
                    C_in=C_in,
                    C_out=C_out,
                    kernel_width=kernel_width,
                    emission_family=emission_family,
                    posterior_family=posterior_family,
                    N_trials=N_trials,
                    **kwargs
                )
                self.layers.append(new_efh)

                # for next time around
                #########
                # This won't quite work if one EFH is conv and the next is not...
                C_in = C_out
                emission_family = posterior_family
                #########

        else:
            # self.layers = nn.ModuleList()
            pdb.set_trace()

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        # support for slicing like dbn[:i]
        if isinstance(idx, slice):
            # return DBN(nn.ModuleList(self.layers[idx]))
            return DBN(layers=self.layers[idx])
        return self.layers[idx]

    @torch.no_grad()
    def infer(self, vis, USE_MEANS=False):
        Y = vis
        for efh in self.layers:
            # w_shape = efh.W.shape
            # if w_shape[2] == 1 and w_shape[3] == 1:
            #     Y = Y.flatten(start_dim=1)
            mu_H, H = efh.infer(Y)
            Y = mu_H if USE_MEANS else H
        return Y

    @torch.no_grad()
    def emit(self, hid, USE_MEANS=False):
        X = hid
        for efh in reversed(self.layers):
            mu_V, V = efh.emit(X)
            X = mu_V if USE_MEANS else V
        return X

    @torch.no_grad()
    def updown(self, vis, USE_MEANS=True):
        hid = self.infer(vis, USE_MEANS=USE_MEANS)
        vis = self.emit(hid, USE_MEANS=USE_MEANS)
        return vis

    @torch.no_grad()
    def generate(self, N_CD_steps, num_examples, H=None, W=None, USE_MEANS=True):
        # compute H and W are for the *visible layer* of the *deepest* EFH from
        #  H and W for the *first* EFH.

        if len(self.layers) > 0:
            shapes = self.compute_spatial_path(H, W)
            H, W = shapes[-2]  # penultimate layer

            X = self.layers[-1].generate(
                N_CD_steps, num_examples, H=H, W=W, USE_MEANS=USE_MEANS
            )
            for efh in reversed(self.layers[:-1]):
                mu_V, V = efh.emit(X)
                X = mu_V if USE_MEANS else V
            return X

    def compute_spatial_path(self, H, W):
        """
        Traces H and W through every layer to find the shapes 
        at each 'junction' in the DBN.

        Courtesy of Gemini
        """

        shapes = [(H, W)]
        curr_h, curr_w = H, W
        
        for efh in self.layers:
            # kernel_width, stride, padding
            k = efh.W.shape[2]
            s = efh.stride
            p = getattr(efh, 'padding', 0) 
            
            # standard Conv2d output shape formula
            curr_h = ((curr_h + 2*p - k) // s) + 1
            curr_w = ((curr_w + 2*p - k) // s) + 1
            shapes.append((curr_h, curr_w))
            
        # a list of (H, W) for every junction
        return shapes  


class DBNtrainer():
    @auto_attribute
    def __init__(
        self,
        dbn,
        layer0_loader,
        validator,
        N_CD_steps=1,
        N_steps=1000,
        N_steps_print=None,
        data_init_fraction=0,  # 0.04
    ):
        N_EFHs = len(dbn)

        if type(N_CD_steps) is int:
            self.N_CD_steps = [N_CD_steps]*N_EFHs
        if type(N_CD_steps) is list:
            assert len(N_CD_steps) == N_EFHs

        if type(N_steps) is int:
            self.N_steps = [N_steps]*N_EFHs
        if type(N_steps) is list:
            assert len(N_steps) == N_EFHs

    def __call__(self):
        
        ### efh_validator = self.validator
        efh_validator = lambda efh, step: None
        for iEFH, (efh, CD_steps, steps) in enumerate(zip(
            self.dbn, self.N_CD_steps, self.N_steps
        )):

            # validate the entire DBN up to this point
            print('%i-EFH reconstruction: ' % iEFH, end='')
            self.validator(self.dbn[:iEFH], 0)
            print()

            # train 
            trainer = EFHtrainer(
                efh,
                LayeredDataset(self.layer0_loader, self.dbn[:iEFH]),
                efh_validator,
                N_CD_steps=CD_steps,
                N_steps=steps,
                N_steps_print=self.N_steps_print,
                data_init_fraction=self.data_init_fraction,
            )
            trainer()
            print('------------------')
            print('trained EFH %i' % iEFH)
            print('------------------')

            # but you can't validate deeper EFHs the normal way....
            efh_validator = lambda efh, step: None

        # validate entire DBN
        self.validator(self.dbn, 0)


class LayeredDataset(IterableDataset):
    def __init__(self, source_loader, dbn):
        self.source_loader = source_loader
        self.dbn = dbn

    def __iter__(self):
        for batch in self.source_loader:
            with torch.no_grad():
                datum, label = batch
                # Apply the whole stack in one tight loop
                for efh in self.dbn:
                    mu_H, H = efh.infer(datum)
                    # use means
                    datum = mu_H
                yield datum, label


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
        eta_single = F.conv_transpose2d(h_single, efh.W, stride=efh.stride, padding=0)
        comp = torch.exp(eta_single) # The Poisson rate contribution
        
        axes[i+2].imshow(comp[0, 0].cpu(), cmap='magma')
        axes[i+2].set_title(f"Comp: Ch {ch}")
        
    for ax in axes:
        ax.axis('off')
        plt.show()


def generate_and_plot(dbn, N_CD_steps, C, H, W, N_rows=5, N_cols=5):
    # This functon makes sense for CIFAR and MNIST, but not multisensory data:
    #  (1) FLAT multisensory data are not equivalent to flattened NCHW data,
    #      because you want to allow for unequally sized populations.  So the
    #      reshaping for (CONV=False) networks is wrong.
    #  (2) In any case, imshow doesn't know what to do with 2 channels and
    #      fails (it can only handle 1, 3, or 4 channel data).

    try:
        # it's a DBN
        w_shape = dbn[0].W.shape
    except TypeError:
        # it's an EFH
        w_shape = dbn.W.shape
    except IndexError:
        return

    if w_shape[2] == 1 and w_shape[3] == 1:
        # it's a "1x1" convolution and was probably flattened
        VF = dbn.generate(N_CD_steps, N_rows*N_cols, 1, 1)
        VF = VF.reshape(N_rows*N_cols, C, H, W)
    else:
        # it's a real convolution
        VF = dbn.generate(N_CD_steps, N_rows*N_cols, H, W)


    # imshow wants channels *last*
    VF = VF.permute(0, 2, 3, 1)

    fig, AXES = plt.subplots(N_rows, N_cols)
    for i, axes in enumerate(AXES):
        for k, ax in enumerate(axes):
            if C in [1, 3, 4]:
                # imshow can handle 1, 3, or 4 channels
                ax.imshow(VF[k + i*N_cols].cpu().detach())
                ax.set_axis_off()
            else:
                # just plot the first channel
                ax.imshow(VF[k + i*N_cols].cpu().detach())
                ax.set_axis_off()
    plt.show()


#####
# DEPRECATED
# This is a *non*-convolutional EFH.  You now interpret all EFHs as convs, just
#  possibly 1x1 convs....  (You do adjust the logic for computing the gradient in
#  that case, because it's 200x faster.)
#####
# class EFH(nn.Module):
#     def __init__(
#         self,
#         C_in=784,
#         C_out=256,
#         kernel_width=1,
#         emission_family='Bernoulli',
#         posterior_family='Bernoulli',
#         N_trials=4,  # for multinomial sampler
#     ):

#         super().__init__()
#         self.b_vis = nn.Parameter(torch.zeros(C_in))
#         self.b_hid = nn.Parameter(torch.zeros(C_out))
#         self.W = nn.Parameter(torch.randn(C_out, C_in) * 0.001)
        
#         self.emission_family = emission_family
#         self.posterior_family = posterior_family
#         self.N_trials = N_trials

#     @torch.no_grad()
#     def moments_to_samples(self, moments, distribution, **kwargs):
#         match distribution:
#             case 'Bernoulli':
#                 return torch.distributions.Bernoulli(probs=moments, **kwargs).sample()
#             case 'Poisson':
#                 return torch.distributions.Poisson(rate=moments, **kwargs).sample()
#             case 'multinomial':
#                 samples = multinomial_gemini(moments/self.N_trials, self.N_trials)
#                 # mean-field
#                 # samples = mu
                
#                 return samples
#             case _:
#                 raise NotImplementedError(
#                     'Sampling not implemented for %s distribution' % distribution
#                 )

#     def inverse_link(self, natural_parameters, distribution):
#         match distribution:
#             case 'Bernoulli':
#                 return torch.sigmoid(natural_parameters)
#             case 'Poisson':
#                 ######
#                 # clamped exp or softplus?
#                 return torch.exp(natural_parameters)
#                 ######
#             case 'multinomial':
#                 # assume probs sum in dimension 1!!!
#                 # E[X] = N*p
#                 return self.N_trials*F.softmax(natural_parameters, dim=1)
#             case _:
#                 raise NotImplementedError(
#                     'Inverse link not implemented for %s distribution' % distribution
#                 )

#     def conditional(self, X, weights, biases, family):
#         eta = F.linear(X, weights, biases)
#         mu = self.inverse_link(eta, family)
#         samples = self.moments_to_samples(mu, family)
#         return mu, samples

#     def infer(self, vis):
#         return self.conditional(vis, self.W, self.b_hid, self.posterior_family)

#     def emit(self, hid):
#         return self.conditional(hid, self.W.T, self.b_vis, self.emission_family)

#     def forward(self, hid, N_CD_steps):
#         # block Gibbs sample
#         for _ in range(N_CD_steps):
#             mu_vis, vis = self.emit(hid)
#             mu_hid, hid = self.infer(vis)

#         return mu_vis, vis, mu_hid, hid

#     @torch.no_grad()
#     def updown(self, vis, USE_MEANS=True):
#         mu_H, H = self.infer(vis)
#         hid = mu_H if USE_MEANS else H
#         mu_V, V = self.emit(hid)
#         vis = mu_V if USE_MEANS else V
#         return vis

#     @torch.no_grad()
#     def generate(self, N_CD_steps, num_examples, USE_MEANS=True):

#         # ...
#         data_shape = (num_examples, self.W.shape[1])

#         # assume that generation starts from standard normal noise
#         V0 = torch.randn(data_shape)
        
#         mu_H0, H0 = self.infer(V0)
#         hid = mu_H0 if USE_MEANS else H0
#         mu_VN, _, _, _ = self.forward(hid, N_CD_steps)
    
#         return mu_VN

#     @torch.no_grad()
#     def _gradient_update(self, V0, H0, VN, HN):
#         N_examples = V0.shape[0]
#         self.W.grad = - (H0.T @ V0 - HN.T @ VN) / N_examples
#         self.b_vis.grad = -(V0 - VN).sum(dim=0) / N_examples
#         self.b_hid.grad = -(H0 - HN).sum(dim=0) / N_examples


