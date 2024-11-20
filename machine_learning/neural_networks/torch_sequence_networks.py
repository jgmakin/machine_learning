# standard libraries
import pdb
from termcolor import cprint
from IPython.display import clear_output
import os
import math
from functools import partial

# third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
#######
# from torch.profiler import profile, record_function, ProfilerActivity
#######

# local
from machine_learning.data_mungers import TFRecordDataLoader
from machine_learning.torch_helpers import (
    get_word_error_rate, sequences_tools, reverse_sequences
)
from utils_jgm.toolbox import (
    auto_attribute, wer_vector, close_factors, MutableNamedTuple
)
from utils_jgm.machine_compatibility_utils import MachineCompatibilityUtils
MCUs = MachineCompatibilityUtils()


'''
Neural networks for sequence-to-label and sequence-to-sequence problems.

Some portions inspired by the tutorial here:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
 
 :Author: J.G. Makin (except where otherwise noted)

Created: November 2023
  by JGM
'''


###############
# (15) batch size?
# (16) NB: the penalty_scale probably means something different for MFCCs
#   b/c you are summing rather than averaging across the 13 dimensions...
# (17) serial transfer learning
# (18) print PER
###############


'''
Data orderings:
    canonical:          (N_cases x T x N_features)
    Conv1d:             (N_cases x N_features x T)
    Embedding input:    (*)
    Embedding output:   (*, N_features)
    RNN inputs:         (T x N_cases x N_features)    
    RNN outputs:        (T x N_cases x N_features)
    RNN states:         (N_layers*N_directions x N_cases x N_features)
'''


class Sequence2Sequence(nn.Module):
    @auto_attribute(CHECK_MANIFEST=True)
    def __init__(
        self,
        manifest,
        subnets_params,
        #####
        # kwargs set in the manifest
        layer_sizes=None,
        FF_dropout=None,
        RNN_dropout=None,
        TEMPORALLY_CONVOLVE=None,  ### currently does nothing
        #####
        ENCODER_RNN_IS_BIDIRECTIONAL=True,
        training_GPUs=None,
        EOS_token='<EOS>',
        pad_token='<pad>',
        TARGETS_ARE_SEQUENCES=True,
        max_hyp_length=20,
        # coupling='final_state',
        decoder_type='final_state_coupled',
        RNN_type='LSTM',
        VERBOSE=True,
    ):
        super().__init__()

        if not TARGETS_ARE_SEQUENCES:
            assert decoder_type == 'classifier', (
                "A classifier is required for non-sequence targets!"
            )

        # useful subnet_params
        #-------#
        # for now, assume there is only one decoder_targets_list
        self.decoder_targets_list = subnets_params[-1].data_manifests[
            'decoder_targets'].get_feature_list()
        #-------#
        self.EOS_token = EOS_token
        self.pad_token = pad_token

        self.EOS_id = self.decoder_targets_list.index(self.EOS_token)
        self.pad_id = self.decoder_targets_list.index(self.pad_token)

        self.vprint('USING %s IN THE RNNs' % RNN_type)

        # ENCODER
        self.encoder = EncoderRNN(
            self.layer_sizes, self.FF_dropout, self.RNN_dropout, subnets_params,
            ENCODER_RNN_IS_BIDIRECTIONAL, RNN_type, decoder_type=decoder_type,
            VERBOSE=VERBOSE
        )

        # reshape the context to pass to the decoder
        # self.reshape_context = lambda states: context_reshape(
        #     states, len(self.layer_sizes['decoder_rnn']),
        #     ENCODER_RNN_IS_BIDIRECTIONAL+1
        # )
        self.reshape_context = partial(
            context_reshape,  N_layers=len(self.layer_sizes['decoder_rnn']),
            N_directions=ENCODER_RNN_IS_BIDIRECTIONAL+1
        )

        # DECODER
        self.vprint('COUPLING ENCODER ', end='')
        match decoder_type:
            case 'final_state_coupled':
                self.decoder = DecoderRNN(
                    self.layer_sizes, self.FF_dropout, self.RNN_dropout,
                    subnets_params, 
                    self.EOS_id,  # use EOS as SOS, as in the TF1 version
                    max_hyp_length, RNN_type
                )
                self.vprint('VIA FINAL HIDDEN STATE TO DECODER')
            case 'attention_coupled':
                self.decoder = DecoderAttentionRNN(
                    self.layer_sizes, self.FF_dropout, self.RNN_dropout,
                    subnets_params, 
                    self.EOS_id,  # use EOS as SOS, as in the TF1 version
                    max_hyp_length, RNN_type
                )
                self.vprint('VIA ATTENTION TO DECODER')
            case 'classifier':
                self.decoder = FinalStateClassifier(
                    self.layer_sizes, self.FF_dropout, subnets_params,
                    ENCODER_RNN_IS_BIDIRECTIONAL
                )
                self.vprint('TO A FINAL-STATE CLASSIFIER')
            case _:
                raise ValueError('Unrecognized decoder_type')

        # accumulate the loss functions over subjects and their losses
        self.loss_fxn_dicts = {}
        for subnet_params in subnets_params:
            subnet_id = str(subnet_params.subnet_id)
            self.loss_fxn_dicts[subnet_id] = {}
            for key in subnet_params.data_mapping:
                if key.endswith('targets'):
                    data_manifest = subnet_params.data_manifests[key]
                    self.loss_fxn_dicts[subnet_id][key] = (
                        get_cross_entropy_fxn(data_manifest.distribution),
                        data_manifest.penalty_scale
                    )

    def forward(self, inputs, subnet_id, targets=None):
        '''
        The inputs, targets, and natural_params have JGM "canonical ordering,"

            (N_cases x T x N_features) and (N_cases x T)

        The RNN outputs have RNN ordering,

            (T x N_cases x N_features)

        and the RNN states ("similarly") have shape

            (N_layers*N_directions x N_cases x N_features)
        '''

        # for storing useful outputs
        natural_params_dict = {}
        image_dict = {}

        # encode; project outputs; init decoder state; decode; project outputs
        encoder_rnn_outputs, encoder_final_states = self.encoder(
            inputs, subnet_id, natural_params_dict, image_dict
        )
        if isinstance(encoder_final_states, tuple):
            context = tuple(
                self.reshape_context(states) for states in encoder_final_states
            )
        else:
            context = self.reshape_context(encoder_final_states)
        self.decoder(
            encoder_rnn_outputs, context, targets, natural_params_dict,
            image_dict,
        )

        return natural_params_dict, image_dict

    def print_sentences(
        self, most_probable_classes, decoder_targets, on_clr, N_sentences=10,
        PRINT_CRUDE_WER=False
    ):

        if PRINT_CRUDE_WER:
            accumulated_targets = []
            accumulated_predictions = []

        print()
        for iExample, (predicted_classes, target_classes) in enumerate(zip(
            most_probable_classes, decoder_targets
        )):
            predicted_words = class_indices_to_sequence(
                predicted_classes, self.decoder_targets_list,
                self.EOS_token, self.pad_token
            )
            target_words = class_indices_to_sequence(
                target_classes, self.decoder_targets_list,
                self.EOS_token, self.pad_token
            )

            # reduce cluter; don't print sentences for training data
            if True:  # data_partition == 'validation':
                cprint(
                    '{0:60} {1}'.format(target_words, predicted_words),
                    on_color=on_clr
                )

            if PRINT_CRUDE_WER:
                accumulated_targets.append(target_words.split())
                accumulated_predictions.append(predicted_words.split())

            # only print 30 sentences
            if iExample > N_sentences:
                break

        if PRINT_CRUDE_WER:
            print(' WERb: %1.3f' % np.mean(
                wer_vector(accumulated_targets, accumulated_predictions)
            ))

    def vprint(self, *args, **kwargs):
        if self.VERBOSE:
            print(*args, **kwargs)


class EncoderRNN(nn.Module):
    def __init__(
        self,
        layer_sizes,
        FF_dropout, 
        RNN_dropout,
        subnets_params,
        BIDIRECTIONAL,
        RNN_type,
        decoder_type=None,
        MAX_POOL=False,
        VERBOSE=True,
    ):
        super().__init__()

        # ...
        if len(np.unique(layer_sizes['encoder_rnn'])) > 1:
            raise NotImplementedError('Expected the same layer size for all layers')
        else:
            N_hidden = layer_sizes['encoder_rnn'][0]

        self.FF_dropout = FF_dropout
        self.RNN_dropout = RNN_dropout
        self.decoder_type = decoder_type

        # accumulate proprietary components (embeddings, projections)
        self.embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.decimation_factors = {}
        for subnet_params in subnets_params:
            subnet_id = str(subnet_params.subnet_id)

            # embedding_layers
            self.embeddings[subnet_id] = MLConvEmbedding(
                subnet_params.data_manifests['encoder_inputs'].num_features,
                layer_sizes['encoder_embedding'], subnet_params,
                self.FF_dropout, MAX_POOL, VERBOSE=VERBOSE
            )

            # decimation_factors
            self.decimation_factors[subnet_id] = subnet_params.decimation_factor

            # projections
            self.projections[subnet_id] = nn.ModuleDict()
            for key in subnet_params.data_mapping:
                if key.startswith('encoder') and key.endswith('targets'):

                    # useful quantities
                    data_manifest = subnet_params.data_manifests[key]
                    output_layer = int(key.split('_')[1])
                    N_outputs = data_manifest.num_features
                    Ns_hidden = layer_sizes['encoder_%i_projection' % output_layer]
                    N_inputs = layer_sizes['encoder_rnn'][output_layer]*(
                        BIDIRECTIONAL + 1
                    )

                    # accumulate this encoder projection
                    self.projections[subnet_id][key] = MultiLayerProjection(
                        N_inputs, Ns_hidden, N_outputs, self.FF_dropout,
                        input_list_index=output_layer,
                    )

        # the RNN; you may need outputs from intermediate layers, so you have
        #  to construct this one layer at a time
        self.RNNs = nn.ModuleList()
        RNN = getattr(nn, RNN_type)
        N_in = layer_sizes['encoder_embedding'][-1]
        for N_hidden in layer_sizes['encoder_rnn']:
            self.RNNs.append(
                RNN(N_in, N_hidden, num_layers=1, bidirectional=BIDIRECTIONAL)
            )
            N_in = N_hidden*(1 + BIDIRECTIONAL)
        
        ###############
        # flatten_parameters()
        ###############

    def forward(self, inputs, subnet_id, natural_params_dict, image_dict):

        # embed with temporal convolutions
        X = self.embeddings[subnet_id](inputs)

        # get lengths of *downsampled* input sequences
        inputs_indices, inputs_lengths = sequences_tools(
            inputs[:, ::self.decimation_factors[subnet_id], :]
        )

        # reverse? (a la Sutskever 2014); canonical ordering -> RNN ordering
        if self.decoder_type == 'final_state_coupled':
            X = reverse_sequences(X, inputs_indices, inputs_lengths)
        X = X.permute(1, 0, 2)
        
        # the original TF version used dropout on the RNN *inputs*
        X = F.dropout(
            X, self.RNN_dropout, training=self.training,
            inplace=True
        )

        # put into RNN ordering (T x N_cases x N_features) and "pack"
        X = nn.utils.rnn.pack_padded_sequence(
            X, inputs_lengths.to('cpu'), enforce_sorted=False
        )

        # run through RNN---layer by layer, to accumulate outputs for encoder
        #  targeting
        all_outputs = []
        all_final_states = []
        for RNN_layer in self.RNNs:
            X, final_states = RNN_layer(X)
            X, _ = nn.utils.rnn.pad_packed_sequence(X)
            X = F.dropout(
                X, self.RNN_dropout, training=self.training,
                inplace=True
            )
            all_outputs.append(X)
            all_final_states.append(final_states)
            X = nn.utils.rnn.pack_padded_sequence(
                X, inputs_lengths.to('cpu'), enforce_sorted=False
            )

        # pack the final states together the way PyTorch does
        if isinstance(all_final_states[0], tuple):
            # LSTM
            all_final_states = tuple(
                torch.stack(states).flatten(end_dim=1)
                for states in zip(*all_final_states)
            )
        else:
            # GRU
            all_final_states = torch.stack(all_final_states).flatten(end_dim=1)

        # "project" into natural params and convert back to canonical ordering
        for key, projection in self.projections[subnet_id].items():
            natural_params_dict[key] = projection(all_outputs).permute(1, 0, 2)

        return all_outputs, all_final_states


class FinalStateClassifier(nn.Module):
    def __init__(
        self,
        layer_sizes,
        FF_dropout,
        subnets_params,
        ENCODER_RNN_IS_BIDIRECTIONAL
    ):
        super().__init__()

        #-------#
        # for now, assume that the decoder has no proprietary layers
        N_outputs = subnets_params[-1].data_manifests['decoder_targets'].num_features
        #-------#

        # add a "projection"
        self.classifier_head = MultiLayerProjection(
            layer_sizes['encoder_rnn'][-1]*(1 + ENCODER_RNN_IS_BIDIRECTIONAL),
            layer_sizes['decoder_projection'],
            N_outputs, FF_dropout,
        )

    def forward(
        self, encoder_rnn_outputs, encoder_final_states, targets,
        natural_params_dict, image_dict
    ):
        '''
        encoder_rnn_outputs, targets, and natural_params_dict aren't necessary
        but are included here for consistency with DecoderAttentionRNN
        '''

        # use only hidden states, not cell states
        if isinstance(encoder_final_states, tuple):
            # LSTM
            hidden_states = encoder_final_states[0]
        else:
            hidden_states = encoder_final_states

        # "project" (expects a *list*) and expand to length-1 sequence
        natural_params_dict['decoder_targets'] = self.classifier_head(
            hidden_states)[:, None, :]

        ##############
        # targets, image_dict
        ##############


class DecoderRNN(nn.Module):
    def __init__(
        self,
        layer_sizes,
        FF_dropout,
        RNN_dropout,
        subnets_params,
        SOS_id,
        max_hyp_length,
        RNN_type
    ):
        super().__init__()

        # these are required at run time
        self.FF_dropout = FF_dropout
        self.RNN_dropout = RNN_dropout
        self.SOS_id = SOS_id
        self.max_hyp_length = max_hyp_length

        # generalizing beyond this would be a lot of work
        if len(np.unique(layer_sizes['decoder_rnn'])) > 1:
            raise NotImplementedError('Expected the same layer size for all layers')
        else:
            N_hidden = layer_sizes['decoder_rnn'][0]

        #-------#
        # for now, assume that the decoder has no proprietary layers
        N_outputs = subnets_params[-1].data_manifests['decoder_targets'].num_features
        #-------#

        # (possibly) multi-layer embedding
        self.embedding = MLLinearEmbedding(
            N_outputs, layer_sizes['decoder_embedding'], self.FF_dropout
        )
        
        ############
        # You could in theory just make the decoder like the encoder: *loop*
        #  across layers of the RNN and save all outputs.  E.g., you could
        #  imagine targeting different decoder layers....
        # If len()==1, this dropout will have no effect
        N_in = layer_sizes['decoder_embedding'][-1]
        N_layers_RNN = len(layer_sizes['decoder_rnn'])
        self.RNN = getattr(nn, RNN_type)(
            N_in, N_hidden, num_layers=N_layers_RNN, dropout=RNN_dropout
        )

        # flatten_parameters()
        ############

        # add a "projection"
        self.decoder_projection = MultiLayerProjection(
            layer_sizes['decoder_rnn'][-1], layer_sizes['decoder_projection'],
            N_outputs, self.FF_dropout,
        )

    def forward(
        self, encoder_rnn_outputs, encoder_final_states, targets,
        natural_params_dict, image_dict
    ):
        '''
        encoder_rnn_outputs aren't really necessary but are included here for
        consistency with DecoderAttentionRNN
        '''

        # Are we testing or training?
        if targets is None:
            # testing: use most probable prev. word as input; go one step at a time

            # encoder_rnn_outputs are in RNN ordering
            N_cases = encoder_rnn_outputs[-1].shape[1]

            # and the input to the embedding must have size (N_cases x 1)
            inputs = torch.full([N_cases, 1], self.SOS_id).to(
                encoder_rnn_outputs[-1].device
            )
            
            # loop
            states = encoder_final_states
            natural_params = []
            for i in range(self.max_hyp_length):
                one_step_natural_params, states = self.forward_core(inputs, states)
                natural_params.append(one_step_natural_params)
                _, most_probable_classes = one_step_natural_params.topk(1)
                inputs = most_probable_classes[:, :, 0].detach()
                
                # terminate if all most_probable_classes are EOS?

            final_states = states
            natural_params = torch.cat(natural_params, dim=1)
        else:
            # ...we're training; use *shifted* targets as inputs...
            X = targets.roll(1, 1)
            X[:, 0] = self.SOS_id
            natural_params, final_states = self.forward_core(X, encoder_final_states)

        # update the natural_params_dict
        natural_params_dict['decoder_targets'] = natural_params

    def forward_core(self, inputs, initial_state):

        # embed inputs
        X = self.embedding(inputs)

        # the original TF implementation used dropout at the RNN *inputs*
        X = F.dropout(
            X, self.RNN_dropout, training=self.training,
            inplace=True
        )

        # run through RNN, converting canonical to RNN ordering
        outputs, final_states = self.RNN(X.permute(1, 0, 2), initial_state)

        # "project" (expects a *list*) and convert back to canonical ordering
        natural_params = self.decoder_projection([outputs]).permute(1, 0, 2)

        return natural_params, final_states


class DecoderAttentionRNN(DecoderRNN):
    def __init__(
        self,
        layer_sizes,
        FF_dropout,
        RNN_dropout,
        subnets_params,
        SOS_id,
        max_hyp_length,
        RNN_type,
        N_hidden_attention=200,
    ):
        super().__init__(
            layer_sizes, FF_dropout, RNN_dropout, subnets_params, SOS_id,
            max_hyp_length, RNN_type,
        )

        # attention; Ns are necessarily the case
        N_hidden = layer_sizes['decoder_rnn'][0]
        N_outputs_RNN = N_hidden
        N_layers_RNN = len(layer_sizes['decoder_rnn'])
        self.attention = BahdanauAttention(
            N_hidden, N_outputs_RNN, N_hidden_attention, N_layers_RNN
        )
        
        # the RNN input is [embedded_input, "context"], so have to recreate it
        N_in = self.RNN.input_size + N_hidden
        self.RNN = getattr(nn, RNN_type)(
            N_in, N_hidden, num_layers=N_layers_RNN, dropout=RNN_dropout
        )

    def forward(
        self, encoder_rnn_outputs, encoder_final_states, targets,
        natural_params_dict, image_dict
    ):

        # encoder_rnn_outputs are in RNN ordering
        N_cases = encoder_rnn_outputs[-1].shape[1]
        iMax = targets.shape[1] if targets is not None else self.max_hyp_length
            
        # The decoder will compute attention for the outputs at layer l using
        #  the states at layer l.  Here we collect the outputs into a tensor of
        #  size (N_layers x T x N_cases x N_features)
        encoder_outputs = torch.stack(encoder_rnn_outputs[-self.RNN.num_layers:])
        
        # and the input to the embedding must have size (N_cases x 1)
        inputs = torch.full([N_cases, 1], self.SOS_id).to(
            encoder_outputs.device
        )
        states = encoder_final_states
        natural_params = []
        attn_weights = []

        for i in range(iMax):
            one_step_natural_params, states, one_step_attn_weights = self.forward_step(
                inputs, states, encoder_outputs
            )
            natural_params.append(one_step_natural_params)
            attn_weights.append(one_step_attn_weights)

            if targets is None:
                # testing: use most probable prev. word as input
                _, most_probable_classes = one_step_natural_params.topk(1)
                inputs = most_probable_classes[:, :, 0].detach()
            else:
                # training: use *actual* previous word as input
                inputs = targets[:, i, None]

        # (N_cases x T_out x N_out)
        natural_params_dict['decoder_targets'] = torch.cat(natural_params, dim=1)
        
        # just for plotting
        if not self.training:
            # (N_layers x Te x N_cases x Td) -> (N_cases x N_layers x Td x Te)
            attn_weights = torch.cat(attn_weights, dim=3).permute([2, 0, 3, 1])
            # attn_weights /= attn_weights.amax(dim=(2, 3), keepdim=True)
            attn_weights /= attn_weights.amax(dim=(2), keepdim=True)

            # final-layer attention only
            attention_images = torchvision.utils.make_grid(
                attn_weights[:, -1:, :, :]
            )
            # "average" over the duplicated single channel
            image_dict['attention'] = attention_images.mean(dim=0)

    def forward_step(self, inputs, states, encoder_outputs):
        ''' 
        inputs:             (N_cases x 1)
        states:             (N_layers x N_cases x N_hidden_RNN)
        encoder_outputs:    (N_layers x T x N_cases x N_out_encoder_RNN)
        '''

        # get attention weights
        if isinstance(states, tuple):
            # use only hidden state, not cell state
            queries = states[0][:, None, :, :]
        else:
            queries = states[:, None, :, :]
        context, attn_weights = self.attention(queries, encoder_outputs)

        # embed inputs---into size (N_cases x 1 x M)
        X = self.embedding(inputs)
        
        # concatenate "context" onto embedded input
        X = torch.cat((X, context[:, None, :]), dim=2)

        # the original TF implementation used dropout at the RNN *inputs*
        X = F.dropout(
            X, self.RNN_dropout, training=self.training, inplace=True
        )

        # run through RNN, converting canonical to RNN ordering
        outputs, new_states = self.RNN(X.permute(1, 0, 2), states)

        # "project" (expects a *list*) and convert back to canonical ordering
        natural_params = self.decoder_projection([outputs]).permute(1, 0, 2)

        # we return attn_weights only for plotting purposes
        return natural_params, new_states, attn_weights


class BahdanauAttention(nn.Module):
    def __init__(self, query_length, key_length, N_hidden, N_layers):
        super().__init__()

        self.Q_list = nn.ModuleList(
            nn.Linear(query_length, N_hidden) for _ in range(N_layers)
        )
        self.K_list = nn.ModuleList(
            nn.Linear(key_length, N_hidden) for _ in range(N_layers)
        )
        self.V_list = nn.ModuleList(
            nn.Linear(N_hidden, 1) for _ in range(N_layers)
        )
        self.N_layers = N_layers

    def forward(self, queries, keys):
        '''
        queries:    (N_layers x 1 x N_cases x query_length)
        keys:       (N_layers x T x N_cases x key_length)

        context:    (N_cases x key_length)
        weights:    (N_layers x T x N_cases x 1)
        '''

        # a tensor of size (N_layers x T x N_cases x 1)
        scores = torch.stack([
            V(torch.tanh(Q(query) + K(key))) for V, Q, K, query, key in zip(
                self.V_list, self.Q_list, self.K_list, queries, keys
            )
        ])
        
        # -> (N_layers*T x N_cases) to normalize over *both* layers and time
        N_cases = scores.shape[2]
        scores = scores.reshape([-1, N_cases])

        # normalize and compute convex combination of keys
        weights = F.softmax(scores, dim=0)
        weights = weights.reshape([self.N_layers, -1, N_cases, 1])

        # sum across N_layers, T -> (N_cases x key_length)
        context = torch.sum(weights*keys, dim=(0, 1))
        
        # we return the attention weights only for plotting purposes
        return context, weights


class MLLinearEmbedding(nn.Module):
    def __init__(
        self,
        N_in,
        layer_sizes,
        dropout,
    ):
        '''
        Really just a MLP, but with the first layer expecting one-hot inputs

        '''
        super().__init__()

        self.dropout = dropout

        self.layers = nn.ModuleList()
        for iLayer in range(len(layer_sizes)):
            N_out = layer_sizes[iLayer]
            Layer = nn.Embedding if iLayer == 0 else nn.Linear
            self.layers.append(Layer(N_in, N_out))
            N_in = N_out

    def forward(self, inputs):

        # embed inputs
        X = inputs
        for layer in self.layers:
            X = F.dropout(
                F.relu(layer(X)), self.dropout, training=self.training,
                inplace=False
            )

        return X


class MLConvEmbedding(nn.Module):
    def __init__(
        self,
        N_in,
        layer_sizes,
        subnet_params,
        dropout,
        MAX_POOL=False,
        VERBOSE=True,
    ):
        super().__init__()

        # useful things for `forward`
        self.decimation_factor = subnet_params.decimation_factor
        self.dropout = dropout

        # there may be multiple layers
        self.layers = nn.ModuleList()
        
        # distribute decimation over multiple layers
        layer_strides = close_factors(self.decimation_factor, len(layer_sizes))
        if VERBOSE:
            print('Temporally convolving with strides ' + repr(layer_strides))
        
        # construct embedding network
        for N_out, layer_stride in zip(layer_sizes, layer_strides):
            self.layers.append(nn.Conv1d(
                N_in, N_out, layer_stride, layer_stride,
                bias=MAX_POOL,
                padding='valid',
            ))
            N_in = N_out

            #########
            # max pool...
            #########

    def forward(self, inputs):

        # canonical ordering -> conv ordering, (N_cases x N_features x T)
        X = inputs.permute(0, 2, 1)

        # In 'VALID'-style convolution, the data are not padded to accommodate
        #  the filter, and the final (right-most) elements that don't fit a
        #  filter are simply dropped.  Here we pad by a sufficient amount to
        #  ensure that no data are dropped.  There's no danger in padding too
        #  much because we will subsequently extract out only sequences of the
        #  right inputs_lengths
        X = F.pad(X, [0, 4*self.decimation_factor])

        # "embed"
        for layer in self.layers:
            X = F.dropout(
                layer(X), self.dropout, training=self.training, inplace=True
            )

        # return in canonical ordering
        return X.permute(0, 2, 1)


class MultiLayerProjection(nn.Module):
    def __init__(
        self,
        N_inputs,
        Ns_hidden,
        N_outputs,
        FF_dropout,
        input_list_index=-1
    ):
        super().__init__()

        self.FF_dropout = FF_dropout
        self.input_list_index = input_list_index
        self.projections = nn.ModuleList()
        Ns_out = Ns_hidden + [N_outputs]
        N_in = N_inputs
        for N_out in Ns_out:
            self.projections.append(nn.Linear(N_in, N_out))
            N_in = N_out

    def forward(self, inputs):

        X = inputs[self.input_list_index]
        for projection in self.projections[:-1]:
            X = F.dropout(
                F.relu(projection(X)), self.FF_dropout, training=self.training,
                inplace=False
            )

        # no nonlinearity on the final layer--but there is dropout (!)
        #############
        ### return F.dropout(self.projections[-1](X), self.FF_dropout, training=self.training)
        return self.projections[-1](X)
        #############
        
        
def context_reshape(states, N_layers, N_directions=2):
    '''
    Pytorch RNNs return the states with size

        (N_layers*N_directions x N_cases x other).

    Thus, the two different directions of a bidirectional RNN are concatenated
    together along the *layers* dimension.  Now, to initialize a unidirectional
    RNN with 2*N_features from the final states of a bidirectional RNN with
    N_features, we need to unpack the first dimension, permute, and then
    flatten again:

        (N_layers x N_cases x N_directions*N_features)

    This function will also only select the last N_layers' worth of states, and
    so can be used to hook up encoder and decoder RNNs of different depths.
    Notice that if N_directions==1, then this is the *only* effect of this fxn.

    The concatention ordering of the RNN was taken from here:
        https://discuss.pytorch.org/t/
        how-can-i-know-which-part-of-h-n-of-bidirectional-rnn-is-for-backward-process/3883
    '''    

    # useful sizes
    _, N_cases, N_features = states.shape

    # break layers and directions into separate dimensions
    states = states.reshape([-1, N_directions, N_cases, N_features])

    # only grab the last N_layers' worth of states
    states = states[-N_layers:]

    # put directions next to features and flatten
    states = states.permute([0, 2, 1, 3])
    return torch.flatten(states, start_dim=2)


class SequenceTrainer():
    @auto_attribute(CHECK_MANIFEST=True)
    def __init__(
        self,
        manifest,
        subnets_params,
        #####
        # kwargs set in the manifest
        temperature=None,
        EMA_decay=None,
        beam_width=None,
        assessment_epoch_interval=None,
        tf_summaries_dir=None,
        #####
        N_cases=128,
        assessment_op_set={
            'decoder_word_error_rate',
            'decoder_accuracy',
            'loss'
        },
        REPORT_TRAINING_LOSS=True,
    ):

        class AssessmentTuple(MutableNamedTuple):
            __slots__ = (['decoder_word_error_rates'] + list(self.assessment_op_set))

        # create data loaders, tensorboard writers, assessments objects
        data_partitions = ['training', 'validation']
        self.loaders = dict.fromkeys(data_partitions)
        self.writers = dict.fromkeys(data_partitions)
        self.assessments = dict.fromkeys(data_partitions)
        for data_partition in data_partitions:

            # only assess on the *last* subject
            params = (
                subnets_params[-1:] if data_partition == 'validation'
                else subnets_params
            )
            self.loaders[data_partition] = TFRecordDataLoader(
                params, data_partition, N_cases
            )
            self.writers[data_partition] = SummaryWriter(
                log_dir=os.path.join(self.tf_summaries_dir, data_partition)
            )
            self.assessments[data_partition] = AssessmentTuple(
                decoder_word_error_rates=None,
                **dict.fromkeys(self.assessment_op_set)
            )

    def train_and_assess(self, N_epochs, sequence_net, device):

        ########
        # temporary hack
        # self.assessment_epoch_interval = N_epochs - 1
        ########

        # init
        optimizer = torch.optim.Adam(sequence_net.parameters(), lr=3e-4)
        N_assessments = math.ceil(N_epochs/self.assessment_epoch_interval)+1
        for assessment in self.assessments.values():
            assessment.decoder_word_error_rates = np.zeros((N_assessments))
        sequence_net.to(device)

        def batch_op_core(
            device_batch, natural_params_dict, loss_fxn_dict, epoch_loss_dict
        ):

            # overkill but organized this way for "elegance"
            metadata_dicts = dict.fromkeys(['encoder', 'decoder'])
            for coder in metadata_dicts.keys():

                metadata_dicts[coder] = {}
                if coder == 'encoder':
                    d = sequence_net.encoder.decimation_factors[device_batch['subnet_id']]
                    inds, lens = sequences_tools(device_batch['encoder_inputs'][:, ::d, :])
                    metadata_dicts[coder]['decimation_factor'] = d
                else:
                    inds, lens = sequences_tools(device_batch['decoder_targets'])
                metadata_dicts[coder]['indices'] = inds
                metadata_dicts[coder]['lengths'] = lens

            # compute losses
            complete_loss = 0
            for key, natural_params in natural_params_dict.items():

                # assemble the targets, their indices, and lengths
                coder = key.split('_')[0]
                targets = device_batch[key]
                indices = metadata_dicts[coder]['indices']
                lengths = metadata_dicts[coder]['lengths']

                # *encoder* targets are decimated and possibly reversed
                if coder == 'encoder':
                    d = metadata_dicts[coder]['decimation_factor']
                    targets = targets[:, ::d, :]
                    if sequence_net.decoder_type == 'final_state_coupled':
                        targets = reverse_sequences(targets, indices, lengths)

                # accumulate loss
                complete_loss += penalize_RNN(
                    natural_params, targets, indices,
                    *loss_fxn_dict[key], epoch_loss_dict, key
                )

            return complete_loss

        for epoch in range(N_epochs):

            # with profile(
            #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     record_shapes=True
            # ) as prof:
            #     with record_function("model_inference"):
            self.batch_train(batch_op_core, sequence_net, optimizer, epoch, device)
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            # validate
            if (epoch % self.assessment_epoch_interval) == 0:
                # clear output only when ready to print new validation results
                clear_output(wait=True)
                for data_partition in ['validation', 'training']:
                    self.batch_assess(
                        batch_op_core, sequence_net, data_partition, epoch, device
                    )

                    # store
                    self.assessments[data_partition].decoder_word_error_rates[
                        epoch//self.assessment_epoch_interval
                    ] = self.assessments[data_partition].decoder_word_error_rate

        # for backward compatibility with unitrain
        return self.assessments

    def batch_train(self, batch_op_core, net, optimizer, epoch, device):
        '''
        Train for one epoch (all batches)
        '''

        net.train()
        N_examples = 0
        epoch_loss_dict = {}
        for batch in self.loaders['training']:

            # put the data on (presumably) the GPU and pass thru network
            device_batch = {
                key: val.decode() if key == 'subnet_id'
                else torch.tensor(val).to(device)
                for key, val in batch.items()
            }
            natural_params_dict, image_dict = net(
                device_batch['encoder_inputs'], device_batch['subnet_id'],
                device_batch['decoder_targets'][:, :, 0]
            )

            # accumulate total number of examples
            N_examples += device_batch['encoder_inputs'].shape[0]

            # ...
            optimizer.zero_grad()
            loss = batch_op_core(
                device_batch, natural_params_dict,
                net.loss_fxn_dicts[device_batch['subnet_id']], epoch_loss_dict
            )

            # backprop and take a step downhill
            loss.backward()
            optimizer.step()

        # print the per-example loss
        if self.REPORT_TRAINING_LOSS:
            loss_string = ' '.join([
                '%s: %.2e' % (loss_name, loss_value/N_examples)
                for loss_name, loss_value in epoch_loss_dict.items()
            ])
            print('\n[ training ]  epoch: %3i  %s' % (epoch, loss_string), end='\t')

    def batch_assess(self, batch_op_core, net, data_partition, epoch, device):
        '''
        Assess on this data_partition
        '''

        # init
        net.eval()
        N_examples = 0
        epoch_WER = 0
        epoch_loss_dict = {}
        on_clr = 'on_yellow' if data_partition == 'training' else 'on_cyan'
        with torch.no_grad():
            for batch in self.loaders[data_partition]:

                # put the data on (presumably) the GPU and pass thru network
                device_batch = {
                    key: val.decode() if key == 'subnet_id'
                    else torch.tensor(val).to(device)
                    for key, val in batch.items()
                }

                # put the data on (presumably) the GPU and pass thru network; 
                # DO NOT PASS TARGETS
                natural_params_dict, image_dict = net(
                    device_batch['encoder_inputs'], device_batch['subnet_id']
                )

                # accumulate total number of examples
                N_examples += device_batch['encoder_inputs'].shape[0]

                # update losses in epoch_loss_dict
                batch_op_core(
                    device_batch, natural_params_dict,
                    net.loss_fxn_dicts[device_batch['subnet_id']], epoch_loss_dict
                )

                # only consider the single sequence of most probable classes
                _, most_probable_classes = natural_params_dict['decoder_targets'].topk(1)
                if net.TARGETS_ARE_SEQUENCES:
                    most_probable_classes = terminate_sequences(
                        most_probable_classes, net.EOS_id, net.pad_id
                    )

                # accumulate word error rates
                WERs = get_word_error_rate(
                    device_batch['decoder_targets'], most_probable_classes
                )
                epoch_WER += sum(WERs).item()

                # ...
                net.print_sentences(
                    most_probable_classes, device_batch['decoder_targets'], on_clr
                )

                # just evaluate on a single batch
                break
            else:
                raise ValueError('No %s data!' % data_partition)

        # report cross entropy(s)
        print('[assessment]  epoch: %3i ' % epoch, end='')
        for loss_name, loss_value in epoch_loss_dict.items():

            # divide cumulative errors by number of examples
            per_example_loss = loss_value/N_examples

            # print to screen and write to tensorboard
            print(' %s: %.2e' % (loss_name, per_example_loss), end='')
            self.writers[data_partition].add_scalar(
                'summarize_%s' % loss_name, per_example_loss, epoch
            )
        for image_name, image in image_dict.items():
            self.writers[data_partition].add_image(
                'image_name', image, dataformats='HW',
                # max_outputs=16
            )

        # report WER(s)
        per_example_WER = epoch_WER/N_examples
        print(
            ' WER: %1.3f' % per_example_WER,
            ' (%s data)' % data_partition,
            end=''
        )
        self.writers[data_partition].add_scalar(
            'summarize_decoder_word_error_rate', per_example_WER, epoch
        )
        self.writers[data_partition].flush()

        ############
        # hard-coded; generally, these might not be in the assessments
        # store the assessments
        self.assessments[data_partition].loss = per_example_loss
        self.assessments[data_partition].decoder_word_error_rate = per_example_WER
        # fake it
        self.assessments[data_partition].decoder_accuracy = 1 - per_example_WER
        ############


def penalize_RNN(
    natural_params, targets, targets_indices, loss_fxn, penalty_scale,
    epoch_loss_dict, key
):

    # compute loss
    loss = loss_fxn(natural_params[targets_indices], targets[targets_indices])

    # *accumulate* loss
    CE_key = swap(key, 'cross_entropy')
    if CE_key not in epoch_loss_dict:
        epoch_loss_dict[CE_key] = 0
    epoch_loss_dict[CE_key] += loss.item()
    
    return penalty_scale*loss


def class_indices_to_sequence(classes, targets_list, EOS_token, pad_token):
    word_sequence = ''.join([targets_list[c] for c in classes]).replace(
        '_', ' ').replace(pad_token, '').replace(EOS_token, '').rstrip()

    return word_sequence


def terminate_sequences(sequence_tensor, EOS_id, pad_id):

    # Create matrix like [[0, 1, 2, 3], [0, 1, 2, 3]]
    all_inds = torch.arange(sequence_tensor.shape[1]).tile(
        (sequence_tensor.shape[0], 1)
    ).to(sequence_tensor.device)

    # In the worst case, no EOS_id occurs in a hypothesis; mark this explicitly
    #  (which allows the argmax to work).  (Technically, this could fix a
    #  sentence that should have had an EOS_id at max_hyp_length but didn't.)
    #  NB that argmax returns the *first* instance where there are multiple.
    sequence_tensor[:, -1, :] = EOS_id
    final_inds = torch.argmax((sequence_tensor == 1).to(dtype=torch.int), dim=1)

    # now write pad_id into all entries of sequence_tensor beyond first EOS_id
    sequence_tensor = torch.where(
        all_inds[:, :, None] > final_inds[:, None, :], pad_id, sequence_tensor
    )

    return sequence_tensor


def get_cross_entropy_fxn(distribution):
    '''
    Get the cross entropy function appropriate to this distribution.  NB that
    these functions  *sum* across all examples.

    Cross entropies are computed in bits.

    However, the Gaussian cross entropy is off by an additive constant (the log
    normalizer)

    Also NB: no lambda functions for compatibility with pickle
    '''

    if distribution == 'Gaussian':
        return Gaussian_cross_entropy
    elif distribution == 'categorical':
        return categorical_cross_entropy
    else:
        raise NotImplementedError('%s cross entropy not yet implemented!' % distribution)


def Gaussian_cross_entropy(natural_params, targets):
    # in TF1, you averaged across features
    return np.log2(np.e)*nn.MSELoss(reduction='sum')(natural_params, targets)


def categorical_cross_entropy(natural_params, targets):
    # Generic ints don't work so convert to int64.  Also, your code expects
    #  targets to have a final dimension of 1; CrossEntropyLoss does not. 
    return nn.CrossEntropyLoss(reduction='sum')(
        natural_params, targets[:, 0].to(torch.int64)
    )


def swap(key, string):
    # In SequenceNetworks, keys are often constructed from the data_manifest
    #  key by swapping out the word 'targets' for some other string.  This is
    #  just a shortcut for that process.
    return key.replace('targets', string)
