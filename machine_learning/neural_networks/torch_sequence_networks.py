# standard libraries
import pdb
from termcolor import cprint
from IPython.display import clear_output
import os

# third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#######
# from torch.profiler import profile, record_function, ProfilerActivity
#######

# local
from machine_learning.data_mungers import TFRecordDataLoader
from machine_learning.torch_helpers import (
    get_word_error_rate, sequences_tools, reverse_sequences
)
from utils_jgm.toolbox import auto_attribute, wer_vector
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
# (17) 
# (N) Try using powers of 2 for all layer sizes....
###############


'''
Data orderings:
    canonical:          (batch_size x T x N_features)
    Conv1d:             (batch_size x N_features x T)
    Embedding input:    (*)
    Embedding output:   (*, N_features)
    RNN inputs:         (T x batch_size x N_features)    
    RNN outputs:        (T x batch_size x N_features)
    RNN states:         (N_layers*N_directions x batch_size x N_features)
'''


class Sequence2Sequence(nn.Module):
    @auto_attribute(CHECK_MANIFEST=True)
    def __init__(
        self,
        manifest,
        subnet_params,
        #######
        layer_sizes=None,
        FF_dropout=None,
        RNN_dropout=None,
        #######
        ENCODER_RNN_IS_BIDIRECTIONAL=True,
        training_GPUs=None,
        EOS_token='<EOS>',
        pad_token='<pad>',
        max_hyp_length=20,
        coupling='final_state',
        RNN_type='LSTM',
    ):
        super().__init__()

        # useful subnet_params
        self.decoder_target_list = subnet_params.data_manifests[
            'decoder_targets'].get_feature_list()
        self.EOS_token = EOS_token
        self.pad_token = pad_token

        self.EOS_id = self.decoder_target_list.index(self.EOS_token)
        self.pad_id = self.decoder_target_list.index(self.pad_token)

        print('USING %s in the RNNs' % RNN_type)

        # ENCODER
        N_inputs = subnet_params.data_manifests['encoder_inputs'].num_features
        self.encoder = EncoderRNN(
            N_inputs, self.layer_sizes, self.FF_dropout, self.RNN_dropout,
            ENCODER_RNN_IS_BIDIRECTIONAL, subnet_params, RNN_type
        )

        # reshape the context to pass to the decoder
        self.reshape_context = lambda states: context_reshape(
            states, len(self.layer_sizes['decoder_rnn']),
            ENCODER_RNN_IS_BIDIRECTIONAL+1
        )

        # DECODER
        match coupling:
            case 'final_state':
                Decoder = DecoderRNN
            case 'attention':
                Decoder = DecoderAttentionRNN
            case _:
                raise ValueError('Unrecognized decoder_type')
        N_outputs = subnet_params.data_manifests['decoder_targets'].num_features
        self.decoder = Decoder(
            N_outputs, self.layer_sizes, self.FF_dropout, self.RNN_dropout,
            self.EOS_id,  # use EOS as SOS, as in the TF1 version
            max_hyp_length, RNN_type
        )

        # accumulate the loss functions
        self.loss_fxn_dict = {}
        for key in subnet_params.data_mapping:
            if key.endswith('targets'):

                data_manifest = subnet_params.data_manifests[key]
                self.loss_fxn_dict[key] = (
                    get_cross_entropy_fxn(data_manifest.distribution),
                    data_manifest.penalty_scale
                )

    def forward(self, inputs, targets=None):
        '''
        The inputs, targets, and natural_params have JGM "canonical ordering,"

            (batch_size x T x N_features) and (batch_size x T)

        The RNN outputs have RNN ordering,

            (T x batch_size x N_features)

        and the RNN states ("similarly") have shape

            (N_layers*N_directions x batch_size x N_features)
        '''

        # encode; project outputs; init decoder state; decode; project outputs
        encoder_rnn_outputs, encoder_final_states, natural_params_dict = self.encoder(inputs)
        if isinstance(encoder_final_states, tuple):
            context = tuple(
                self.reshape_context(states) for states in encoder_final_states
            )
        else:
            context = self.reshape_context(encoder_final_states)
        natural_params_dict['decoder_targets'] = self.decoder(
            encoder_rnn_outputs, context, targets
        )

        return natural_params_dict

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
                predicted_classes, self.decoder_target_list,
                self.EOS_token, self.pad_token
            )
            target_words = class_indices_to_sequence(
                target_classes, self.decoder_target_list,
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


class EncoderRNN(nn.Module):
    def __init__(
        self,
        N_inputs,
        layer_sizes,
        FF_dropout, 
        RNN_dropout,
        BIDIRECTIONAL,
        subnet_params,
        RNN_type,
    ):
        super().__init__()
        
        # ...
        if len(np.unique(layer_sizes['encoder_rnn'])) > 1:
            raise NotImplementedError('Expected the same layer size for all layers')
        else:
            N_hidden = layer_sizes['encoder_rnn'][0]

        self.FF_dropout = FF_dropout
        self.RNN_dropout = RNN_dropout

        # hard-coded
        conv_stride = subnet_params.decimation_factor
        conv_kernel_width = conv_stride

        # the "convolutional embedding"
        N_in = N_inputs
        self.embeddings = nn.ModuleList()
        for N_out in layer_sizes['encoder_embedding']:
            self.embeddings.append(nn.Conv1d(
                N_in, N_out, conv_kernel_width, conv_stride,
                bias=False,  # to match TF1 version
                padding='valid',
            ))
            N_in = N_out

        # the RNN; you may need outputs from intermediate layers, so you have
        #  to construct this one layer at a time
        self.RNNs = nn.ModuleList()
        RNN = getattr(nn, RNN_type)
        for N_hidden in layer_sizes['encoder_rnn']:
            self.RNNs.append(
                RNN(N_in, N_hidden, num_layers=1, bidirectional=BIDIRECTIONAL)
            )
            N_in = N_hidden*(1 + BIDIRECTIONAL)
        self.conv_stride = conv_stride

        # accumlate any "projections" and corresponding loss functions
        self.encoder_projections = nn.ModuleDict()
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

                # accumulate the encoder projections
                self.encoder_projections[key] = RNNProjection(
                    N_inputs, Ns_hidden, N_outputs, self.FF_dropout,
                    input_list_index=output_layer,
                )

        ###############
        # flatten_parameters()
        ###############

    def forward(self, inputs):
        # get lengths of *downsampled* input sequences
        inputs_indices, inputs_lengths = sequences_tools(inputs[:, ::self.conv_stride, :])

        # canonical ordering -> conv ordering, (batch_size x N_features x T)
        X = inputs.permute(0, 2, 1)

        # In 'VALID'-style convolution, the data are not padded to accommodate
        #  the filter, and the final (right-most) elements that don't fit a
        #  filter are simply dropped.  Here we pad by a sufficient amount to
        #  ensure that no data are dropped.  There's no danger in padding too
        #  much because we will subsequently extract out only sequences of the
        #  right inputs_lengths
        X = F.pad(X, [0, 4*self.conv_stride])

        # "embed"
        for embedding in self.embeddings:
            X = F.dropout(
                embedding(X), self.FF_dropout, training=self.training,
                inplace=True
            )

        # put into canonical ordering so we can reverse (a la Sutskever 2014)
        X = reverse_sequences(X.permute(0, 2, 1), inputs_indices, inputs_lengths)

        # the original TF version used dropout on the RNN *inputs*
        X = F.dropout(
            X, self.RNN_dropout, training=self.training,
            inplace=True
        )

        # put into RNN ordering (T x batch_size x N_features) and "pack"
        X = nn.utils.rnn.pack_padded_sequence(
            X.permute(1, 0, 2), inputs_lengths.to('cpu'), enforce_sorted=False
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
        natural_params_dict = {}
        for key, projection in self.encoder_projections.items():
            natural_params_dict[key] = projection(all_outputs).permute(1, 0, 2)

        return all_outputs, all_final_states, natural_params_dict


class DecoderRNN(nn.Module):
    def __init__(
        self,
        N_outputs,
        layer_sizes,
        FF_dropout,
        RNN_dropout,
        SOS_id,
        max_hyp_length,
        RNN_type
    ):
        super().__init__()
        
        print('COUPLING ENCODER TO DECODER WITH FINAL HIDDEN STATE')

        # generalizing beyond this would be a lot of work
        if len(np.unique(layer_sizes['decoder_rnn'])) > 1:
            raise NotImplementedError('Expected the same layer size for all layers')
        else:
            N_hidden = layer_sizes['decoder_rnn'][0]

        # ...
        self.FF_dropout = FF_dropout
        self.RNN_dropout = RNN_dropout

        # embed (followed perhaps by linear layers)
        N_in = N_outputs
        self.embeddings = nn.ModuleList()
        for iLayer in range(len(layer_sizes['decoder_embedding'])):
            N_out = layer_sizes['decoder_embedding'][iLayer]
            Layer = nn.Embedding if iLayer == 0 else nn.Linear
            self.embeddings.append(Layer(N_in, N_out))
            N_in = N_out

        ##############
        # You could in theory just make the decoder like the encoder: *loop*
        #  across layers of the RNN and save all outputs.  E.g., you could
        #  imagine targeting different decoder layers....
        ##############
        # If len()==1, this dropout will have no effect
        RNN = getattr(nn, RNN_type)
        self.RNN = RNN(
            N_in, N_hidden, num_layers=len(layer_sizes['decoder_rnn']),
            dropout=RNN_dropout
        )
        ##############

        # ...
        self.SOS_id = SOS_id
        self.max_hyp_length = max_hyp_length

        ###############
        # flatten_parameters()
        ###############

        # add a "projection"
        self.decoder_projection = RNNProjection(
            layer_sizes['decoder_rnn'][-1], layer_sizes['decoder_projection'],
            N_outputs, self.FF_dropout,
        )

    def forward(self, encoder_rnn_outputs, encoder_final_states, targets=None):
        '''
        Encoder_outputs isn't really necessary; here for generality w/attention
        '''

        # Are we testing or training?
        if targets is None:
            # testing: use most probable prev. word as input; go one step at a time

            # encoder_rnn_outputs are in RNN ordering
            batch_size = encoder_rnn_outputs[-1].shape[1]

            # and the input to the embedding must have size (batch_size x 1)
            inputs = torch.full([batch_size, 1], self.SOS_id).to(
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

        # ...
        return natural_params

    def forward_core(self, inputs, initial_state):

        # embed inputs
        X = inputs
        for embedding in self.embeddings:
            X = F.dropout(
                F.relu(embedding(X)), self.FF_dropout, training=self.training,
                inplace=False
            )

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


class DecoderAttentionRNN(nn.Module):
    def __init__(
        self,
        N_outputs,
        layer_sizes,
        FF_dropout,
        RNN_dropout,
        SOS_id,
        max_hyp_length,
        RNN_type,
        N_hidden_attention=200,
    ):
        super().__init__()

        print('COUPLING ENCODER TO DECODER WITH ATTENTION')

        # generalizing beyond this would be a lot of work
        if len(np.unique(layer_sizes['decoder_rnn'])) > 1:
            raise NotImplementedError('Expected the same layer size for all layers')
        else:
            N_hidden = layer_sizes['decoder_rnn'][0]

        # this is necessarily the case
        N_outputs_RNN = N_hidden
        N_layers_RNN = len(layer_sizes['decoder_rnn'])

        # ...
        self.FF_dropout = FF_dropout
        self.RNN_dropout = RNN_dropout

        # embed (followed perhaps by linear layers)
        N_in = N_outputs
        self.embeddings = nn.ModuleList()
        for iLayer in range(len(layer_sizes['decoder_embedding'])):
            N_out = layer_sizes['decoder_embedding'][iLayer]
            Layer = nn.Embedding if iLayer == 0 else nn.Linear
            self.embeddings.append(Layer(N_in, N_out))
            N_in = N_out

        # (query_length, key_length, N_hidden_attention, N_layers_decoder_RNN)
        self.attention = BahdanauAttention(
            N_hidden, N_outputs_RNN, N_hidden_attention, N_layers_RNN
        )
        
        ##############
        # You could in theory just make the decoder like the encoder: *loop*
        #  across layers of the RNN and save all outputs.  E.g., you could
        #  imagine targeting different decoder layers....
        ##############
        # If len()==1, this dropout will have no effect
        RNN = getattr(nn, RNN_type)
        self.RNN = RNN(
            N_in + N_hidden, N_hidden, num_layers=N_layers_RNN,
            dropout=RNN_dropout
        )
        ##############

        # ...
        self.SOS_id = SOS_id
        self.max_hyp_length = max_hyp_length

        ###############
        # flatten_parameters()
        ###############

        # add a "projection"
        self.decoder_projection = RNNProjection(
            layer_sizes['decoder_rnn'][-1], layer_sizes['decoder_projection'],
            N_outputs, self.FF_dropout,
        )

    def forward(self, encoder_rnn_outputs, encoder_final_states, targets=None):

        # encoder_rnn_outputs are in RNN ordering
        batch_size = encoder_rnn_outputs[-1].shape[1]
        iMax = targets.shape[1] if targets is not None else self.max_hyp_length
            
        # The decoder will compute attention for the outputs at layer l using
        #  the states at layer l.  Here we collect the outputs into a tensor of
        #  size (N_layers x T x batch_size x N_features)
        encoder_outputs = torch.stack(encoder_rnn_outputs[-self.RNN.num_layers:])
        
        # and the input to the embedding must have size (batch_size x 1)
        inputs = torch.full([batch_size, 1], self.SOS_id).to(
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

        # (batch_size x T_out x N_out)
        natural_params = torch.cat(natural_params, dim=1)
        
        # just for plotting
        if not self.training:
            # (N_layers x batch_size x T_in x T_out)
            attn_weights = torch.cat(attn_weights, dim=3).permute([0, 2, 1, 3])

        return natural_params

    def forward_step(self, inputs, states, encoder_outputs):
        ''' 
        inputs:             (batch_size x 1)
        states:             (N_layers x batch_size x N_hidden_RNN)
        encoder_outputs:    (N_layers x T x batch_size x N_out_encoder_RNN)
        '''

        # get attention weights
        if isinstance(states, tuple):
            # use only hidden state, not cell state
            queries = states[0][:, None, :, :]
        else:
            queries = states[:, None, :, :]
        context, attn_weights = self.attention(queries, encoder_outputs)

        # embed inputs---into size (batch_size x 1 x M)
        X = inputs
        for embedding in self.embeddings:
            X = F.dropout(
                F.relu(embedding(X)), self.FF_dropout, training=self.training,
                inplace=False
            )

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
        queries:    (N_layers x 1 x batch_size x query_length)
        keys:       (N_layers x T x batch_size x key_length)

        context:    (batch_size x key_length)
        weights:    (N_layers x T x batch_size x 1)
        '''

        # a tensor of size (N_layers x T x batch_size x 1)
        scores = torch.stack([
            V(torch.tanh(Q(query) + K(key))) for V, Q, K, query, key in zip(
                self.V_list, self.Q_list, self.K_list, queries, keys
            )
        ])
        
        # -> (N_layers*T x batch_size) to normalize over *both* layers and time
        batch_size = scores.shape[2]
        scores = scores.reshape([-1, batch_size])

        # normalize and compute convex combination of keys
        weights = F.softmax(scores, dim=0)
        weights = weights.reshape([self.N_layers, -1, batch_size, 1])

        # sum across N_layers, T -> (batch_size x key_length)
        context = torch.sum(weights*keys, dim=(0, 1))
        
        return context, weights


class RNNProjection(nn.Module):
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
        # return F.dropout(self.projections[-1](X), self.FF_dropout, training=self.training)
        return self.projections[-1](X)
        #############
        
        
def context_reshape(states, N_layers, N_directions=2):
    '''
    Pytorch RNNs return the states with size

        (N_layers*N_directions x batch_size x other).

    Thus, the two different directions of a bidirectional RNN are concatenated
    together along the *layers* dimension.  Now, to initialize a unidirectional
    RNN with 2*N_features from the final states of a bidirectional RNN with
    N_features, we need to unpack the first dimension, permute, and then
    flatten again:

        (N_layers x batch_size x N_directions*N_features)

    This function will also only select the last N_layers' worth of states, and
    so can be used to hook up encoder and decoder RNNs of different depths.
    Notice that if N_directions==1, then this is the *only* effect of this fxn.

    The concatention ordering of the RNN was taken from here:
        https://discuss.pytorch.org/t/
        how-can-i-know-which-part-of-h-n-of-bidirectional-rnn-is-for-backward-process/3883
    '''    

    # useful sizes
    _, batch_size, N_features = states.shape

    # break layers and directions into separate dimensions
    states = states.reshape([-1, N_directions, batch_size, N_features])

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
    ):

        ##########
        batch_size = 128
        ##########

        # tfrecord_pipe = TFRecordPipe(subnet_params)
        # self.training_loader = torch.utils.data.DataLoader(
        #     tfrecord_pipe.construct_pipe(),
        #     batch_size=batch_size,
        #     # shuffle=True,
        #     # num_workers=8,
        #     num_workers=1,
        #     pin_memory=True,
        #     collate_fn=tfrecord_pipe.pad_collate,
        # )

        data_partitions = ['training', 'validation']
        self.loaders = dict.fromkeys(data_partitions)
        self.writers = dict.fromkeys(data_partitions)
        for data_partition in data_partitions:
            self.loaders[data_partition] = TFRecordDataLoader(
                subnets_params, data_partition, batch_size
            )
            self.writers[data_partition] = SummaryWriter(
                log_dir=os.path.join(self.tf_summaries_dir, data_partition)
            )

        # for key in manifest['data_mapping']:
        #     if key.endswith('targets'):

    def train_and_assess(self, N_epochs, sequence_net, device):

        optimizer = torch.optim.Adam(sequence_net.parameters(), lr=3e-4)

        def batch_op_core(
            encoder_inputs, decoder_targets, batch, natural_params_dict, 
            loss_fxn_dict, epoch_loss_dict
        ):

            # get targets indices/lengths
            ######
            # This is redundant but the alternative is to pass them around....
            encoder_targets_indices, encoder_targets_lengths = sequences_tools(
                encoder_inputs[:, ::sequence_net.encoder.conv_stride, :]
            )
            ######
            decoder_targets_indices, _ = sequences_tools(decoder_targets)

            # compute losses
            complete_loss = 0
            for key, natural_params in natural_params_dict.items():

                # get and reverse *encoder* targets
                if key.startswith('encoder'):
                    targets = torch.tensor(batch[key]).to(device)
                    targets_indices = encoder_targets_indices
                    targets = reverse_sequences(
                        targets[:, ::sequence_net.encoder.conv_stride, :],
                        encoder_targets_indices, encoder_targets_lengths
                    )
                else:
                    targets = decoder_targets
                    targets_indices = decoder_targets_indices

                # accumulate loss
                complete_loss += penalize_RNN(
                    natural_params, targets, targets_indices,
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

            print()

    def batch_train(self, batch_op_core, net, optimizer, epoch, device):
        '''
        Train for one epoch (all batches)
        '''

        net.train()
        N_examples = 0
        epoch_loss_dict = {}
        for batch in self.loaders['training']:

            # put the data on the GPU and pass thru network
            encoder_inputs = torch.tensor(batch['encoder_inputs']).to(device)
            decoder_targets = torch.tensor(batch['decoder_targets']).to(device)
            natural_params_dict = net(encoder_inputs, decoder_targets[:, :, 0])

            # accumulate total number of examples
            N_examples += encoder_inputs.shape[0]

            # ...
            optimizer.zero_grad()
            loss = batch_op_core(
                encoder_inputs, decoder_targets, batch, natural_params_dict, 
                net.loss_fxn_dict, epoch_loss_dict
            )

            # backprop and take a step downhill
            loss.backward()
            optimizer.step()

        # print the per-example loss
        loss_string = ' '.join([
            '%s: %.2e' % (loss_name, loss_value/N_examples)
            for loss_name, loss_value in epoch_loss_dict.items()
        ])
        print('[ training ]  epoch: %3i  %s' % (epoch, loss_string), end='\t')

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

                # put the data on the GPU and pass thru network
                encoder_inputs = torch.tensor(batch['encoder_inputs']).to(device)
                natural_params_dict = net(encoder_inputs)

                # accumulate total number of examples
                N_examples += encoder_inputs.shape[0]

                # update losses in epoch_loss_dict
                decoder_targets = torch.tensor(batch['decoder_targets']).to(device)
                batch_op_core(
                    encoder_inputs, decoder_targets, batch, natural_params_dict, 
                    net.loss_fxn_dict, epoch_loss_dict
                )

                # only consider the single sequence of most probable classes
                _, most_probable_classes = natural_params_dict['decoder_targets'].topk(1)
                most_probable_classes = terminate_sequences(
                    most_probable_classes, net.EOS_id, net.pad_id
                )
                WERs = get_word_error_rate(decoder_targets, most_probable_classes)
                epoch_WER += sum(WERs).item()

                # ...
                net.print_sentences(most_probable_classes, decoder_targets, on_clr)

                # just evaluate on a single batch
                break

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
    '''

    if distribution == 'Gaussian':
        # in TF1, you averaged across features
        return lambda natural_params, targets: np.log2(np.e)*nn.MSELoss(reduction='sum')(
            natural_params, targets)
    elif distribution == 'categorical':
        # Generic ints don't work so convert to int64.  Also, your code expects
        #  targets to have a final dimension of 1; CrossEntropyLoss does not. 
        return lambda natural_params, targets: nn.CrossEntropyLoss(reduction='sum')(
            natural_params, targets[:, 0].to(torch.int64)
        )
    else:
        raise NotImplementedError('%s cross entropy not yet implemented!' % distribution)


def swap(key, string):
    # In SequenceNetworks, keys are often constructed from the data_manifest
    #  key by swapping out the word 'targets' for some other string.  This is
    #  just a shortcut for that process.
    return key.replace('targets', string)
