# standard libraries
import pdb

# third-party packages
import tensorflow as tf
from tensor2tensor.layers import common_layers

# local
from . import tf_helpers as tfh

'''
A collection of methods for neural networks built in tensorflow.

:Author: J.G. Makin (except where otherwise noted)

Created: June 2017
'''


def bias_decorator(preactivation_fxn):
    def bias_wrapper(inputs, Nin, Nout, **kwargs):
        USE_BIASES = kwargs.get('USE_BIASES', True)
        stiffness = kwargs.get('stiffness', 0)
        preactivation = preactivation_fxn(inputs, Nin, Nout, **kwargs)
        if USE_BIASES:
            biases = create_biases([Nout], stiffness=stiffness)
            preactivation += biases
        return preactivation

    return bias_wrapper


@bias_decorator
def tf_matmul_wrapper(
        inputs, Nin, Nout, stiffness=0, transpose_b=False, num_shards=None,
        USE_BIASES=True):
    wts_shape = (Nout, Nin) if transpose_b else (Nin, Nout)
    wts = create_weights(wts_shape, stiffness=stiffness, num_shards=num_shards)
    if (common_layers.shape_list(inputs)[-1] == 1) and (Nin != 1):
        # Shortcut notation: the inputs may be indices in lieu of a one-hot
        #  representation.  In that case, rather than tf.matmul'ing by the
        #  weight matrix, just extract the indexed column with tf.gather.
        return tf.gather(wts, tf.reshape(inputs, [-1]))
    else:
        return tf.matmul(inputs, wts, transpose_b=transpose_b)


@bias_decorator
def tf_conv2d_wrapper(
        inputs, Nin, Nout, name, stiffness=0, filter_height=1, filter_width=1,
        strides=[1, 1, 1, 1], num_shards=None, USE_BIASES=True):
    wts_shape = [filter_height, filter_width, Nin, Nout]
    wts = create_weights(wts_shape, stiffness=stiffness, num_shards=num_shards)
    preactivations = tf.nn.conv2d(input=inputs, filters=wts, strides=strides, padding='VALID')
    ### provide the name to conv2d??  otherwise eliminate....
    return preactivations


def tf_max_pool_wrapper(inputs, name, ksize, strides):
    return tf.nn.max_pool2d(
        input=inputs, name=name, ksize=ksize, strides=strides, padding='VALID')


def tf_avg_pool_wrapper(inputs, name, ksize, strides):
    return tf.nn.avg_pool2d(
        value=inputs, name=name, ksize=ksize, strides=strides, padding='VALID')


def feed_forward_multi_layer(
        get_activations, Ninputs, layer_sizes, dropout_rate, net_name,
        preactivation_fxns=None, activation_fxns=None):

    Nlayers = len(layer_sizes)

    # fill in in default lists
    if not preactivation_fxns:
        preactivation_fxns = [tf_matmul_wrapper]*Nlayers
    if not activation_fxns:
        activation_fxns = [tf.nn.relu]*Nlayers

    for iLayer, (Noutputs, preactivation_fxn, activation_fxn) in enumerate(zip(
                    layer_sizes, preactivation_fxns, activation_fxns)):

        # for consistency w/t2t translation model, swap for output projections
        wts_shape = (Ninputs, Noutputs)
        if hasattr(preactivation_fxn, 'TRANSPOSED'):
            if preactivation_fxn.TRANSPOSED:
                wts_shape = (Noutputs, Ninputs)
        layer_name = '%s_%i_%i_%i' % (net_name, *wts_shape, iLayer)

        # now run through a single layer
        get_activations = feed_forward_one_layer(
            get_activations, layer_name, Nin=Ninputs, Nout=Noutputs,
            preactivation_fxn=preactivation_fxn, activation_fxn=activation_fxn)
        get_activations = tf.nn.dropout(get_activations, rate=dropout_rate)
        Ninputs = Noutputs
    return get_activations, Ninputs


def feed_forward_one_layer(
        input_tensor, layer_name, Nin=1, Nout=1,
        preactivation_fxn=tf_matmul_wrapper, activation_fxn=tf.nn.relu):
    """NB that for convnets, Nout is number of *channels*"""
    with tf.compat.v1.variable_scope(layer_name, reuse=tf.compat.v1.AUTO_REUSE):
        preactivations = preactivation_fxn(input_tensor, Nin, Nout)
        activations = activation_fxn(preactivations, name='activation')

        # for visualization in tensorboard
        # variable_summaries(wts, 'weights')
        # variable_summaries(biases, 'biases')
        # variable_summaries(preactivations, 'preactivations_summary')
        # variable_summaries(activations, 'activations_summary')
    return activations


def create_weights(weight_shape, stiffness=0, num_shards=None):
    # NB that lambda initialization is required for use under
    #  tensorflow control loops
    if num_shards:
        # ...then create a tensor for each shard and merge
        # Borrowed from tensor2tensor.layers.modalities.py
        #  to facilitate restoration of tensor2tensor models:
        shards = []
        for iShard in range(num_shards):
            shard_size = (weight_shape[0] // num_shards) + (
                1 if iShard < weight_shape[0] % num_shards else 0)
            var_name = "weights_%d" % iShard
            shards.append(tf.compat.v1.get_variable(
                var_name, shape=[shard_size] + weight_shape[1:],
                initializer=lambda shape, dtype, partition_info:
                    tf.compat.v1.truncated_normal(shape, stddev=0.1)))
        weights = tf.concat(shards, 0)
        # ret = eu.convert_gradient_to_tensor(ret)
    else:
        weights = tf.compat.v1.get_variable(
            'weights', shape=weight_shape,
            initializer=lambda shape, dtype, partition_info:
                ###tf.compat.v1.truncated_normal(shape, stddev=0.005))
                tf.compat.v1.truncated_normal(shape, stddev=0.1))
    #cost = tf.multiply(tf.nn.l2_loss(weights), stiffness, name='weight_loss')
    #tf.add_to_collection(tf.GraphKeys.LOSSES, cost)
    return weights


def create_biases(bias_shape, stiffness=0):
    '''
    initial_values = tf.constant(0.1, shape=bias_shape)
    biases = tf.get_variable('biases', initializer=initial_values)
    '''
    biases = tf.compat.v1.get_variable(
        'biases', shape=bias_shape,
        initializer=lambda shape, dtype, partition_info:
            ###tf.constant(0.0, shape=shape))
            tf.constant(0.1, shape=shape))
    #cost = tf.multiply(tf.nn.l2_loss(biases), stiffness, name='weight_loss')
    #tf.add_to_collection(tf.GraphKeys.LOSSES, cost)
    return biases


def LSTM_rnn(batch_sequences, sequence_lengths, hidden_layer_sizes,
             dropout, name, initial_state=None, BIDIRECTIONAL=False):
    # Borrowed from the tensor2tensor library, and modified
    '''
    Run LSTM cell on inputs, assuming they have size
    [Ncases x max_sequence_length x Ninputs].

    Input arguments:
    -------
    batch_sequences:
    sequence_lengths:
    hidden_size:
    num_hidden_layers:
    dropout:
    name:
    initial_state:

    Outputs:
    -------
    lstm_outputs:
    lstm_final_states:


    Un-/poorly documented behavior: tf.nn.dynamic_rnn returns two arguments,
     an output and a (final) state.  Providing the sequence_lengths as input
     will "copy-through state [the second output] and zero-out outputs [the
     first output] when past a batch element's sequence length" (per the tf
     documentation).  Therefore, since we're passing the sequence_lengths, we
     can safely use the last element of the final_state.  So far so good.

    However, the final_state is itself a tuple, consisting of a cell state (c)
     and a hidden state (h).  Per some stackexchange posts, e.g. this one,

        https://stackoverflow.com/questions/36817596/

     you use the hidden state.  On the difference b/n output and state, see

        https://stats.stackexchange.com/questions/330176

    Incidentally, the hidden state, final_state.h, is itself an array, with
     as many elements as layers in the RNN.  For decoding purposes, you will
     typically want to use the last, final_state.h[-1].


    '''

    '''
    with tf.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        layers = [
            tf.keras.layers.LSTM(
                hidden_size, dropout=dropout, return_sequences=True)
            for _ in range(num_hidden_layers)
            ### tf.keras.layers.CuDNNLSTM doesn't support masking
        ]
        if BIDIRECTIONAL:
            ### This is messed up b/c the initial state could be for multiple
            ###  layers
            layers = [tf.keras.layers.Bidirectional(layer, initial_state)
                      for layer in layers]
        stacked_layers = tf.keras.layers.StackedRNNCells(layers)
        masked_sequences = tf.keras.layers.Masking()(batch_sequences)
        outputs, states_tuple = stacked_layers(masked_sequences)


    '''
    # for brevity
    def variational_dropout_lstm_cell(input_size):
        LSTMcell = tf.compat.v1.nn.rnn_cell.LSTMCell(input_size, name='basic_lstm_cell')
        return tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            LSTMcell,
            input_keep_prob=1-dropout,
        )
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        forward_layers = [variational_dropout_lstm_cell(layer_size)
                          for layer_size in hidden_layer_sizes]
        if BIDIRECTIONAL:
            backward_layers = [variational_dropout_lstm_cell(layer_size)
                               for layer_size in hidden_layer_sizes]

            # see https://stackoverflow.com/questions/49242266/
            (outputs, final_state_fw, final_state_bw
             ) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                forward_layers,
                backward_layers,
                batch_sequences,
                sequence_length=sequence_lengths,
                initial_states_fw=initial_state,
                initial_states_bw=None,
                dtype=tf.float32)

            states_tuple = tuple(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(
                tf.concat((state_fw.c, state_bw.c), 1),
                tf.concat((state_fw.h, state_bw.h), 1))
                for state_fw, state_bw in zip(final_state_fw, final_state_bw))
        else:
            outputs, states_tuple = tf.compat.v1.nn.dynamic_rnn(
                tf.compat.v1.nn.rnn_cell.MultiRNNCell(forward_layers),
                batch_sequences,
                sequence_length=sequence_lengths,
                initial_state=initial_state,
                dtype=tf.float32)
        return outputs, states_tuple


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard
    visualization)."""
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var, name=name)
        tf.compat.v1.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        tf.compat.v1.summary.histogram(name, var)


def sequences_tools(sequences):
    '''
    Input arguments:
    --------
    sequences:
        tensor of size (Ncases x max_sequence_length x Ndims)

    Returns:
    --------
    index_sequences_elements:
        (sum_i^Nsequences seq_len(i) x 2) tensor listing all the non-zero
        indices in the tensor of sequences
    get_sequences_lengths:
        int32 tensor of size (Ncases)
    '''

    # mask_binariwise is a (Ncases x max_sequence_length) matrix with 0s
    #  wherever all elements of an input token are simultaneously zero,
    #  and 1s elsewhere.  Since all elements of an input token are
    #  simultaneously zero only in the zero-padding, the 1s will be
    #  contiguous, and the number of them in each row will be the
    #  corresponding true sequence length
    mask_binariwise = tf.sign(tf.reduce_max(input_tensor=tf.abs(sequences), axis=2))
    get_sequences_lengths = tf.reduce_sum(input_tensor=mask_binariwise, axis=1)
    get_sequences_lengths = tf.cast(get_sequences_lengths, tf.int32)
    # shouldn't it already be an int?
    index_sequences_elements = tf.cast(
        tf.compat.v1.where(tf.equal(mask_binariwise, 1)), tf.int32)

    return index_sequences_elements, get_sequences_lengths


def occlude_sequence_features(get_sequences, occluded_features):
    '''
    For all sequences in the zero-padded tensor get_sequences (with shape
    (Ncases x T_max x Nfeatures), replace the features labeled by index in
    occluded_features with their average values (exlcuding the zero-padding,
    of course).
    '''

    index_sequences, _ = sequences_tools(get_sequences)
    desequence_sequences = tf.gather_nd(get_sequences, index_sequences)
    desequenced_shape = common_layers.shape_list(desequence_sequences)

    average_feature_activities = tf.reduce_mean(
        input_tensor=desequence_sequences, axis=0)
    occlude_desequenced_sequences = tf.stack(
        [
            tf.fill((desequenced_shape[0], ), average_feature_activities[i])
            if i in occluded_features else desequence_sequences[:, i]
            for i in range(desequenced_shape[1])
        ], axis=1
    )

    return tf.scatter_nd(
        index_sequences, occlude_desequenced_sequences,
        tf.shape(input=get_sequences))


def tf_expected_word_error_rates(
    sequence_data_op_dict, get_sequence_log_probs,
    USE_BUILTIN=True, EXCLUDE_EOS=False, eos_id=1
):
    '''
    Compute word error rate on the results of a beam search.  In particular,
    tile the references to the beam size, and reshape into (Ncases*beam_width
    x max_sequence_length); then compute the word error rate in a vectorized
    way.

    Input arguments:
    --------
    sequence_data_op_dict: a dictionary that contains
        'decoder_targets': (Ncases x 1 x max_ref_length)
        'decoder_outputs': (Ncases x beam_width x max_hyp_length)
    get_sequence_log_probs': (Ncases x beam_width)

    For a categorical distribution, the natural params are (possibly
    unnormalized) log probabilities.  (I believe that the tensor2tensor
    beam_search, on whose outputs this is generally run, actually normalizes
    probabilities within the beam, but to be sure we treat these as
    unnormalized log probabilities.)

    Returns:
    --------
    average_word_error_rate
    '''

    # Ns
    Ncases = common_layers.shape_list(sequence_data_op_dict['decoder_targets'])[0]
    beam_width = common_layers.shape_list(sequence_data_op_dict['decoder_outputs'])[1]
    N_sentences = Ncases*beam_width

    # tile references to have the same shape as hypotheses
    references = tf.reshape(
        tf.tile(sequence_data_op_dict['decoder_targets'], [1, beam_width, 1]),
        [N_sentences, -1]
    )
    hypotheses = tf.reshape(
        sequence_data_op_dict['decoder_outputs'], [N_sentences, -1]
    )

    # get word error rates
    get_word_error_rate = (tf_word_error_rates_built_in if USE_BUILTIN
                           else tf_word_error_rates)
    word_error_rate_matrix = tf.reshape(
        get_word_error_rate(references, hypotheses, EXCLUDE_EOS, eos_id),
        [-1, beam_width]
    )

    # take average under the hypotheses' probabilities
    logZ = tf.reduce_logsumexp(get_sequence_log_probs, axis=1)
    logXpctWplus1 = tf.reduce_logsumexp(
        get_sequence_log_probs + tf.math.log(word_error_rate_matrix + 1.0),
        axis=1) - logZ

    # we added 1 to avoid log(0); now subtract that 1
    return tf.exp(logXpctWplus1) - 1


def tf_word_error_rates(references, hypotheses, EXCLUDE_EOS=False, eos_id=1):
    """
    Tensorflow implementation of a vectorized version of word error rate,
    based on the Levenstein distance.  The underlying algorithm is a variant
    on Wagner-Fisher/Needleman-Wunsch.

    Input arguments:
    --------
        references: a tensor with shape (N_sentences x max_ref_length)
        hypotheses: a tensor with shape (N_sentences x max_hyp_length)

    Returns:
    --------
        tensor with shape (N_sentences)


    Example:
    --------
        etc.....

    """
    # Created: 02/14/18
    #   by JGM

    ######
    # TO DO:
    #   (1) Make use of the args EXCLUDE_EOS and eos_id
    ######

    # Ns
    N_sentences, max_ref_length = common_layers.shape_list(references)
    max_hyp_length = common_layers.shape_list(hypotheses)[1]

    # upper bound on WER
    d_maxes = tf.fill((N_sentences,),
                      tf.maximum(max_ref_length, max_hyp_length) + 1)

    # get all the sequence lengths
    _, get_ref_lengths = sequences_tools(tf.expand_dims(references, axis=2))
    _, get_hyp_lengths = sequences_tools(tf.expand_dims(hypotheses, axis=2))

    # the conditions and bodies for a pair of nested while loops
    def inner_cond(i_ref, i_hyp, distances): return i_hyp < max_hyp_length

    def outer_cond(i_ref, i_hyp, distances): return i_ref < max_ref_length

    def inner_body(i_ref, i_hyp, distances):
        return i_ref, i_hyp+1, tf_fisher_wagner_body(i_ref, i_hyp, distances)

    def outer_body(i_ref, i_hyp, distances):
        i_ref, _, distances = tf.while_loop(
            cond=inner_cond, body=inner_body, loop_vars=(i_ref, i_hyp, distances),
            parallel_iterations=1, back_prop=False)
        return i_ref+1, tf.constant(0), distances

    def tf_fisher_wagner_body(i_ref, i_hyp, distances):
        match = tf.compat.v1.where(
            tf.equal(references[:, i_ref], hypotheses[:, i_hyp]),
            distances[:, i_ref, i_hyp],
            d_maxes
        )
        substitution = distances[:, i_ref, i_hyp] + 1
        insertion = distances[:, i_ref+1, i_hyp] + 1
        deletion = distances[:, i_ref, i_hyp+1] + 1
        updates = tf.reduce_min(input_tensor=tf.stack(
            [match, substitution, insertion, deletion], axis=1), axis=1)
        indices = tf.stack((tf.range(N_sentences),
                            tf.fill([N_sentences], i_ref+1),
                            tf.fill([N_sentences], i_hyp+1)), axis=1)
        return distances + tf.scatter_nd(
            indices, updates,
            shape=[N_sentences, max_ref_length+1, max_hyp_length+1])

    # initialize
    i_ref0 = tf.constant(0)
    i_hyp0 = tf.constant(0)
    row_indices = tf.range(max_ref_length+1)
    first_row_indices = tf.stack(
        (row_indices, tf.fill([max_ref_length+1], 0)), axis=1)
    col_indices = tf.range(max_hyp_length+1)
    first_col_indices = tf.stack(
        (tf.fill([max_hyp_length+1], 0), col_indices), axis=1)
    indices = tf.concat((first_col_indices, first_row_indices), axis=0)
    updates = tf.concat((col_indices, row_indices), axis=0)
    distances0 = tf.scatter_nd(indices, updates,
                               shape=[max_ref_length+1, max_hyp_length+1])
    distances0 = tf.transpose(
        a=tf.tile(tf.expand_dims(distances0, axis=2), [1, 1, N_sentences]),
        perm=[2, 0, 1])

    # run the nested while loops (Fisher-Wagner algorithm)
    _, _, distance_tensor = tf.while_loop(
        cond=outer_cond, body=outer_body, loop_vars=(i_ref0, i_hyp0, distances0),
        parallel_iterations=1, back_prop=False)

    # return just the distances at the end of each sentence
    distance_vector = tf.cast(tf.divide(tf.gather_nd(distance_tensor, tf.stack(
        [tf.range(N_sentences), get_ref_lengths, get_hyp_lengths], axis=1)),
        get_ref_lengths), tf.float32)
    return distance_vector


def tf_word_error_rates_built_in(
    references, hypotheses, EXCLUDE_EOS=False, eos_id=1
):
    # Use tensorflow's word_error_rate calculator

    # HARD-CODED PAD_ID
    pad_id = 0
    ignore_ids = [pad_id, eos_id] if EXCLUDE_EOS else [pad_id]

    #####
    # You should make this into a general function ("tf_broadcast_equal")...
    def extract_non_ignore_indices(sequences):
        return tf.compat.v1.where(tf.reduce_all(input_tensor=tf.not_equal(tf.expand_dims(
            sequences, 2), [[ignore_ids]]), axis=2))
    #####

    # ...
    index_references = extract_non_ignore_indices(references)
    sparse_references = tf.SparseTensor(
        index_references,
        tf.gather_nd(references, index_references),
        tf.cast(tf.shape(input=references), tf.int64)
    )
    index_hypotheses = extract_non_ignore_indices(hypotheses)
    sparse_hypotheses = tf.SparseTensor(
        index_hypotheses,
        tf.gather_nd(hypotheses, index_hypotheses),
        tf.cast(tf.shape(input=hypotheses), tf.int64)
    )
    return tf.edit_distance(sparse_hypotheses, sparse_references)


def seq_log_probs_to_word_log_probs(
    get_beam_outputs, get_sequence_log_probs, Nclasses,
    index_sequences_elements, max_targ_length, padding_value=0
):
    '''
    :param get_outputs: (Nsequences x beam_width x max_prediction_length)
    :param get_sequence_log_probs: (Nsequences x beam_width)
    :param Nclasses: scalar
    :param index_sequence_elements: (sum_i^Nsequences seq_len(i) x 2), a list
        of all the (putative) non-zero indices in the tensor of sequences
    :param max_targ_length: scalar tensor
    :return: score_as_unnorm_log_probs: (sum_i^Nsequences seq_len(i) x Nclasses),
        a tensor of log probabilities for each id, de-sequenced

    A sensible set of variables for a beam search to return is the set of the K
    most probable sequences and their probabilities, where K=beam_width. (These
    sequence_log_probs are not assumed to be normalized.)

    We want to expand the log probabilities to cover *all* tokens, not just the
    K most likely.  Conceptually, this is straightforward: For each element of
    each sequence, exponentiate the log probabilities; compute the "leftover"
    probability for all ids outside the beam, and divide it up equally among
    them; compute the logarithm elementwise.  Computationally, however, it is
    more complicated, b/c an effort must be made to avoid over- and underflows.

    Furthermore, to avoid doing any serious calculations, we have to make some
    simplifying choice for how to compute the "leftover" probabilities.  Here,
    we basically assign each non-selected id probability 1/S, S=total number of
    possible sequences.  That is, we pretend that each non-selected *sequence*
    has equal probability, 1/S, and then assume (what is certainly false) that
    each non-selected token at each time step *in each beam* can be assigned to
    exactly one of these non-selected sequences.  Hence e.g., even if token 324
    appears in at least one beam at time step t, it will still be assigned
    probability p at t in all beams where it did *not* appear. This facilitates
    summing log probabilities across the beams.

    Total number of sequences: For simplicity, ignore the end-of-sequence
    tokens.  For a vocabulary of size N and a maximum sequence length of M,
    there are N possible sequences that end at the first step; N^2 that end
    at the second step; and so forth up to N^M. Thus altogether there are
            N^1 + N^2 + N^3 + ... + N^M
        =   N^0 + N^1 + N^2 + N^3 + ... + N^M - 1
        =   (N^(M+1) - 1)/(N - 1) - 1
        ~=  N^M
    sequences, where the approximation follows from the fact that, for N or M
    of any reasonable size, the -1s don't matter.  Likewise, subtracting out
    the K in-beam sequences has no appreciable effect for any reasonable K.
    Hence the probability of each out-of-beam sequence is approximately N^-M,
    or again:
        log(out_beam_prob) = -M*log(N)

    Given the approximations, and more importantly since no attempt is made to
    decrease the in-beam probabilities by the probability assigned to out-of-
    beam ids, the result of logsumexp will be *unnormalized* log probabilities.
    These values are furthermore desequenced into shape
        (sum_i^Ncases targ_seq_len(i) x Nclasses)
    before returning.
    '''

    # one-hotify and scale by log probabilities
    #   -> (Ncases x beam_width x max_pred_length x Nclasses)
    # NB that the resulting tensor does *not* represent log probs, b/c it has
    #  *zeros* in the out-of-beam locations
    in_beam_log_probs = tf.multiply(
        tf.one_hot(get_beam_outputs, Nclasses, axis=-1),
        tf.expand_dims(tf.expand_dims(get_sequence_log_probs, axis=-1), axis=-1)
    )

    # pad out to max_targ_length
    #   -> (Ncases x beam_width x max_targ_length x Nclasses)
    max_pred_length = common_layers.shape_list(get_beam_outputs)[2]
    in_beam_log_probs = tf.pad(
        tensor=in_beam_log_probs,
        paddings=[
            [0, 0],
            [0, 0],
            [0, tf.maximum(max_targ_length - max_pred_length, 0) + 1],
            [0, 0]
        ],
        constant_values=padding_value
    )
    ###
    # This assumes the pad token=0.  Ideally, you'd pass this in explicitly,
    #  and then set constant_values=<pad value> in tf.pad.
    ###

    # fill in zeros with (approximate) out-of-beam log probs (see above)
    out_beam_log_prob = tf.multiply(
        tf.cast(-max_targ_length, tf.float32),
        tf.math.log(tf.cast(Nclasses, tf.float32)))
    out_beam_log_probs = tf.fill(
        common_layers.shape_list(in_beam_log_probs), out_beam_log_prob)
    IS_OUT_OF_BEAM = tf.equal(in_beam_log_probs, 0)
    beam_log_probs = tf.compat.v1.where(
        IS_OUT_OF_BEAM, out_beam_log_probs, in_beam_log_probs)

    # collapse across beam -> (Ncases x max_targ_length x Nclasses)
    score_as_unnorm_log_probs = tf.reduce_logsumexp(beam_log_probs, axis=1)

    # de-sequence -> (sum_i^Ncases targ_seq_len(i) x Nclasses)
    score_as_unnorm_log_probs = tf.gather_nd(
        score_as_unnorm_log_probs, index_sequences_elements)

    return score_as_unnorm_log_probs


def fake_beam_for_sequence_targets(
    desequenced_op_dict, unique_targets_list, beam_width, pad_token
):
    '''
    This function breaks each target and prediction at any spaces they contain,
    and treats the resulting lists as sentences between which to compute word
    error rates.  (For targets that don't contain spaces, nothing interesting
    happens.)

    Returns:
    --------
    references:
        (Ncases x 1 x max_ref_length) tensor of "reference" sentences
    hypotheses:
        (Ncases x beam_width x max_hyp_length)
        int32 tensor of size (Ncases)
    fake_beam_natural_params:
        (Ncases x beam_width)


    ####
    Why convert to strings?  Why not just compute with the word indices??
    ####
    '''

    # make tensors for the list of unique targets (single words or sentences)
    unique_targets_tensor = tf.constant(
        unique_targets_list, shape=[len(unique_targets_list), 1])

    # ...and the list of unique *tokens* that the targets comprise
    unique_tokens_list = targets_to_tokens(unique_targets_list, pad_token)
    unique_tokens_tensor = tf.constant(
        unique_tokens_list, shape=[1, 1, len(unique_tokens_list)])

    # pretend targets and predictions are themselves sequences
    _, fake_beam_ids = tf.nn.top_k(
        desequenced_op_dict['decoder_natural_params'], k=beam_width
    )
    make_target_matrix = tf_sentence_to_word_ids(
        desequenced_op_dict['decoder_targets'], unique_targets_tensor,
        unique_tokens_tensor, pad_token
    )
    make_prediction_matrix = tf_sentence_to_word_ids(
        fake_beam_ids, unique_targets_tensor, unique_tokens_tensor, pad_token)
    references = tf.expand_dims(make_target_matrix, axis=1)
    hypotheses = tf.reshape(
        make_prediction_matrix, [tf.shape(fake_beam_ids)[0], beam_width, -1]
    )

    # from *all* natural params, extract just some for a fake beam
    row_inds = tf.cast(tf.tile(
        tf.expand_dims(tf.range(tf.shape(fake_beam_ids)[0]), 1),
        (1, beam_width)), tf.int32)
    fake_beam_inds = tf.stack((tf.reshape(row_inds, [-1]),
                               tf.reshape(fake_beam_ids, [-1])), 1)
    fake_beam_natural_params = tf.reshape(
        tf.gather_nd(desequenced_op_dict['decoder_natural_params'], fake_beam_inds),
        [-1, beam_width]
    )

    return references, hypotheses, fake_beam_natural_params


def targets_to_tokens(unique_targets_list, pad_token):
    '''
    This only does something interesting for 'trial' data, i.e. if the unique
    targets are single strings of *sentences* (containing spaces).  In this
    case, the unique_tokens_list contains all words in those sentences, plus
    the pad_token.  If the unique targets are single words, then the unique
    tokens will just be identical, although possibly re-ordered, and adding the
    pad_token if it wasn't originally present.

    '''
    # get unique tokens by splitting into pieces (at spaces); exclude pad_token
    unique_tokens_list = list(set([
        item for target in unique_targets_list for item in target.split(' ')
        if item != pad_token]))
    unique_tokens_list.sort()  # enforce deterministic behavior

    # make sure the pad_token is at the beginning of the list [why?]
    unique_tokens_list = [pad_token] + unique_tokens_list

    return unique_tokens_list


def tf_sentence_to_word_ids(
        sentence_ids, unique_targets_tensor, unique_tokens_tensor, pad_token):
    '''
    (1) Given sentence *target* ids, extract the corresponding sentences
    (2) Then break these into words (Ncases, max_len, 1)
    (3) Convert from words to *token* ids, by broadcasting tf.equals.  The
        resulting matrix will have the word ID in its third column, and its
        location (sentence number, word number) in the first two columns
    (4) Scatter back into a matrix, with zero padding
    '''

    extract_sentences = tf.gather(unique_targets_tensor, sentence_ids)
    Ncases = tf.size(extract_sentences)
    extract_words = tf.expand_dims(tf.sparse.to_dense(tf.compat.v1.string_split(
        tf.reshape(extract_sentences, [-1])), default_value=pad_token), -1)
    id_tokens = tf.compat.v1.where(tf.equal(unique_tokens_tensor, extract_words))
    return tf.scatter_nd(
        id_tokens[:, 0:2], id_tokens[:, 2],
        [tf.cast(Ncases, tf.int64), 1+tf.reduce_max(input_tensor=id_tokens[:, 1])])


def average_gradients(tower_grads):
    # Cribbed from the tensorflow tutorials:
    # tutorials/image/cifar10/cifar10_multi_gpu_train.py
    """Calculate the average gradient for each shared variable across
     all towers.

    Note that this function provides a synchronization point across
    all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The
        outer list is over individual gradients. The inner list is
        over the gradient calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has
        been averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over
            # below.
            grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(input_tensor=grad, axis=0)

            # Keep in mind that the Variables are redundant because they
            # are shared across towers. So .. we will just return the
            # first tower's pointer to the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def tf_linear_interpolation(X, stretch_factor, axis=0):
    '''
    Linearly interpolate sequences in `X ` along axis `axis`, according to
    the `stretch_factor`.

    Each point in a linear interpolation is a weighted sum of the closest
    values, above and below.
    '''

    # get the original and new (resampled) max sample
    T_orig = tf.shape(input=X)[1]
    T_new = tf.round(stretch_factor*tf.cast(T_orig-1, tf.float32))

    # interpolate "indices"--but NB that these are actually floats
    interpolate_inds = tf.range(T_new)/stretch_factor

    # the closest integer-valued indices *below* each interpolated value
    get_lower_inds = tf.cast(tf.floor(interpolate_inds), tf.int32)

    # extract points at these lower inds and at the next higher inds
    extract_lower_vals = tfh.fancy_indexing(X, get_lower_inds, axis=axis)
    extract_upper_vals = tfh.fancy_indexing(X, get_lower_inds+1, axis=axis)

    # linear interpolant is a weighted sum of these values
    get_w_lower = tf.cast(get_lower_inds + 1, tf.float32) - interpolate_inds
    get_w_upper = interpolate_inds - tf.cast(get_lower_inds, tf.float32)

    new_shape = [tf.constant(1) if i != axis else tf.shape(input=get_lower_inds)[0]
                 for i in range(len(common_layers.shape_list(X)))]
    get_w_lower = tf.reshape(get_w_lower, new_shape)
    get_w_upper = tf.reshape(get_w_upper, new_shape)

    return get_w_lower*extract_lower_vals + get_w_upper*extract_upper_vals
