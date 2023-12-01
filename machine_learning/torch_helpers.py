# standard libraries
import pdb

# third-party libraries
import numpy as np
import torch


# for compatibility with tensorflow
def parse_protobuf_seq2seq_example(example_proto, data_manifests):
    '''
    NB that all sequence_matrices are [T x N_features]
    '''

    example_dict = dict.fromkeys(data_manifests.keys())

    for key, data_manifest in data_manifests.items():
        # ..."unflatten" the sequence of (possibly length-1) vectors and xform

        sequence_vector = np.array(example_proto[data_manifest.sequence_type])
        sequence_matrix = sequence_vector.reshape(
            [-1, data_manifest.num_features_raw]
        )
        sequence_matrix = data_manifest.transform(sequence_matrix)
        if not (type(sequence_matrix) is torch.Tensor):
            sequence_matrix = torch.tensor(sequence_matrix)
        example_dict[key] = sequence_matrix

    return example_dict


def fancy_indexing(X, extract_inds, axis=0):
    ##############
    # FIX ME
    ##############
    '''
    Select indices along axis `axis` from tensor `X` with indices in rank-1
    tensor `extract_inds`.
    '''
    # Expand me to deal with all the interesting numpy cases

    # get the indices for gathering/scattering
    X_shape = tf.shape(input=X)
    make_grid_coords = [tf.range(X_shape[i]) for i in range(axis)]
    make_grid = tf.meshgrid(*make_grid_coords, extract_inds, indexing='ij')
    vectorize_grid = [tf.reshape(grid_coords, [-1]) for grid_coords in make_grid]
    matricize_grid = tf.stack(vectorize_grid, axis=1)

    # gather them up and reshape
    new_shape = [
        X_shape[i] if i != axis else tf.shape(input=extract_inds)[0]
        for i in range(len(common_layers.shape_list(X)))
    ]
    return tf.reshape(tf.gather_nd(X, matricize_grid), new_shape)


def string_seq_to_index_seq(
    sequence_matrix, unique_targets_list, eos_id_list, OOV_index
):
    '''
    Convert a sequence of strings (sequence_matrix) into a sequence of indices
    into the unique_targets_list, where
        indices[:, 1] = target_ids
        indices[:, 0] = sequence positions.
    NB that the sequence_matrix must have size (N x 1) as opposed to (N, ).
    Strings not found in the unique_targets_list are converted to the OOV_id.

    As a final step, the sequence is appended with eos_id_list, which typically
    will hold either a single id or none at all (e.g. for single-word data).
    Note that the returned sequence is, like the input, (N x 1).
    '''

    # naively get the indices for all elements of this sequence
    unique_bytes_list = [t.encode('utf-8') for t in unique_targets_list]
    sequence_positions, target_ids = torch.where(torch.tensor(
        np.array(unique_bytes_list)[None, :] == sequence_matrix
    ))

    # If a sequence element is missing (because that target wasn't in the
    #  unique_targets_list), replace it with the OOV_index.
    target_shape = sequence_matrix.shape[0:1]
    all_OOV_vector = torch.full(target_shape, OOV_index)
    index_sequence = all_OOV_vector.scatter_(0, sequence_positions, target_ids)

    # append the EOS_id
    index_sequence = torch.cat((index_sequence, torch.tensor(eos_id_list)))

    # return a "matrix"
    return index_sequence[:, None]


def get_word_error_rate(
    references, hypotheses, m_cost=0, s_cost=1, i_cost=1, d_cost=1, cost_fxn=None
):
    """
    Vectorized calculation of word error rate with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time and space complexity.

    Input arguments:
    --------
        references : tensor of right-zero-padded rows of integers
            (N_sentences x max_length x 1)
        hypotheses : tensor of right-zero-padded rows of integers
            (N_sentences x max_length x 1)

    (Tensors are expected because of where this function gets used....)

    Returns:
    --------
        numpy array (vector) of len(references) (== len(hypotheses))

    Revised: 11/20/23
        re-wrote for PyTorch
    Created: 02/12/18
        by JGM
        Inspired by scalar version found here:
            https://martin-thoma.com/word-error-rate-calculation/
    """

    # ...
    device = references.get_device()
    N_sentences = references.shape[0]
    if hypotheses.shape[0] != N_sentences:
        raise ValueError('no. of hypotheses must equal no. of references')

    # ...
    _, references_lengths = sequences_tools(references)
    _, hypotheses_lengths = sequences_tools(hypotheses)

    N_ref_max = max(references_lengths)
    N_hyp_max = max(hypotheses_lengths)
    d_max = max(N_ref_max, N_hyp_max)

    # initialize
    if cost_fxn is None:
        def cost_fxn(ref, hyp):
            return m_cost, s_cost, i_cost, d_cost

        distance_tensor = torch.zeros(
            (N_sentences, N_ref_max + 1, N_hyp_max + 1), dtype=torch.uint8
        ).to(device)
        distance_tensor[:, 0] = torch.arange(N_hyp_max + 1)[None, :]
        distance_tensor[:, :, 0] = torch.arange(N_ref_max + 1)[None, :]
    else:
        distance_tensor = torch.full(
            (N_sentences, N_ref_max + 1, N_hyp_max + 1), torch.inf,
        ).to(device)
        distance_tensor[:, 0, 0] = 0

    # compute minimum edit distance
    for i_ref in range(N_ref_max):
        for i_hyp in range(N_hyp_max):
            m_cost, s_cost, i_cost, d_cost = cost_fxn(
                references[:, i_ref], hypotheses[:, i_hyp])
            match = m_cost + distance_tensor[:, i_ref, i_hyp] + d_max*(
                references[:, i_ref, 0] != hypotheses[:, i_hyp, 0])
            substitution = s_cost + distance_tensor[:, i_ref, i_hyp]
            insertion = i_cost + distance_tensor[:, i_ref + 1, i_hyp]
            deletion = d_cost + distance_tensor[:, i_ref, i_hyp + 1]
            distance_tensor[:, i_ref+1, i_hyp+1], _ = torch.min(torch.stack(
                [match, substitution, insertion, deletion]
            ), dim=0)

    distances = distance_tensor[
        (torch.arange(N_sentences), references_lengths, hypotheses_lengths)
    ]

    return distances/references_lengths


def sequences_tools(sequences, as_tuple=True):
    '''
    Input arguments:
    --------
    sequences:
        tensor of size (Ncases x max_sequence_length x Ndims)

    Returns:
    --------
    sequences_indices:
        (sum_i^Nsequences seq_len(i) x 2) tensor listing all the non-zero
        indices in the tensor of sequences
    sequences_lengths:
        int32 tensor of size (Ncases)
    '''

    # mask_binariwise is a (Ncases x max_sequence_length) matrix with 0s
    #  wherever all elements of an input token are simultaneously zero,
    #  and 1s elsewhere.  Since all elements of an input token are
    #  simultaneously zero only in the zero-padding, the 1s will be
    #  contiguous, and the number of them in each row will be the
    #  corresponding true sequence length
    max_vals, _ = torch.max(torch.abs(sequences), axis=2)
    binary_mask = torch.sign(max_vals)
    sequences_lengths = torch.sum(binary_mask, axis=1, dtype=torch.int32)
    sequences_indices = torch.nonzero(binary_mask, as_tuple=as_tuple)

    return sequences_indices, sequences_lengths


def reverse_sequences(sequences, indices, lengths):
    # You can get indices and lengths from sequences_tools, but NB that in 
    #  practice you often won't run this on sequences but on some precursor
    #  thereof.

    reversed_indices = (
        indices[0], torch.repeat_interleave(lengths, lengths) - indices[1] - 1,
    )
    new_sequences = sequences.clone()
    new_sequences[indices] = sequences[reversed_indices]

    return new_sequences
