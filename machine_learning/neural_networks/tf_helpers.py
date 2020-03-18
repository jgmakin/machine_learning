# standard libraries
import pdb
import sys
import os

# third-party packages
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensor2tensor.layers import common_layers

# local
from utils_jgm.toolbox import auto_attribute

'''
A collection of helper functions for use with tensorflow

:Author: J.G. Makin (except where otherwise noted)

Cribbed from other JGM code: June 2018
'''

PS_OPS = ("Variable", "VariableV2", "AutoReloadVariable",
          "MutableHashTable", "MutableHashTableV2",
          "MutableHashTableOfTensors", "MutableHashTableOfTensorsV2",
          "MutableDenseHashTable", "MutableDenseHashTableV2",
          "VarHandleOp", "BoostedTreesEnsembleResourceHandleOp"
          "Assert", "StringFormat", "PrintV2"   # added by JGM
          )


class GraphBuilder:
    @auto_attribute
    def __init__(
        self,
        # functions:
        training_data_fxn,
        assessment_data_fxn,
        training_net_builder,
        assessment_net_builder,
        optimizer,
        assessor,
        # other arguments:
        checkpoints_path,
        final_epoch,
        # arguments with default values:
        initial_epoch=0,
        EMA_decay=0.0,
        reuse_vars_scope=None,
        training_GPUs=None,
        assessment_GPU=0,
        # private; don't assign these to self:
        _restore_epoch=None,
        _restore_model=None,
    ):

        # construct and store other useful parameters
        if _restore_epoch is None:
            _restore_epoch = final_epoch
        self.last_checkpoint = self.checkpoints_path + '-%i' % _restore_epoch
        if reuse_vars_scope:
            self.final_epoch += _restore_epoch
            self.initial_epoch += _restore_epoch
        if _restore_model:
            token_type = os.path.split(os.path.split(os.path.split(
                self.last_checkpoint)[0])[0])[1]
            self.last_checkpoint = self.last_checkpoint.replace(
                token_type, _restore_model + '_' + token_type)

    def train_and_assess(self, assessment_epoch_interval=1):
        '''
        Train and assess a neural netword built in tensorflow.
        '''

        # construct, initialize training and assessment graphs
        (update_params, initialize_training_data, training_saver, training_sess,
         ) = self._build_training_graph()
        (assessment_sess, assessment_saver, assessments
         ) = self._build_assessment_graph()

        # start training
        assessment_step = 0
        try:
            for training_epoch in range(self.initial_epoch, self.final_epoch):
                print('training...')
                training_sess.run(initialize_training_data)
                while True:
                    try:
                        training_sess.run(update_params)
                    except tf.errors.OutOfRangeError:
                        break
                if training_epoch % assessment_epoch_interval == 0:
                    print('assessing...')
                    assessments = self._save_and_assess(
                        training_sess, training_saver, training_epoch,
                        assessment_sess, assessment_saver, assessment_step,
                        assessments)
                    assessment_step += 1
            else:
                # be sure to save the model (and assess) after the last epoch
                assessments = self._save_and_assess(
                    training_sess, training_saver, training_epoch+1,
                    assessment_sess, assessment_saver, assessment_step,
                    assessments)

            self.close_all(
                training_sess, assessment_sess,
                *[struct.writer for (key, struct) in assessments.items()],
                # writer
            )
            return assessments

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(str(exc_type))
            print(str(exc_value))
            print('AT LINE ' + str(exc_traceback.tb_lineno))
            print('cleaning up...')
            self.close_all(
                training_sess, assessment_sess,
                *[struct.writer for (key, struct) in assessments.items()],
                # writer
            )
            print('...cleaned up!')

            return assessments

    def assess(self):
        (assessment_sess, assessment_saver, assessments
         ) = self._build_assessment_graph()
        assessment_saver.restore(assessment_sess, self.last_checkpoint)
        for data_partition in assessments.keys():
            assessments[data_partition] = self.assessor(
                assessment_sess, assessments[data_partition],
                self.final_epoch, 0, data_partition)
        return assessments

    def get_saliencies(self):
        # get gradients (of inputs)
        controller = '/cpu:0'
        initialize_data, tower_grads = self._parallel_differentiator(controller)
        with tf.compat.v1.name_scope("apply_gradients"), tf.device(controller):
            get_input_saliencies = self._average_tower_gradients(tower_grads)

        # create the session and restore the graph
        EMA = tf.train.ExponentialMovingAverage(
            decay=self.EMA_decay) if self.EMA_decay else None
        sess, saver = get_session_and_saver(
            EMA=EMA, allow_soft_placement=True, allow_growth=True)
        saver.restore(sess, self.last_checkpoint)

        # an abuse of the assessor...
        return self.assessor(sess, initialize_data, get_input_saliencies[0][0])

    def _build_training_graph(self):

        training_graph = tf.Graph()
        with training_graph.as_default():
            # set up trainer; open session; restore saved weights
            update_params, initialize_data, EMA = self._parallel_trainer()
            sess, saver = get_session_and_saver(
                EMA=EMA, allow_soft_placement=True, allow_growth=True)
            self._restore_weights(sess, training_graph, EMA)
            training_graph.finalize()

        print('Training graph built...')
        return update_params, initialize_data, saver, sess

    def _restore_weights(self, sess, training_graph, EMA):
        if self.reuse_vars_scope:
            reuse_vars = tf.compat.v1.trainable_variables(self.reuse_vars_scope)
            if EMA is not None:
                # The keys are EMA names; the values are EMA variables where
                #  they exist and otherwise the "regular" variables. NB that it
                #  includes the AdaM variables, although these are filtered out
                #  below by requiring the vars be in reuse_vars.
                EMA_reuse_var_dict = EMA.variables_to_restore()
                reuse_vars_dict = {EMA.average_name(var): EMA_reuse_var_dict[
                    EMA.average_name(var)] for var in reuse_vars}
            else:
                reuse_vars_dict = {
                    reuse_var.name.split(':')[0]:
                        training_graph.get_tensor_by_name(reuse_var.name)
                    for reuse_var in reuse_vars
                }
            # pdb.set_trace()
            training_restore_saver = tf.compat.v1.train.Saver(reuse_vars_dict)
            training_restore_saver.restore(sess, self.last_checkpoint)

    def _build_assessment_graph(self):

        # for malloc'ing some data storage
        num_epochs = self.final_epoch - self.initial_epoch

        # under this graph (and device)...
        assessment_graph = tf.Graph()
        #with assessment_graph.as_default(), tf.device(
        #        '/gpu:%i' % self.assessment_GPU):
        with assessment_graph.as_default():
            # ...create data; assess; prepare plots, init storage; open session
            GPU_op_dict, CPU_op_dict, assessments = self.assessment_data_fxn(
                num_epochs)
            self.assessment_net_builder(GPU_op_dict, CPU_op_dict)
            EMA = (tf.train.ExponentialMovingAverage(decay=self.EMA_decay)
                   if self.EMA_decay else None)
            sess, saver = get_session_and_saver(EMA=EMA, allow_growth=True)
            assessment_graph.finalize()

        print('Assessment graph built...')
        return sess, saver, assessments

    def _save_and_assess(
        self, training_sess, training_saver, training_epoch,
        assessment_sess, assessment_saver, assessment_step, assessments
    ):
        this_checkpoint = training_saver.save(
            training_sess, self.checkpoints_path, global_step=training_epoch)
        assessment_saver.restore(assessment_sess, this_checkpoint)
        for data_partition in assessments.keys():
            assessments[data_partition] = self.assessor(
                assessment_sess, assessments[data_partition],
                training_epoch, assessment_step, data_partition)
        return assessments

    def _parallel_trainer(self):
        # http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/
        initialize_data, tower_grads = self._parallel_differentiator()
        update_weights = self._parallel_weight_updater(tower_grads)
        if self.EMA_decay:
            EMA = tf.train.ExponentialMovingAverage(decay=self.EMA_decay)
            update_moving_averages = EMA.apply(tf.compat.v1.trainable_variables())
            update_params = tf.group((update_weights, update_moving_averages))
        else:
            EMA = None
            update_params = update_weights

        return update_params, initialize_data, EMA

    def _parallel_differentiator(self, controller='/cpu:0'):
        # http://blog.s-schoener.com/2017-12-15-parallel-tensorflow-intro/

        # return a list of device ids like`['/gpu:0', '/gpu:1']`
        devices = get_available_gpus()
        if not devices:
            devices = ['/cpu:0']
            print('No GPUs available or requested! using the CPU -- jgm')
        elif self.training_GPUs is not None:
            devices = [devices[i] for i in self.training_GPUs]
        else:
            print('Using *all* %i GPUs...' % len(devices))

        # the ops make data, to be placed either on the GPU or the CPU
        (GPU_op_dict, CPU_op_dict, initialize_data
         ) = self.training_data_fxn(len(devices))

        # Get the current variable scope so we can reuse all variables we need
        #  once we get to the nth iteration of the loop below
        tower_grads = []
        for iDevice, device_id in enumerate(devices):
            print('Setting up tower on %s' % device_id)
            tower_name = 'tower_{}'.format(iDevice)

            # force onto controller device
            ##########
            # with tf.device(self._assign_to_device(device_id, controller)):
            ##########
            with tf.device('/gpu:{}'.format(iDevice)):
                # but see: https://stackoverflow.com/questions/45156542/
                with tf.compat.v1.name_scope(tower_name) as scope:
                    # compute gradients
                    model_outputs = self.training_net_builder(
                        {key: op[iDevice] for key, op in GPU_op_dict.items()},
                        CPU_op_dict,
                        tower_name=tower_name
                    )
                    with tf.compat.v1.name_scope("compute_gradients"):
                        # get list of (gradient, variable) pairs
                        grads_and_vars = self.optimizer.compute_gradients(
                            *model_outputs)
                        tower_grads.append(grads_and_vars)

        return initialize_data, tower_grads

    def _parallel_weight_updater(self, tower_grads, controller='/cpu:0'):

        # Apply the gradients on the controlling device
        with tf.compat.v1.name_scope("apply_gradients"), tf.device(controller):
            # back on the CPU, average the gradients from each tower

            # (gradient, variable) lists -> (gradient, variables) list
            gradients = self._average_tower_gradients(tower_grads)
            update_weights = self.optimizer.apply_gradients(gradients)

        return update_weights

    @staticmethod
    def _average_tower_gradients(tower_grads):
        '''
        Calculate average gradient for each shared variable across all towers.

        See https://github.com/tensorflow/models/blob/master/tutorials/image/
            cifar10/cifar10_multi_gpu_train.py#L101

        Note: this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer
            list ranges over the devices. The inner list ranges over the
            different variables.
        Returns:
                List of pairs of (gradient, variable) where the gradient has
                been averaged across all towers.
        '''
        averaged_grads_and_vars = []
        for grad_and_var_all_towers in zip(*tower_grads):
            # Each grad_and_var_all_towers is
            #  ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

            ####
            # List comprehension fails for certain variables....
            #grads_list = [g for g, _ in grad_and_var_all_towers]
            grads_list = []
            for g, _ in grad_and_var_all_towers:
                expanded_g = tf.expand_dims(g, 0)
                grads_list.append(expanded_g)
            grads = tf.concat(axis=0, values=grads_list)
            ####
            grad = tf.reduce_mean(input_tensor=grads, axis=0)

            # Variables (grad_and_var_all_towers[iTower][1] for all iTower) are
            #  redundant because they are shared across towers, so we can
            #  return just the pointer from tower 0.
            averaged_grads_and_vars.append((grad, grad_and_var_all_towers[0][1]))
        return averaged_grads_and_vars

    @staticmethod
    def _assign_to_device(device, ps_device):
        '''
        Returns a function to place variables on the ps_device.

        See https://github.com/tensorflow/tensorflow/issues/9517

        Args:
            device: Device for everything but variables
            ps_device: Device to put the variables on. Example values are
                /GPU:0 and /CPU:0.

        If ps_device is not set then variables will be placed on the default
        device.  The best device for shared varibles depends on the platform as
        well as the model. Start with CPU:0 and then test GPU:0 to see if there
        is an improvement.
        '''
        def _assign(op):
            node_def = op if isinstance(op, tf.compat.v1.NodeDef) else op.node_def
            if node_def.op in PS_OPS or 'read' in node_def.name:
                # If you don't do this, the 'read' ops for the kernels and
                #   biases in tf.nn.rnn_cell.LSTMCell end up on GPU:0.  For
                #   details on the read op, see:
                #       https://stackoverflow.com/questions/42783909/
                #
                # or '/Assert' in op.name or '/summaries' in op.name:
                return ps_device
            else:
                return device
        return _assign

    @staticmethod
    def close_all(*args):
        for arg in args:
            arg.close()


def get_session_and_saver(
    initialize_graph=None, EMA=None, allow_soft_placement=True,
    allow_growth=False,
):
    # if there isn't an initializer op, create it
    if initialize_graph is None:
        initialize_graph = tf.group(
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer()
        )

    # create a session
    sess_config = tf.compat.v1.ConfigProto(
        log_device_placement=True, allow_soft_placement=allow_soft_placement,
    )
    sess_config.gpu_options.allow_growth = allow_growth
    sess = tf.compat.v1.Session(config=sess_config)

    # exponential moving average
    if EMA:
        saver = tf.compat.v1.train.Saver(EMA.variables_to_restore())
    else:
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    sess.run(initialize_graph)

    return sess, saver


def get_session_with_saved_model(
        restore_dir, allow_soft_placement=True, allow_growth=False):

    # well, this is just what you usually do
    sess_config = tf.compat.v1.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=allow_soft_placement
    )
    sess_config.gpu_options.allow_growth = allow_growth
    sess = tf.compat.v1.Session(config=sess_config)

    # load the model into (?) this session, and return it
    tf.compat.v1.saved_model.loader.load(sess, ["serve"], restore_dir)

    return sess


def get_available_gpus():
    '''
    Returns a list of the identifiers of all visible GPUs.
    See https://stackoverflow.com/questions/38559755/
    '''
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def hide_shape(x):
    '''
    This is a rather subtle little function.  There are times when you would
    like to hide a tensor x's shape from a tensorflow op--e.g., when that op
    is created in a branch of a conditional statement (tf.cond, tf.case) that
    will never be executed, but with which the shape of x is incompatible.
    Since tensorflow sets up the ops in *all* branches of the conditional,
    its shape inference will choke on x in this case.  So first we obscure the
    shape with this function.

    I (JGM) copied it directly from here:
        https://github.com/tensorflow/tensorflow/issues/6906
    '''
    return tf.cond(
        pred=tf.constant(True),
        true_fn=lambda: x,
        false_fn=lambda: tf.compat.v1.placeholder(x.dtype)
    )


def fancy_indexing(X, extract_inds, axis=0):
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
    new_shape = [X_shape[i] if i != axis else tf.shape(input=extract_inds)[0]
                 for i in range(len(common_layers.shape_list(X)))]
    return tf.reshape(tf.gather_nd(X, matricize_grid), new_shape)


def rescale(get_X, xmin, xmax, zmin, zmax):
    scaling = (zmax - zmin)/(xmax - xmin)
    return scaling*(get_X - xmin) + zmin


def tf_print(tensor, message="JGM TENSOR: "):
    print_op = tf.print(message, tensor)
    with tf.control_dependencies([print_op]):
        tensor = tf.identity(tensor)
    return tensor


def make_feature_example(example_dict):
    '''
    For this "example," construct a dictionary of "Features" with the same keys
    as the example_dict passed in.
    '''

    feature_dict = {}
    for key, value in example_dict.items():
        if type(value) is list:
            feature_dict[key] = _featurize_bytes_list(value)
        elif type(value) is np.ndarray:
            # *assume* it's a float, convert to single precision, and flatten
            feature_dict[key] = _featurize_float_list(
                np.float32(value).reshape(-1))
        else:
            raise NotImplementedError(
                "Only list and ndarray features have been implemented")

    # transform the dictionary into Features
    features = tf.train.Features(feature=feature_dict)

    # transform Features into an Example, and return
    return tf.train.Example(features=features)


def _featurize_bytes_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _featurize_int64_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _featurize_float_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse_protobuf_seq2seq_example(example_proto, data_manifests):

    # parse the features using the data_descriptions and prepare the outputs
    feature_dict = {
        data_manifest.sequence_type: data_manifest.feature_value
        for data_manifest in data_manifests.values()
    }
    parsed_features = tf.io.parse_single_example(
        serialized=example_proto, features=feature_dict)
    example_dict = dict.fromkeys(data_manifests.keys())

    # for each data_manifest (the number is indeterminate)...
    for key, data_manifest in data_manifests.items():
        # ..."unflatten" the sequence of (possibly length-1) vectors and xform
        sequence_matrix = tf.reshape(
            parsed_features[data_manifest.sequence_type].values,
            (-1, data_manifest.num_features_raw)
        )
        example_dict[key] = data_manifest.transform(sequence_matrix)

    ############
    return example_dict
    # keys_tensor = tf.constant(example_dict.keys())
    # vals_tensor = tf.constant(example_dict.values())
    # return tf.lookup.StaticHashTable(
    #     tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
    ############


def replace_with_gaussian_noise(data_op):
    return tf.random.normal(tf.shape(input=data_op))


def randomly_rotate_sequence(data_op):
    T = tf.shape(data_op)[0]
    return tf.roll(data_op, shift=T//2, axis=0)


def string_seq_to_index_seq(
    sequence_matrix, unique_targets_list, eos_id_list, OOV_id
):
    '''
    Convert a sequence of strings (sequence_matrix) into a sequence of indices
    into the unique_targets_list, where
        indices[:,1] = target_ids
        indices[:,0] = sequence positions.
    NB that the sequence_matrix must have size (N x 1) as opposed to (N, ).
    Strings not found in the unique_targets_list are converted to the OOV_id.

    As a final step, the sequence is appended with eos_id_list, which typically
    will hold either a single id or none at all (e.g. for single-word data).
    Note that the returned sequence is, like the input, (N x 1).
    '''

    # naively get the indices for all elements of this sequence
    unique_bytes_list = [t.encode('utf-8') for t in unique_targets_list]
    indices = tf.cast(tf.compat.v1.where(tf.equal(
        tf.constant(unique_bytes_list, shape=[1, len(unique_bytes_list)]),
        sequence_matrix
        ### can be [:, None]?
    )), tf.int32)

    # If a sequence element is missing (because that target wasn't in the
    #  unique_targets_list), replace it with the OOV_id.
    target_shape = tf.shape(sequence_matrix)[0:1]
    all_OOV_vector = tf.fill(target_shape, OOV_id)
    updates = indices[:, 1]
    cull_non_OOV_target_ids = tf.scatter_nd(
        indices[:, 0, None], updates, target_shape)
    updates = tf.ones_like(indices[:, 1], dtype=tf.bool)
    mask_non_OOV_target_ids = tf.scatter_nd(
        indices[:, 0, None], updates, target_shape)
    replace_missing_with_OOV = tf.compat.v1.where(
        mask_non_OOV_target_ids, cull_non_OOV_target_ids, all_OOV_vector)

    # append the EOS_id before returning
    return tf.concat((replace_missing_with_OOV, eos_id_list), axis=0)[:, None]
