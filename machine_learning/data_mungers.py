# standard libraries
import os
import pdb
import random
from functools import reduce

# third-party
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision.transforms import v2
    import torchvision

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
except ModuleNotFoundError:
    print('WARNING: torch missing; skipping')

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ModuleNotFoundError:
    print('WARNING: TensorFlow missing; skipping')


# local
from utils_jgm.machine_compatibility_utils import MachineCompatibilityUtils
MCUs = MachineCompatibilityUtils()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LocalCelebA(torchvision.datasets.CelebA):
    def __init__(self, root=os.path.join(MCUs.get_path('data')), *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)

    @property
    def raw_folder(self) -> str:
        # return os.path.join(self.root, self.__class__.__name__, 'raw')
        return os.path.join(self.root, 'MNIST', 'raw')

    def __getitem__(self, index: int) -> tuple[any, any]:
        """
        Args:
            index (int): Index

        Returns:
            image (): matrix of size image_height x image_width
        """

        # apply any 
        class_id = self.targets[index]
        if self.target_transform is not None:
            class_id = self.target_transform(class_id)

        image = self._image_proc(index)

        return image, class_id

    def _image_proc(self, index):
        image = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image.numpy(), mode="L")

        if self.transform is not None:
            image = self.transform(image)

        return image


class LocalOxfordFlowers(torchvision.datasets.Flowers102):
    def __init__(
        self,
        *args,
        root=os.path.join(MCUs.get_path('data')),
        size=(64, 64),
        # OF has a tiny training set, so swap w/the large test set
        split='test',
        **kwargs
    ):
        super().__init__(*args, root=root, split=split, **kwargs)
        self.device = device
        self.size = size
        
        # 1. Flowers102 stores paths in self._image_files
        # We load, resize, and stack them onto the GPU immediately
        processed_imgs = []
        print(f"Loading Oxford Flowers to {device}...")
        
        # Temporary transform to unify sizes for the tensor stack
        init_transform = v2.Compose([
            v2.Resize(self.size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        for img_path in self._image_files:
            img = Image.open(img_path).convert("RGB")
            processed_imgs.append(init_transform(img))
        
        # Stack into (N, 3, H, W) and move to device
        self.data = torch.stack(processed_imgs).to(self.device)
        
        # 2. Move labels to GPU (Flowers102 labels are in self._labels)
        self.targets = torch.tensor(self._labels, device=self.device)

    def __getitem__(self, index: int):
        image = self.data[index]
        label = self.targets[index]

        # Apply any runtime augmentations (Flips, etc.)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class LocalCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
        self, *args, root=os.path.join(MCUs.get_path('data')), **kwargs
    ):
        FLATTEN = kwargs.pop('FLATTEN', False)
        super().__init__(*args, root=root, **kwargs)
        self.device = device
        
        # 1. CIFAR10 stores data as a (50000, 32, 32, 3) numpy array
        # Move to GPU, convert to float, and normalize
        self.data = torch.from_numpy(self.data).to(self.device).float() / 255.0
        
        # 2. (N, H, W, C) -> (N, C, H, W)
        self.data = self.data.permute(0, 3, 1, 2)
        if FLATTEN:
            self.data = self.data.reshape((self.data.shape[0], -1, 1, 1))
        
        # 3. Move targets to GPU
        # self.targets is a list, so we convert to tensor first
        self.targets = torch.tensor(self.targets, device=self.device)

    def __getitem__(self, index: int):
        image = self.data[index]
        label = self.targets[index]

        # Standard transforms like ToTensor() should be avoided here
        # Use only tensor-compatible transforms (e.g., RandomCrop)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class LocalMNIST(torchvision.datasets.MNIST):
    def __init__(
        self, *args, root=os.path.join(MCUs.get_path('data')), **kwargs
    ):

        FLATTEN = kwargs.pop('FLATTEN', False)
        super().__init__(root=root, *args, **kwargs)

        self.device = device
        self.data = self.data.to(self.device).float()/255.0
        if FLATTEN:
            self.data = self.data.reshape((self.data.shape[0], -1, 1, 1))
        else:
            # add channel dimension
            self.data = self.data.unsqueeze(1)  
        
        self.targets = self.targets.to(self.device)

    @property
    def raw_folder(self) -> str:
        # return os.path.join(self.root, self.__class__.__name__, 'raw')
        return os.path.join(self.root, 'MNIST', 'raw')

    def __getitem__(self, index: int) -> tuple[any, any]:
        """
        Args:
            index (int): Index

        Returns:
            image (): matrix of size image_height x image_width
        """

        # apply any 
        class_id = self.targets[index]
        image = self.data[index]

        if self.target_transform is not None:
            class_id = self.target_transform(class_id)

        # ...
        # image = self._image_proc(image)

        return image, class_id

    def _image_proc(self, image):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image.numpy(), mode="L")

        if self.transform is not None:
            image = self.transform(image)

        return image


# define a new class to work around path issues
class DualMNIST(torchvision.datasets.MNIST):
    def __init__(self, root=os.path.join(MCUs.get_path('data')), *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)

        # ...
        N_classes = len(np.unique(self.targets))

        # the indices sorted by label
        self.index_vector = torch.argsort(self.targets)
        self.nums_instances = np.sum(
            self.targets.numpy() == np.arange(N_classes).reshape([-1, 1]), 1
        )

        # cumulative number of instances *not* including current class
        cums_instances = np.cumsum(self.nums_instances)
        self.cums_instances = np.append(0, cums_instances[:-1])

    @property
    def raw_folder(self) -> str:
        # return os.path.join(self.root, self.__class__.__name__, 'raw')
        return os.path.join(self.root, 'MNIST', 'raw')

    def __getitem__(self, index: int) -> tuple[any, any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image_a, image_b, target) where target is index of the
                target class.
        """
        
        # Interpret index argument as index into the *sorted* examples
        #  NB: By default, this will produce sorted examples.  You need to use
        #  shuffle=True during training
        iExample = index

        # convert to index of *unsorted* data and get the class id
        index_i = self.index_vector[iExample]
        class_id = self.targets[index_i]

        # useful integers
        N_instances = self.nums_instances[class_id]
        N_prev_class_examples = self.cums_instances[class_id]
        iInstance = iExample - N_prev_class_examples
        
        # jInstance = index wrt the first index of this class
        # jExample  = index wrt index 0 (of the sorted data)
        jInstance = (random.randrange(1, N_instances) + iInstance) % N_instances
        jExample = jInstance + N_prev_class_examples

        # convert to index of *unsorted* data
        index_j = self.index_vector[jExample]

        # apply any 
        if self.target_transform is not None:
            class_id = self.target_transform(class_id)

        image_i = self._image_proc(index_i)
        image_j = self._image_proc(index_j)

        return torch.cat((image_i, image_j), axis=0), class_id

    def _image_proc(self, index):
        image = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image.numpy(), mode="L")

        if self.transform is not None:
            image = self.transform(image)
            
        return image


# define a new class to work around path issues
class SplitMNIST(torchvision.datasets.MNIST):
    def __init__(self, root=os.path.join(MCUs.get_path('data')), *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)

    @property
    def raw_folder(self) -> str:
        # return os.path.join(self.root, self.__class__.__name__, 'raw')
        return os.path.join(self.root, 'MNIST', 'raw')

    def __getitem__(self, index: int) -> tuple[any, any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class
                and image is (2 x image_height/2 x image_width)
        """

        # apply any 
        class_id = self.targets[index]
        if self.target_transform is not None:
            class_id = self.target_transform(class_id)

        image = self._image_proc(index)

        # break into top and bottom
        # image = image.reshape(2, -1, image.shape[-1])

        # break into left and right
        image = image.swapaxes(1, 2)
        image = image.reshape(2, -1, image.shape[-1])
        image = image.swapaxes(1, 2)

        return image, class_id

    def _image_proc(self, index):
        image = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image.numpy(), mode="L")

        if self.transform is not None:
            image = self.transform(image)

        return image


class TFRecordDataLoader:
    def __init__(
        self, subnets_params, data_partition, N_cases, OOV_token,
        TARGETS_ARE_SEQUENCES=True,
    ):

        # don't let TF allocate the GPU to itself
        tf.config.set_visible_devices([], 'GPU')
        ds = self._tf_records_to_dataset(
            subnets_params, data_partition, N_cases, OOV_token,
            TARGETS_ARE_SEQUENCES,
            # num_shards_to_discard=0, DROP_REMAINDER=False
        )
        N_batches = 0
        for batch in ds:
            N_batches += 1
        self.N_batches = N_batches
        self.ds = tfds.as_numpy(ds)
        # self.N_cases = N_cases
        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __len__(self):
        return self. N_batches

    def _tf_records_to_dataset(
        self, subnets_params, data_partition, num_cases, OOV_token,
        TARGETS_ARE_SEQUENCES, num_shards_to_discard=0, DROP_REMAINDER=False,
    ):
        '''
        Load, shuffle, batch and pad, and concatentate across subnets (for
        parallel transfer learning) all the data.
        '''

        # accumulate datasets, one for each subnetwork
        dataset_list = []
        for subnet_params in subnets_params:
            dataset = tf.data.TFRecordDataset([
                subnet_params.tf_record_partial_path.format(block_id)
                for block_id in subnet_params.block_ids[data_partition]]
            )
            dataset = dataset.map(
                lambda example_proto: _parse_protobuf_seq2seq_example(
                    example_proto, subnet_params.data_manifests
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # filter data to include or exclude only specified decoder targets?
            decoder_targets_list = subnet_params.data_manifests[
                'decoder_targets'].get_feature_list()
            target_filter = TargetFilter(
                decoder_targets_list, subnet_params.target_specs,
                data_partition
            )
            dataset = target_filter.filter_dataset(dataset)

            # filter out words not in the decoder_targets_list
            if not TARGETS_ARE_SEQUENCES:
                # ...then get rid of OOV examples
                OOV_id = (
                    decoder_targets_list.index(OOV_token)
                    if OOV_token in decoder_targets_list else -1
                )
                # NB that x['decoder_targets'].shape = [None, 1]
                dataset = dataset.filter(
                    lambda x: tf.not_equal(x['decoder_targets'][0, 0], OOV_id)
                )

            # discard some of the data?; shuffle; batch (evening out w/padding)
            if num_shards_to_discard > 0:
                dataset = dataset.shard(num_shards_to_discard+1, 0)
            dataset = dataset.shuffle(buffer_size=35000)  # > greatest
            dataset = dataset.padded_batch(
                num_cases,
                padded_shapes=tf.compat.v1.data.get_output_shapes(dataset),
                padding_values={
                    key: data_manifest.padding_value
                    for key, data_manifest in subnet_params.data_manifests.items()
                },
                drop_remainder=DROP_REMAINDER
            )

            # add id for "proprietary" parts of network under transfer learning
            dataset = dataset.map(
                lambda batch_of_protos_dict: {
                    **batch_of_protos_dict, 'subnet_id': tf.constant(
                        str(subnet_params.subnet_id), dtype=tf.string
                    )
                }
            )
            dataset_list.append(dataset)

        # (randomly) interleave (sub-)batches w/o throwing anything away
        dataset = reduce(
            lambda set_a, set_b: set_a.concatenate(set_b), dataset_list
        )
        dataset = dataset.shuffle(buffer_size=3000)
        ######
        # Since your parse_protobuf_seq2seq_example isn't doing much, the
        #  overhead associated with just scheduling the dataset.map will
        #  dominate the cost of applying it.  Therefore, tensorflow
        #  recommends batching first, and applying a vectorized version of
        #  parse_protobuf_seq2seq_example.  But you shuffle first.....
        ######
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  #num_cases)

        return dataset


def _parse_protobuf_seq2seq_example(example_proto, data_manifests):

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

    return example_dict


class TargetFilter:
    def __init__(self, unique_targets, target_specs, this_data_type):

        '''
        # Example:
        target_specs = {
            'validation': [
                ['this', 'was', 'easy', 'for', 'us'],
                ['they', 'often', 'go', 'out', 'in', 'the', 'evening'],
                ['i', 'honour', 'my', 'mum'],
                ['a', 'doctor', 'was', 'in', 'the', 'ambulance', 'with', 'the', 'patient'],
                ['we', 'are', 'open', 'every', 'monday', 'evening'],
                ['withdraw', 'only', 'as', 'much', 'money', 'as', 'you', 'need'],
                ['allow', 'each', 'child', 'to', 'have', 'an', 'ice', 'pop'],
                ['is', 'she', 'going', 'with', 'you']
            ]
        }
        '''

        # fixed
        data_types = {'training', 'validation'}

        # convert target_specs dictionary entries from word- to index-based
        # NB: PROBABLY NOT GENERAL ENOUGH to work w/non-word_sequence data
        self.target_specs = {key: [
            [unique_targets.index(w + '_') for w in target] + [1]
            for target in target_spec] for key, target_spec in target_specs.items()
        }

        # store for later use
        self.this_data_type = this_data_type
        self.other_data_type = (data_types - {this_data_type}).pop()

    def _test_special(self, fetch_target_indices, data_type):
        # Test if this tf_record target is among this dataset's target_specs.
        # NB that this function returns a (boolean) tf.tensor.
        TEST_SPECIAL = tf.constant(False)
        for target_indices in self.target_specs[data_type]:
            TEST_MATCH = tf.reduce_all(
                tf.linalg.diag_part(tf.equal(
                    fetch_target_indices,
                    np.array(target_indices, ndmin=2))
                ))
            TEST_SPECIAL = tf.logical_or(TEST_SPECIAL, TEST_MATCH)
        return TEST_SPECIAL

    def filter_dataset(self, dataset):
        if self.this_data_type in self.target_specs:
            return dataset.filter(
                lambda example_dict: self._test_special(
                    example_dict['decoder_targets'], self.this_data_type
                ))
        elif self.other_data_type in self.target_specs:
            return dataset.filter(
                lambda example_dict: self._test_special(
                    example_dict['decoder_targets'], self.other_data_type
                ))
        else:
            return dataset


# for consistency with generators that *are* infinite
def infinite_dataloader(loader):
    while True:
        for batch in loader:
            yield batch

#####
# DEPRECATED: too slow
# Importing tfrecords into pytorch
#####
# class TFRecordPipe:
#     from machine_learning.torch_helpers import parse_protobuf_seq2seq_example
#     from torchdata.datapipes.iter import FileLister, FileOpener 
#     from tfrecord.torch.dataset import MultiTFRecordDataset

#     def __init__(
#         self,
#         #####
#         # for now, just one
#         subject
#         #####
#     ):
#         self.data_manifests = subject.data_manifests
#         self.partial_path = subject.data_generator.tf_record_partial_path
#         self.block_ids = subject.block_ids

#     def parse_protobuf(self, example_proto):
#         '''
#         Resizes data and converts words to indices.  NB that all matrices in
#         the example_dict have size [T x N_features]
#         '''

#         example_dict = TFRecordPipe.parse_protobuf_seq2seq_example(
#             example_proto, self.data_manifests,
#         )    

#         return example_dict

#     def pad_collate(self, batch):
#         '''
#         Transforms 
#           list of dictionaries of variable-length sequences
#         into
#           dictionary of tensors
#         (padded to account for variable lengths)

#         Each example in the batch is a dict.  Each value in the dict has size
#             (T_i x N_features).
#         where T_i is the length of that particular example.  The elements of
#         the output, batch_dict, have size
#             (N_cases x T x N_features),
#         where T is the length of the longest sequence in the batch.
#         '''

#         batch_dict = {
#              key: torch.nn.utils.rnn.pad_sequence(
#                 # [torch.tensor(example[key]) for example in batch],
#                 [example[key] for example in batch],
#                 batch_first=True,
#                 padding_value=self.data_manifests[key].padding_value
#              ) for key in self.data_manifests.keys()
#         }
               
#         return batch_dict
    
#     def construct_pipe(self):
#         # vahidk or pytorch version?  Both very slow
#         return self.construct_pipe_v()
#         return self.construct_pipe_t()

#     def construct_pipe_v(self):
#         # ...
#         # index_pattern = self.partial_path.replace('.tfrecord', '.tfindex')
#         # description = {
#         #     'ecog_sequence': 'float',
#         #     'phoneme_sequence': 'byte',
#         #     'text_sequence': 'byte',
#         #     "audio_sequence": 'float',
#         # }
        
#         # unnormalized probabilities
#         splits = {block: 1.0 for block in self.block_ids['training']}

#         # ...
#         dataset = MultiTFRecordDataset(
#             self.partial_path,
#             index_pattern=None,
#             splits=splits,
#             description=None,
#             infinite=False,
#             transform=self.parse_protobuf,
#             shuffle_queue_size=512,
#         )
#         return dataset

#     def construct_pipe_t(self):
#         tf_record_dir, tf_record_name = os.path.split(self.partial_path)
#         datapipe = FileLister(tf_record_dir, tf_record_name.format('*'))
#         datapipe = FileOpener(datapipe, mode="b")
#         datapipe = datapipe.load_from_tfrecord()
#         datapipe = datapipe.shuffle()
#         datapipe = datapipe.sharding_filter()
#         datapipe = datapipe.map(self.parse_protobuf)

#     #     # train, valid = tfrecord_datapipe.random_split(
#     #     #     # total_length=10,
#     #     #     weights={"train": 0.8, "valid": 0.2}, seed=0
#     #     # )

#     #     return datapipe
