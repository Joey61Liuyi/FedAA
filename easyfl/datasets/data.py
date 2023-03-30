import copy
import importlib
import json
import logging
import os
import random
import numpy as np
import torch
from torchvision.transforms import ToPILImage, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
import torchvision
from torch.utils.data import Dataset
from easyfl.datasets.dataset import FederatedTensorDataset
from easyfl.datasets.utils.base_dataset import BaseDataset, CIFAR10, CIFAR100
from easyfl.datasets.utils.util import load_dict, save_dict
from PIL import Image

logger = logging.getLogger(__name__)


def read_dir(data_dir):
    clients = []
    groups = []
    data = {}

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(dataset_name, train_data_dir, test_data_dir):
    """Load datasets from data directories.

    Args:
        dataset_name (str): The name of the dataset.
        train_data_dir (str): The directory of training data.
        test_data_dir (str): The directory of testing data.

    Returns:
        list[str]: A list of client ids.
        list[str]: A list of group ids for dataset with hierarchies.
        dict: A dictionary of training data, e.g., {"id1": {"x": data, "y": label}, "id2": {"x": data, "y": label}}.
        dict: A dictionary of testing data. The format is same as training data for FEMNIST and Shakespeare datasets.
            For CIFAR datasets, the format is {"x": data, "y": label}, for centralized testing in the server.
    """
    if dataset_name == CIFAR10 or dataset_name == CIFAR100:
        train_data = load_dict(train_data_dir)
        test_data = load_dict(test_data_dir)
        return [], [], train_data, test_data

    # Data in the directories are `json` files with keys `users` and `user_data`.
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def load_data(root,
              dataset_name,
              num_of_clients,
              split_type,
              min_size,
              class_per_client,
              data_amount,
              iid_fraction,
              user,
              train_test_split,
              quantity_weights,
              alpha):
    """Simulate and load federated datasets.

    Args:
        root (str): The root directory where datasets stored.
        dataset_name (str): The name of the dataset. It currently supports: femnist, shakespeare, cifar10, and cifar100.
            Among them, femnist and shakespeare are adopted from LEAF benchmark.
        num_of_clients (int): The targeted number of clients to construct.
        split_type (str): The type of statistical simulation, options: iid, dir, and class.
            `iid` means independent and identically distributed data.
            `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
            `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
            `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
        min_size (int): The minimal number of samples in each client.
            It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
        class_per_client (int): The number of classes in each client. Only applicable when the split_type is 'class'.
        data_amount (float): The fraction of data sampled for LEAF datasets.
            e.g., 10% means that only 10% of total dataset size are used.
        iid_fraction (float): The fraction of the number of clients used when the split_type is 'iid'.
        user (bool): A flag to indicate whether partition users of the dataset into train-test groups.
            Only applicable to LEAF datasets.
            True means partitioning users of the dataset into train-test groups.
            False means partitioning each users' samples into train-test groups.
        train_test_split (float): The fraction of data for training; the rest are for testing.
            e.g., 0.9 means 90% of data are used for training and 10% are used for testing.
        quantity_weights (list[float]): The targeted distribution of quantities to simulate data quantity heterogeneity.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            The `num_of_clients` should be divisible by `len(weights)`.
            None means clients are simulated with the same data quantity.
        alpha (float): The parameter for Dirichlet distribution simulation, applicable only when split_type is `dir`.

    Returns:
        dict: A dictionary of training data, e.g., {"id1": {"x": data, "y": label}, "id2": {"x": data, "y": label}}.
        dict: A dictionary of testing data.
        function: A function to preprocess training data.
        function: A function to preprocess testing data.
        torchvision.transforms.transforms.Compose: Training data transformation.
        torchvision.transforms.transforms.Compose: Testing data transformation.
    """
    user_str = "user" if user else "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_file = os.path.join(dir_path, "data_process", "{}.py".format(dataset_name))
    if not os.path.exists(dataset_file):
        logger.error("Please specify a valid process file path for process_x and process_y functions.")
    dataset_path = "easyfl.datasets.data_process.{}".format(dataset_name)
    dataset_lib = importlib.import_module(dataset_path)
    process_x = getattr(dataset_lib, "process_x", None)
    process_y = getattr(dataset_lib, "process_y", None)
    transform_train = getattr(dataset_lib, "transform_train", None)
    transform_test = getattr(dataset_lib, "transform_test", None)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    train_data_dir = os.path.join(data_dir, setting, "train")
    test_data_dir = os.path.join(data_dir, setting, "test")

    if not os.path.exists(train_data_dir) or not os.path.exists(test_data_dir):
        dataset_class_path = "easyfl.datasets.{}.{}".format(dataset_name, dataset_name)
        dataset_class_lib = importlib.import_module(dataset_class_path)
        class_name = dataset_name.capitalize()
        dataset = getattr(dataset_class_lib, class_name)(root=data_dir,
                                                         fraction=data_amount,
                                                         split_type=split_type,
                                                         user=user,
                                                         iid_user_fraction=iid_fraction,
                                                         train_test_split=train_test_split,
                                                         minsample=min_size,
                                                         num_of_client=num_of_clients,
                                                         class_per_client=class_per_client,
                                                         setting_folder=setting,
                                                         alpha=alpha,
                                                         weights=quantity_weights)
        try:
            filename = f"{setting}.zip"
            dataset.download_packaged_dataset_and_extract(filename)
            logger.info(f"Downloaded packaged dataset {dataset_name}: {filename}")
        except Exception as e:
            logger.info(f"Failed to download packaged dataset: {e.args}")

        # CIFAR10 generate data in setup() stage, LEAF related datasets generate data in sampling()
        if not os.path.exists(train_data_dir):
            dataset.setup()
        if not os.path.exists(train_data_dir):
            dataset.sampling()

    users, train_groups, train_data, test_data = read_data(dataset_name, train_data_dir, test_data_dir)
    return train_data, test_data, process_x, process_y, transform_train, transform_test



def data_organize(idxs_labels, labels):
    data_dict = {}

    labels = np.unique(labels, axis=0)
    for one in labels:
        data_dict[one] = []

    for i in range(len(idxs_labels[1, :])):
        data_dict[idxs_labels[1, i]].append(idxs_labels[0, i])
    return data_dict


def data_partition(training_data, testing_data, alpha, user_num):
    idxs_train = np.arange(len(training_data))
    idxs_valid = np.arange(len(testing_data))

    if hasattr(training_data, 'targets'):
        labels_train = training_data.targets
        labels_valid = testing_data.targets
    elif hasattr(training_data, 'labels'):
        labels_train = training_data.labels
        labels_valid = testing_data.labels

    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1,:].argsort()]
    idxs_labels_valid = np.vstack((idxs_valid, labels_valid))
    idxs_labels_valid = idxs_labels_valid[:, idxs_labels_valid[1,:].argsort()]

    labels = np.unique(labels_train, axis=0)

    data_train_dict = data_organize(idxs_labels_train, labels)
    data_valid_dict = data_organize(idxs_labels_valid, labels)

    data_partition_profile_train = {}
    data_partition_profile_valid = {}

    for i in range(user_num):
        data_partition_profile_train[i] = []
        data_partition_profile_valid[i] = []

    # ## Setting the public data
    # public_data = set([])
    # for label in data_train_dict:
    #     tep = set(np.random.choice(data_train_dict[label], int(len(data_train_dict[label])/20), replace = False))
    #     public_data = set.union(public_data, tep)
    #     data_train_dict[label] = list(set(data_train_dict[label])-tep)
    #
    # public_data = list(public_data)
    # np.random.shuffle(public_data)


    ## Distribute rest data
    for label in data_train_dict:
        proportions = np.random.dirichlet(np.repeat(alpha, user_num))
        proportions_train = len(data_train_dict[label])*proportions
        proportions_valid = len(data_valid_dict[label]) * proportions

        for user in data_partition_profile_train:

            data_partition_profile_train[user]   \
                = set.union(set(np.random.choice(data_train_dict[label], int(proportions_train[user]) , replace = False)), data_partition_profile_train[user])
            data_train_dict[label] = list(set(data_train_dict[label])-data_partition_profile_train[user])


            data_partition_profile_valid[user] = set.union(set(
                np.random.choice(data_valid_dict[label], int(proportions_valid[user]),
                                 replace=False)), data_partition_profile_valid[user])
            data_valid_dict[label] = list(set(data_valid_dict[label]) - data_partition_profile_valid[user])


        while len(data_train_dict[label]) != 0:
            rest_data = data_train_dict[label][0]
            user = np.random.randint(0, user_num)
            data_partition_profile_train[user].add(rest_data)
            data_train_dict[label].remove(rest_data)

        while len(data_valid_dict[label]) != 0:
            rest_data = data_valid_dict[label][0]
            user = np.random.randint(0, user_num)
            data_partition_profile_valid[user].add(rest_data)
            data_valid_dict[label].remove(rest_data)

    for user in data_partition_profile_train:
        data_partition_profile_train[user] = list(data_partition_profile_train[user])
        data_partition_profile_valid[user] = list(data_partition_profile_valid[user])
        np.random.shuffle(data_partition_profile_train[user])
        np.random.shuffle(data_partition_profile_valid[user])

    profile_train = {}
    profile_valid = {}

    for client in data_partition_profile_train:
        profile_train['f' + str(client).zfill(7)] = data_partition_profile_train[client]

    for client in data_partition_profile_valid:
        profile_valid['f' + str(client).zfill(7)] = data_partition_profile_valid[client]

    return profile_train, profile_valid



class MiniImageNetDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_dict = {}
        self.image_size = (64, 64)


        for label, folder_name in enumerate(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)
        # for file_name in os.listdir(data_dir):
        #     if file_name.endswith('.jpg'):
        #         image_path = os.path.join(data_dir, file_name)
        #         self.image_paths.append(image_path)
        #         # Extract the label from the file name
        #         label = file_name[:9]  # Assuming the label is the first 9 characters of the file name
        #         if label not in self.label_dict:
        #             self.label_dict[label] = len(self.label_dict)
        #         self.labels.append(self.label_dict[label])

    def __getitem__(self, index):
        # Load the image and convert it to a numpy array
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size)

        # Get the label
        label = self.labels[index]

        # Convert the image and label to torch tensors and return them
        image = self.transform(image)
        label = torch.tensor(label).long()

        return image, label

    def __len__(self):
        return len(self.image_paths)

def from_index_to_x_y(index, dataset):

    result = {'x': [], 'y': []}
    for i in index:
        path = dataset.image_paths[i]
        img = Image.open(path).convert('RGB')
        img = img.resize((224,224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        result['x'].append(img_array)
        result['y'].append(dataset.labels[i])

    result['x'] = np.concatenate(result['x'], axis=0)
    result['y'] = np.array(result['y'])
    return result


def construct_datasets(root,
                       dataset_name,
                       num_of_clients,
                       split_type,
                       min_size,
                       class_per_client,
                       data_amount,
                       iid_fraction,
                       user,
                       train_test_split,
                       quantity_weights,
                       alpha):
    """Construct and load provided federated learning datasets.

    Args:
        root (str): The root directory where datasets stored.
        dataset_name (str): The name of the dataset. It currently supports: femnist, shakespeare, cifar10, and cifar100.
            Among them, femnist and shakespeare are adopted from LEAF benchmark.
        num_of_clients (int): The targeted number of clients to construct.
        split_type (str): The type of statistical simulation, options: iid, dir, and class.
            `iid` means independent and identically distributed data.
            `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
            `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
            `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
        min_size (int): The minimal number of samples in each client.
            It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
        class_per_client (int): The number of classes in each client. Only applicable when the split_type is 'class'.
        data_amount (float): The fraction of data sampled for LEAF datasets.
            e.g., 10% means that only 10% of total dataset size are used.
        iid_fraction (float): The fraction of the number of clients used when the split_type is 'iid'.
        user (bool): A flag to indicate whether partition users of the dataset into train-test groups.
            Only applicable to LEAF datasets.
            True means partitioning users of the dataset into train-test groups.
            False means partitioning each users' samples into train-test groups.
        train_test_split (float): The fraction of data for training; the rest are for testing.
            e.g., 0.9 means 90% of data are used for training and 10% are used for testing.
        quantity_weights (list[float]): The targeted distribution of quantities to simulate data quantity heterogeneity.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            The `num_of_clients` should be divisible by `len(weights)`.
            None means clients are simulated with the same data quantity.
        alpha (float): The parameter for Dirichlet distribution simulation, applicable only when split_type is `dir`.

    Returns:
        :obj:`FederatedDataset`: Training dataset.
        :obj:`FederatedDataset`: Testing dataset.
    """

    if dataset_name == 'mini_imagenet':
        user_str = "user" if user else "sample"
        setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                                 data_amount, iid_fraction, user_str, train_test_split, alpha,
                                                 quantity_weights)
        data_dir = os.path.join(root, dataset_name)
        data_dir = os.path.join(data_dir, setting)

        if os.path.exists(data_dir):
            train_data = load_dict(os.path.join(data_dir, 'train'))
            test_data = load_dict(os.path.join(data_dir, 'test'))

            for client in train_data:
                for i in train_data[client]:
                    train_data[client][i] = np.load(train_data[client][i])['data']
            for i in test_data:
                test_data[i] = np.load(test_data[i])['data']

        else:

            dataset = MiniImageNetDataset('./mini_imagenet/')

            num_samples = len(dataset.image_paths)
            indices = np.arange(num_samples)
            train_indices = random.sample(list(indices), 50000)
            test_indices = list(set(indices) - set(train_indices))
            train_data = copy.deepcopy(dataset)
            train_data.image_paths = [train_data.image_paths[i] for i in train_indices]
            train_data.labels = [train_data.labels[i] for i in train_indices]
            test_data = copy.deepcopy(dataset)
            test_data.image_paths = [test_data.image_paths[i] for i in test_indices]
            test_data.labels = [test_data.labels[i] for i in test_indices]

            train_index, test_index = data_partition(train_data, test_data, alpha, num_of_clients)

            for client in train_index:
                train_index[client] = from_index_to_x_y(train_index[client], dataset)

            train_data = train_index
            test_data = from_index_to_x_y(test_indices, dataset)

            os.makedirs(data_dir)
            save_dict(train_data, os.path.join(data_dir, 'train'))
            save_dict(test_data, os.path.join(data_dir, 'test'))

        transform_train = torchvision.transforms.Compose([
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        transform_test = torchvision.transforms.Compose([
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        process_x = None
        process_y = None


    else:

        train_data, test_data, process_x, process_y, transform_train, transform_test = load_data(root,
                                                                                                 dataset_name,
                                                                                                 num_of_clients,
                                                                                                 split_type,
                                                                                                 min_size,
                                                                                                 class_per_client,
                                                                                                 data_amount,
                                                                                                 iid_fraction,
                                                                                                 user,
                                                                                                 train_test_split,
                                                                                                 quantity_weights,
                                                                                                 alpha)
    # CIFAR datasets are simulated.
    public_data = {'x':[], 'y':[]}
    partio = 0.05

    for client in train_data:
        num = len(train_data[client]['y'])
        num_index = int(num*partio)
        public_index = np.random.choice(num, num_index, replace=False)
        res_index = list(set(range(num))-set(public_index))
        # public_index = np.random.choice(rest_index, int(num*partio/10))

        public_data['x'].append(train_data[client]['x'][public_index])
        public_data['y'].append(train_data[client]['y'][public_index])

        train_data[client]['x'] = train_data[client]['x'][res_index]
        train_data[client]['y'] = train_data[client]['y'][res_index]

    public_data['x'] = np.concatenate(public_data['x'])
    public_data['y'] = np.concatenate(public_data['y'])

    test_simulated = False

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        process_x=process_x,
                                        process_y=process_y,
                                        transform=transform_train)
    test_data = FederatedTensorDataset(test_data,
                                       simulated=test_simulated,
                                       do_simulate=False,
                                       process_x=process_x,
                                       process_y=process_y,
                                       transform=transform_test)

    public_data = FederatedTensorDataset(public_data,simulated=test_simulated,
                                       do_simulate=False,
                                       process_x=process_x,
                                       process_y=process_y,
                                       transform=transform_test)

    return train_data, test_data, public_data
