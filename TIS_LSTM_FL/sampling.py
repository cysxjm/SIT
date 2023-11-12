import numpy as np


def get_indices(labels, user_labels, n_samples):
    indices = []
    for selected_label in user_labels:
        label_samples = np.where(labels[1, :] == selected_label)
        label_indices = labels[0, label_samples]
        selected_indices = list(np.random.choice(label_indices[0], n_samples, replace=False))
        indices += selected_indices
    return indices


def get_samples(indices, n_samples):
    selected_indices = list(np.random.choice(indices, n_samples, replace=False))
    return selected_indices


def iid_onepass(dataset_train, dataset_train_size, dataset_test, dataset_test_size, num_users, dataset_name='mnist'):
    train_users = {}
    test_users = {}

    if dataset_name == 'loop':
        train_idxs = np.arange(len(dataset_train))
        test_idxs = np.arange(len(dataset_test))

        for i in range(num_users):
            train_indices = get_samples(train_idxs, n_samples=dataset_train_size)
            test_indices = get_samples(test_idxs, n_samples=dataset_test_size)
            train_users[i] = train_indices
            test_users[i] = test_indices
        return train_users, test_users
