from ._train_semi_supervised_ssam import train as train_semi_supervised
from ._train_unsupervised_ssam import train as train_unsupervised

import torch
import yaml
import numpy as np
import sklearn
import dask.array as da


class _ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, vectors, labels=None, shuffle=True, normalize=True, chunk_size=1000, random_seed=0, size_limit=-1):
        self.vectors = vectors
        self.labels = labels
        self.normalize = normalize
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.size_limit = size_limit

    def __iter__(self):
        if isinstance(self.vectors, da.core.Array):
            dask = True
        else:
            dask = False
        seq_indices = np.arange(self.vectors.shape[0])
        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(seq_indices)
        chunk_indices = [sorted(seq_indices[i:i+self.chunk_size]) for i in range(0, self.vectors.shape[0], self.chunk_size)]
        cnt = 0
        for s in chunk_indices:
            if self.size_limit > -1 and cnt >= self.size_limit:
                break
            chunk = self.vectors[s]
            if dask:
                chunk = chunk.compute()
            if self.normalize:
                chunk = sklearn.preprocessing.normalize(chunk, norm='l2', axis=1)
            if self.labels is not None:
                chunk_labels = self.labels[s]
            for i in range(chunk.shape[0]):
                if self.size_limit > -1 and cnt >= self.size_limit:
                    break
                if self.labels is None:
                    l = -1
                else:
                    l = chunk_labels[i]
                yield chunk[i], l
                cnt += 1
    
    def __len__(self):
        return len(self.vectors)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, vectors, labels=None, normalize=True):
        self.labels = labels
        self.vectors = vectors
        if normalize:
            self.vectors = sklearn.preprocessing.normalize(self.vectors, norm='l2', axis=1)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, index):
        v = self.vectors[index]
        if self.labels is None:
            l = -1
        else:
            l = self.labels[index]
        return v, l


class AAEClassifier:
    def __init__(self, config_path=None, random_seed=0, verbose=True):
        self.random_seed = random_seed
        if config_path:
            #self.config_dict = self._load_configuration("aaec/_config.yml")['semi_supervised']
            self.config_dict = self._load_configuration(config_path)
        else:
            self.config_dict = {
                'unsupervised': {
                    'learning_rates': {
                        'auto_encoder_lr': 0.0008,
                        'generator_lr': 0.002,
                        'discriminator_lr': 0.0002,
                        'info_lr': 0.00001,
                        'mode_lr': 0.0008,
                        'disentanglement_lr': 0.000005
                    },
                    'model': {
                        'hidden_size': 3000,
                        'encoder_dropout': 0.2
                    },
                    'training': {
                        'use_mutual_info': False,
                        'use_mode_decoder': False,
                        'use_disentanglement': True,
                        'use_adam_optimization': True,
                        'use_adversarial_categorial_weights': False,
                        'lambda_z_l2_regularization': 0.15
                    }
                },
                'semi_supervised': {
                    'learning_rates': {
                        'auto_encoder_lr': 0.0008,
                        'generator_lr': 0.001,
                        'discriminator_lr': 0.0002,
                        'classifier_lr': 0.001
                    },
                    'model': {
                        'hidden_size': 1000,
                        'encoder_dropout': 0
                    }
                }
            }
        self.tensor_dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.verbose = verbose
    
    def _load_configuration(self, path):
        with open(path, 'r') as f_cfg:
            self.config_dict = yaml.safe_load(f_cfg)

    def train(self, n_classes, unlabeled_data, labeled_data=None, labels=None, epochs=1000, batch_size=1000, z_size=5, normalize=True, weighted=False):
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        
        n_genes = unlabeled_data.shape[1]
        
        size_limit = 10000
        if labeled_data is not None:
            assert unlabeled_data.shape[1] == labeled_data.shape[1]
            dataset_labeled = _Dataset(labeled_data, labels, normalize=normalize)
            if weighted:
                weights = []
                for i in range(n_classes):
                    s = np.sum(labels == i)
                    if s > 0:
                        weights.append(1./s)
                    else:
                        weights.append(0)
                sampler = torch.utils.data.WeightedRandomSampler(np.array(weights)[labels], len(labels), replacement=True)
                labeled = torch.utils.data.DataLoader(dataset_labeled, batch_size=batch_size, sampler=sampler)
            else:
                labeled = torch.utils.data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=True)
            valid = torch.utils.data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=False)
            size_limit = len(dataset_labeled)

        dataset_unlabeled = _ChunkedDataset(unlabeled_data, shuffle=True, normalize=normalize, chunk_size=int(np.ceil(size_limit / batch_size) * batch_size), random_seed=self.random_seed, size_limit=size_limit)

        unlabeled = torch.utils.data.DataLoader(dataset_unlabeled, batch_size=batch_size)
        
        if labeled_data is None:
            Q, P, P_mode_decoder, learning_curve = train_unsupervised(
                unlabeled,
                epochs=epochs,
                n_classes=n_classes,
                n_features=n_genes,
                z_dim=z_size,
                output_dir=None,
                config_dict=self.config_dict['unsupervised'],
                verbose=self.verbose
            )
            self.P_mode_decoder = P_mode_decoder
        else:
            Q, P, learning_curve = train_semi_supervised(
                labeled,
                unlabeled,
                valid,
                epochs=epochs,
                n_classes=n_classes,
                n_features=n_genes,
                z_dim=z_size,
                output_dir=None,
                config_dict=self.config_dict['semi_supervised'],
                verbose=self.verbose
            )
        self.Q = Q
        self.P = P
        self.learning_curve = learning_curve

    def predict_labels(self, X, n=1, normalize=True):
        if isinstance(X, da.core.Array):
            dask = True
        else:
            dask = False
        chunk_size = 10000
        labels = np.zeros([0, n], dtype=int)
        max_probs = np.zeros([0, n], dtype=float)
        with torch.no_grad():
            for chunk_idx in range(0, X.shape[0], chunk_size):
                X_chunk = X[chunk_idx:chunk_idx+chunk_size]
                if dask:
                    X_chunk = X_chunk.compute()
                if normalize:
                    X_chunk = sklearn.preprocessing.normalize(X_chunk, norm='l2', axis=1)
                X_chunk = torch.tensor(X_chunk).type(self.tensor_dtype)
                arr = self.Q(X_chunk)[0].cpu().detach().numpy()
                max_indices = arr.argsort(axis=1)
                labels_chunk = np.zeros([arr.shape[0], n], dtype=int)
                max_probs_chunk = np.zeros([arr.shape[0], n], dtype=float)
                for i in range(n):
                    labels_chunk[:, i] = max_indices[:, -i-1]
                    max_probs_chunk[:, i] = arr[np.arange(len(arr)), max_indices[:, -i-1]]
                labels = np.vstack([labels, labels_chunk])
                max_probs = np.vstack([max_probs, max_probs_chunk])
        return np.array(labels), np.array(max_probs)