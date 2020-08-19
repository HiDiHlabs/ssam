from ._train_semi_supervised_ssam import train

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import yaml


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, vectors, labels=None):
        self.labels = labels
        self.vectors = vectors
        if isinstance(vectors, da.core.Array):
            self.dask = True
        else:
            self.dask = False
            
    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, index):
        v = self.vectors[index]
        if self.dask:
            v = v.compute()
        if self.labels is None:
            l = -1
        else:
            l = self.labels[index]
        return v, l


class AAEClassifier:
    def __init__(self, config_path=None):
        if config_path:
            #self.config_dict = self._load_configuration("aaec/_config.yml")['semi_supervised']
            self.config_dict = self._load_configuration(config_path)
        else:
            self.config_dict = {
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
        self.tensor_dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    def _load_configuration(self, path):
        with open(path, 'r') as f_cfg:
            self.config_dict = yaml.safe_load(f_cfg)

    def train(self, unlabeled_data, labeled_data, labels, n_classes, epochs=1000, batch_size=1000, z_size=5):
        assert unlabeled_data.shape[1] == labeled_data.shape[1]
        
        n_genes = unlabeled_data.shape[1]
        weights = []
        for i in range(n_classes):
            s = np.sum(labels == i)
            if s > 0:
                weights.append(1./s)
            else:
                weights.append(0)
        sampler = WeightedRandomSampler(np.array(weights)[labels], len(labels), replacement=True)
        
        dataset_labeled = _Dataset(labeled_data, labels)
        dataset_unlabeled = _Dataset(unlabeled_data)

        labeled = DataLoader(dataset_labeled, batch_size=batch_size, sampler=sampler)
        unlabeled = DataLoader(dataset_unlabeled, batch_size=batch_size, shuffle=True)
        valid = DataLoader(dataset_labeled, batch_size=batch_size, shuffle=False)
        
        Q, P, learning_curve = train(
            labeled,
            unlabeled,
            valid,
            epochs=epochs,
            n_classes=n_classes,
            n_features=n_genes,
            z_dim=z_size,
            output_dir=None,
            config_dict=self.config_dict
        )
        self.Q = Q
        self.P = P
        self.learning_curve = learning_curve
    
    def predict_labels(self, X):
        batch_size = 10000
        labels = []
        max_probs = []
        X = torch.tensor(X).type(self.tensor_dtype)
        with torch.no_grad():
            for batch_idx in range(0, X.shape[0], batch_size):
                arr = self.Q(X[batch_idx:batch_idx+batch_size])[0].cpu().detach().numpy()
                labels += list(arr.argmax(axis=1))
                max_probs += list(arr.max(axis=1))
        return np.array(labels), np.array(max_probs)