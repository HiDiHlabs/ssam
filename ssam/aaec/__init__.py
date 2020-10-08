from ._train_semi_supervised_ssam import train as train_semi_supervised
from ._train_unsupervised_ssam import train as train_unsupervised

import torch
import yaml
import numpy as np
import sklearn
import dask.array as da
import threading
import multiprocessing
import ctypes


# https://gist.github.com/liuw/2407154
def _ctype_async_raise(thread_obj, exception):
    found = False
    target_tid = 0
    for tid, tobj in threading._active.items():
        if tobj is thread_obj:
            found = True
            target_tid = tid
            break

    if not found:
        raise ValueError("Invalid thread object")

    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_tid), ctypes.py_object(exception))
    # ref: http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc
    if ret == 0:
        raise ValueError("Invalid thread ID")
    elif ret > 1:
        # Huh? Why would we notify more than one threads?
        # Because we punch a hole into C level interpreter.
        # So it is better to clean up the mess.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, NULL)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    #print("Successfully set asynchronized exception for", target_tid)


class _ChunkedDataset(torch.utils.data.IterableDataset):
    def __init__(self, vectors, labels=None, shuffle=True, normalized=False, chunk_size=1000, sample_size=0):
        self.vectors = vectors
        if not isinstance(self.vectors, da.core.Array):
            self.vectors = da.array(self.vectors)
        self.labels = labels
        self.normalized = normalized
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        if sample_size > 0 and sample_size < len(self.vectors):
            if not self.shuffle:
                print("Warning: 'sample_size' is smaller than the data size. 'shuffle' flag has been turned on.")
                self.shuffle = True
            self.sample_size = sample_size
        else:
            self.sample_size = len(self.vectors)
        self.proc = None

    def __iter__(self):
        try:
            seq_indices = np.arange(self.vectors.shape[0])
            if self.shuffle:
                np.random.shuffle(seq_indices)
            seq_indices = seq_indices[:self.sample_size]
            chunked_indices = [sorted(seq_indices[i:i+self.chunk_size]) for i in range(0, self.sample_size, self.chunk_size)]

            self.load_next_chunk_async(chunked_indices[0])
            for chunk_idx in range(len(chunked_indices)):
                chunk = self.get_chunk()
                if chunk_idx < len(chunked_indices) - 1:
                    self.load_next_chunk_async(chunked_indices[chunk_idx+1])
                if self.labels is not None:
                    chunked_labels = self.labels[chunked_indices[chunk_idx]]
                for i in range(chunk.shape[0]):
                    if self.labels is None:
                        l = -1
                    else:
                        l = chunked_labels[i]
                    yield chunk[i], l
        except KeyboardInterrupt:
            if self.proc:
                print("Recived KeyboardInterrupt, trying to stop the loader thread...")
                _ctype_async_raise(self.proc, KeyboardInterrupt)
                self.proc.join()
            raise KeyboardInterrupt
            
    def get_chunk(self):
        self.proc.join()
        arr = np.frombuffer(self.buffer, dtype='float32').reshape(self.chunk_shape)
        rtn = np.zeros_like(arr, dtype='float32')
        np.copyto(rtn, arr)
        return rtn
    
    def load_next_chunk_async(self, idx):
        self.buffer = multiprocessing.RawArray('f', len(idx) * self.vectors.shape[1])
        self.chunk_shape = [len(idx), self.vectors.shape[1]]
        self.proc = threading.Thread(target=self._proc, args=(idx, ))
        self.proc.daemon = True
        self.proc.start()
        
    def _proc(self, idx):
        data = self.vectors[idx].compute().astype('float32')
        if self.normalized:
            data = sklearn.preprocessing.normalize(data, norm='l2', axis=1)
        arr = np.frombuffer(self.buffer, dtype='float32').reshape(data.shape)
        np.copyto(arr, data)
        
    def __len__(self):
        return min(self.sample_size, len(self.vectors))
    
    
class _ChunkedRandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, vectors, labels=None, normalized=False, chunk_size=100000, sample_size=0):
        self.vectors = vectors
        if not isinstance(self.vectors, da.core.Array):
            self.vectors = da.array(self.vectors)
        self.labels = labels
        self.normalized = normalized
        self.chunk_size = chunk_size
        if sample_size > 0 and sample_size < len(self.vectors):
            self.sample_size = sample_size
        else:
            self.sample_size = len(self.vectors)
        self.proc = None
        self.load_next_chunk_async()
        self._cursor = -1

    def __iter__(self):
        try:
            sample_cnt = 0
            while True:
                chunk, indices = self.get_chunk()
                if self._cursor == -1:
                    self.load_next_chunk_async()
                    self._cursor = 0
                if self.labels is not None:
                    chunked_labels = self.labels[indices]
                for i in range(self._cursor, chunk.shape[0]):
                    if self.labels is None:
                        l = -1
                    else:
                        l = chunked_labels[i]
                    self._cursor = i + 1
                    sample_cnt += 1
                    yield chunk[i], l
                    if sample_cnt == self.sample_size:
                        return
                self._cursor = -1
        except KeyboardInterrupt:
            if self.proc:
                print("Recived KeyboardInterrupt, trying to stop the loader thread...")
                _ctype_async_raise(self.proc, KeyboardInterrupt)
                self.proc.join()
            raise KeyboardInterrupt
            
    def get_chunk(self):
        if self.proc.is_alive():
            self.proc.join()
        arr = np.frombuffer(self.buffer, dtype='float32').reshape(self.chunk_shape)
        rtn = np.zeros_like(arr, dtype='float32')
        np.copyto(rtn, arr)
        return rtn, np.array(self.rand_indices, copy=True)
    
    def load_next_chunk_async(self):
        self.rand_indices = sorted(np.random.randint(0, self.vectors.shape[0], self.chunk_size))
        self.buffer = multiprocessing.RawArray('f', len(self.rand_indices) * self.vectors.shape[1])
        self.chunk_shape = [len(self.rand_indices), self.vectors.shape[1]]
        self.proc = threading.Thread(target=self._proc)
        self.proc.daemon = True
        self.proc.start()
        
    def _proc(self):
        idx = self.rand_indices
        data = self.vectors[idx].compute().astype('float32')
        if self.normalized:
            data = sklearn.preprocessing.normalize(data, norm='l2', axis=1)
        arr = np.frombuffer(self.buffer, dtype='float32').reshape(data.shape)
        np.copyto(arr, data)
        
    def __len__(self):
        return self.sample_size


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, vectors, labels=None, normalized=False):
        self.labels = labels
        self.vectors = vectors
        if normalized:
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

    def train(self, n_classes, unlabeled_data, labeled_data=None, labels=None, epochs=1000, batch_size=1000, z_dim=2, sample_size=0, chunk_size=10000, normalized=False, beta=0, noise=0, use_forget_labels=False):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        
        n_genes = unlabeled_data.shape[1]
        if sample_size == 0:
            sample_size = len(unlabeled_data)
        if labeled_data is None:
            if sample_size == 0 or sample_size == len(unlabeled_data):
                dataset_unlabeled = _ChunkedDataset(unlabeled_data, normalized=normalized, chunk_size=chunk_size)
            else:
                dataset_unlabeled = _ChunkedRandomDataset(unlabeled_data, normalized=normalized, chunk_size=chunk_size, sample_size=sample_size)
            unlabeled = torch.utils.data.DataLoader(dataset_unlabeled, batch_size=batch_size)
            
            Q, P, learning_curve = train_unsupervised(
                unlabeled,
                epochs=epochs,
                n_classes=n_classes,
                n_features=n_genes,
                z_dim=z_dim,
                noise=noise,
                output_dir=None,
                config_dict=self.config_dict['unsupervised'],
                verbose=self.verbose
            )
        else:
            assert unlabeled_data.shape[1] == labeled_data.shape[1]
            uniq_labels, samples_per_cls = np.unique(labels, return_counts=True)
            if uniq_labels[0] == -1:
                uniq_labels = uniq_labels[1:]
                samples_per_cls = samples_per_cls[1:]
                labels = np.array(labels)
                labeled_data = np.array(labeled_data)[labels != -1]
                labels = labels[labels != -1]
            dataset_labeled = _Dataset(labeled_data, labels, normalized=normalized)
            batch_size = min(batch_size, len(dataset_labeled))
            sample_size = len(dataset_labeled)
            if beta > 0:
                effective_num = 1.0 - np.power(beta, samples_per_cls)
                weights = (1.0 - beta) / effective_num
                #weights = weights / np.sum(weights) * len(samples_per_cls)
                sampler = torch.utils.data.WeightedRandomSampler(weights[labels], len(labels), replacement=True)
                labeled = torch.utils.data.DataLoader(dataset_labeled, batch_size=batch_size, sampler=sampler)
            else:
                labeled = torch.utils.data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=True)
            valid = torch.utils.data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=False)

            dataset_unlabeled = _ChunkedRandomDataset(unlabeled_data, normalized=normalized, chunk_size=chunk_size, sample_size=sample_size)
            unlabeled = torch.utils.data.DataLoader(dataset_unlabeled, batch_size=batch_size)
            
            Q, P, learning_curve = train_semi_supervised(
                labeled,
                unlabeled,
                valid,
                epochs=epochs,
                n_classes=n_classes,
                n_features=n_genes,
                z_dim=z_dim,
                noise=noise,
                output_dir=None,
                config_dict=self.config_dict['semi_supervised'],
                verbose=self.verbose,
                use_forget_labels=use_forget_labels
            )
        self.Q = Q
        self.P = P
        self.learning_curve = learning_curve

    def predict_labels(self, X, n=1, normalized=False, chunk_size=10000):
        if isinstance(X, da.core.Array):
            dask = True
        else:
            dask = False
            
        labels = np.zeros([0, n], dtype=int)
        max_probs = np.zeros([0, n], dtype=float)
        with torch.no_grad():
            for chunk_idx in range(0, X.shape[0], chunk_size):
                X_chunk = X[chunk_idx:chunk_idx+chunk_size]
                if dask:
                    X_chunk = X_chunk.compute()
                if normalized:
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