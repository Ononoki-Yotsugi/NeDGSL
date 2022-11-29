import os.path as osp
import os
import pickle as pkl
import sys
import numpy as np
import scipy.sparse as sp
import networkx as nx
from dgl.data.utils import generate_mask_tensor
from dgl import from_networkx
import dgl.backend as F
import urllib.request


def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask


def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.asarray(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense())


def download_cora(root, name):
    url = 'https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/'
    try:
        print('Downloading', url)
        urllib.request.urlretrieve(url + name, osp.join(root, name))
        print('Done!')
    except:
        raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')


class CoraData:

    def __init__(self, root, verbose):
        self.root = root
        self.verbose = verbose
        self.load_cora()

    def load_cora(self):
        if not osp.exists(self.root):
            os.makedirs(self.root)
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = "ind.{}.{}".format('cora', names[i])
            data_filename = osp.join(self.root, name)

            if not osp.exists(data_filename):
                download_cora(self.root, name)

            with open(data_filename, 'rb') as f:
                objects.append(_pickle_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_file = "ind.{}.test.index".format('cora')
        if not osp.exists(osp.join(self.root, test_idx_file)):
            download_cora(self.root, test_idx_file)
        test_idx_reorder = _parse_index_file(osp.join(self.root, test_idx_file))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        graph = nx.DiGraph(nx.from_dict_of_lists(graph))

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = generate_mask_tensor(_sample_mask(idx_train, labels.shape[0]))
        val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0]))
        test_mask = generate_mask_tensor(_sample_mask(idx_test, labels.shape[0]))

        self._g = from_networkx(graph)

        self._g.ndata['train_mask'] = train_mask
        self._g.ndata['val_mask'] = val_mask
        self._g.ndata['test_mask'] = test_mask
        self._g.ndata['label'] = F.tensor(labels)
        self._g.ndata['feat'] = F.tensor(_preprocess_features(features), dtype=F.data_type_dict['float32'])
        self.num_classes = onehot_labels.shape[1]

        if self.verbose:
            print('Finished data loading and preprocessing.')
            print('  NumNodes: {}'.format(self._g.number_of_nodes()))
            print('  NumEdges: {}'.format(self._g.number_of_edges()))
            print('  NumFeats: {}'.format(self._g.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))
            print('  NumTrainingSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['test_mask']).shape[0]))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

if __name__ == '__main__':
    CoraData('./cora', True)

