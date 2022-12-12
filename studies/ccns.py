from data.pyg_load import pyg_load_dataset
import torch
import numpy as np

ds_name = 'texas'
verbose = True
device = torch.device('cuda')


# Different smoothing items
# Euclidean Distance
def euclidean(mat, adj=None, mat_cal=True):
    if mat_cal:
        return adj*torch.norm(mat.unsqueeze(1)-mat.unsqueeze(0), p=2, dim=2)
    else:
        if adj.is_sparse:
            edges = adj.indices().cpu().numpy()
            values = torch.zeros(int(edges.shape[1]/2)).to(device)
            for i in range(edges.shape[1]):
                v1 = edges[0,i]
                v2 = edges[1,i]
                values[dic[(v1,v2)]] = ((mat[v1]-mat[v2])**2).sum()
            return values.cpu().numpy()
        else:
            distances = torch.zeros((n_nodes, n_nodes)).to(device)
            for v1, v2 in adj.nonzero():
                distances[v1,v2] = ((mat[v1]-mat[v2])**2).sum()
            return distances.cpu().numpy()

def normalized_euclidean(mat, adj, mat_cal=True):
    if mat_cal:
        # error
        return adj*torch.norm(mat.unsqueeze(1)-mat.unsqueeze(0), p=2, dim=2)
    else:
        if adj.is_sparse:
            # get degrees
            degrees = torch.sparse.sum(adj, dim=1)
            ids = degrees.indices()
            d = torch.zeros(adj.shape[0]).to(device)
            for i in range(len(ids)):
                d[ids[i]] = degrees.values()[i]
            d = torch.sqrt(d)

            edges = adj.indices().cpu().numpy()
            values = torch.zeros(int(edges.shape[1]/2)).to(device)
            for i in range(edges.shape[1]):
                v1 = edges[0,i]
                v2 = edges[1,i]
                values[dic[(v1,v2)]] = ((mat[v1]/d[v1]-mat[v2]/d[v2])**2).sum()
            return values.cpu().numpy()
        else:
            # get degrees
            d = torch.sqrt(adj.sum(1))
            distances = torch.zeros((n_nodes, n_nodes)).to(device)
            for v1, v2 in adj.nonzero():
                distances[v1,v2] = ((mat[v1]/d[v1]-mat[v2]/d[v2])**2).sum()
            return distances.cpu().numpy()


def cosine(mat, mat_cal=True):
    if mat_cal:
        norm_mat = torch.nn.functional.normalize(mat, dim=1)
        return norm_mat@norm_mat.T

def compute_ccns(A, Y):
    c = Y.sum(0).unsqueeze(0)
    B = A @ Y
    B = torch.nn.functional.normalize(B, dim=1)
    H = B @ B.T
    ccns = Y.T @ H @ Y
    ccns = ccns.div(c.T @ c)
    return ccns


dataset_raw = pyg_load_dataset(ds_name, '../data/')
g = dataset_raw[0]
feats = g.x.to(device)
n_nodes = feats.shape[0]
labels = g.y.to(device)
dim_feats = feats.shape[1]
n_classes = dataset_raw.num_classes
edges = g.edge_index
adj = torch.sparse.FloatTensor(edges, torch.ones(edges.shape[1]), [n_nodes, n_nodes]).coalesce().to(device)
n_edges = g.num_edges
adj = adj.to_dense()   # not very large
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj.fill_diagonal_(0)
adj = adj.to_sparse()
if verbose:
    print("""----Data statistics------'
                #Nodes %d
                #Edges %d
                #Classes %d""" %
          (n_nodes, n_edges, n_classes))

Y = torch.zeros(n_nodes, n_classes).to(device)
for i in range(n_nodes):
    Y[i,labels[i]]=1
c = Y.sum(0).unsqueeze(0)
A = adj.to_dense()
ccns = Y.T@cosine(A@Y)@Y
ccns = ccns.div(c.T@c)
print(ccns)
print(compute_ccns(adj, Y))

'''
dic = {}
t = 0
edges = adj.indices().cpu().numpy()
for i in range(edges.shape[1]):
    v1 = edges[0, i]
    v2 = edges[1, i]
    if (v1, v2) in dic.keys() or (v2, v1) in dic.keys():
        continue
    else:
        dic[(v1, v2)] = dic[(v2, v1)] = t
        t += 1

vec1 = normalized_euclidean(feats, adj, mat_cal=False)
cos_mat = cosine(feats, adj.to_dense())
vec2 = np.zeros(int(n_edges/2))
for i in range(edges.shape[1]):
    v1 = edges[0, i]
    v2 = edges[1, i]
    vec2[dic[(v1,v2)]]=cos_mat[v1,v2]
print(np.corrcoef(vec1,vec2))

# print(cosine(feats, adj.to_dense()))
'''
