#from munkres import Munkres, print_matrix
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn import metrics
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def cal_info(pseudo_label_list, true_label):
    """
    pseudo_label_list: 二维列表
    true_label: 一维numpy
    """
    p_acc_list = []
    num_pseudo_label = len(pseudo_label_list)
    NMI = np.zeros((num_pseudo_label, num_pseudo_label))
    for pseudo_label in pseudo_label_list:
        p_acc_list.append(eval_acc(true_label, maplabels(true_label, np.array(pseudo_label))))
    for i in range(num_pseudo_label):
        for j in range(num_pseudo_label):
            if i == j:
                continue
            try:
                NMI[i][j] = metrics.normalized_mutual_info_score(np.array(pseudo_label_list[i]),np.array(pseudo_label_list[j]))
            except:
                print(i,j)
                exit()
    mean_NMI = np.mean(NMI, axis = 1).tolist()
    data = [(acc, nmi) for acc, nmi in zip(p_acc_list, mean_NMI)]
    data.sort()
    p_acc_list = [acc for acc, _ in data]
    mean_NMI = [nmi for _, nmi in data]
    print("####")
    print(p_acc_list)
    print(mean_NMI)
    print("####")



def compute_kld2(p_logit, q_logit):
    p = F.softmax(p_logit, dim = 1) # (B, n_class) 
    q = F.softmax(q_logit, dim = 1) # (B, n_class)
    return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = 1)
    
def compute_kld(p_logit, q_logit):
    p = F.softmax(p_logit, dim = -1) # (B, B, n_class) 
    q = F.softmax(q_logit, dim = -1) # (B, B, n_class)
    return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = -1) # (B, B)
    

def data_split(examples, ratio = [0.6, 0.2, 0.2]):
    from sklearn.utils import shuffle
    examples = shuffle(examples)
    train_num = int(len(examples) * ratio[0])
    dev_num = int(len(examples) * ratio[1])
    train_examples = examples[:train_num]
    dev_examples = examples[train_num:train_num + dev_num]
    test_examples = examples[train_num + dev_num:]
    return train_examples, dev_examples, test_examples

def data_split2(examples, ratio = [6/7, 1/7]):
    from sklearn.utils import shuffle
    examples = shuffle(examples)
    dev_num = int(len(examples) * ratio[0])
    dev_examples = examples[:dev_num]
    test_examples = examples[dev_num:]
    return dev_examples, test_examples

def L2Reg(net):
    reg_loss = 0
    for name, params in net.named_parameters():
        if name[-4:] != 'bias':
            reg_loss += torch.sum(torch.pow(params, 2))
    return reg_loss

def flush():#清下gpu
    '''
    :rely on:import torch
    '''
    torch.cuda.empty_cache()
    '''
    或者在命令行中
    ps aux | grep python
    kill -9 [pid]
    或者
    nvidia-smi --gpu-reset -i [gpu_id]
    '''
    
def clean_text(text): # input is word list
    ret = []
    for word in text:
        normalized_word = re.sub(u"([^\u0020-\u007f])", "", word)
        if normalized_word == '' or normalized_word == ' ' or normalized_word == '    ':
            normalized_word = '[UNK]'
        ret.append(normalized_word)
    return ret
    

def get_pseudo_label(pseudo_label_list, idxes):    
    ret = []
    for idx in idxes:
        ret.append(pseudo_label_list[idx])
    ret = torch.tensor(ret).long()
    return ret


def eval_acc(L1, L2):
    sum = np.sum(L1[:]==L2[:])
    return sum/len(L2)

def eval_p_r_f1(ground_truth, label_pred):
    def _bcubed(i):
        C = label_pred[i]
        n = 0
        for j in range(len(label_pred)):
            if label_pred[j] != C:
                continue
            if ground_truth[i] == ground_truth[j]:
                n += 1
        p = n / num_cluster[C]
        r = n / num_class[ground_truth[i]]
        return p, r

    ground_truth -= 1
    label_pred -= 1
    num_class = [0] * 16
    num_cluster = [0] * 16
    for c in ground_truth:
        num_class[c] += 1
    for c in label_pred:
        num_cluster[c] += 1

    precision = recall = fscore = 0.

    for i in range(len(label_pred)):
        p, r = _bcubed(i)
        precision += p
        recall += r
    precision = precision / len(label_pred)
    recall = recall / len(label_pred)
    fscore = 2 * precision * recall / (precision + recall) # f1 score

    return precision, recall, fscore

def tsne_and_save_data(data, label, epoch):
    '''
    Args:
        data: ndarray (num_examples, dims)
        label: ndarray (num_examples)
    '''
    from sklearn.manifold import TSNE
    import numpy as np
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data) # (num_examples, n_components)
    #fig, ax = plt.subplots()
    x, y, c = result[:, 0], result[:, 1], label
    f = "./{}.npz".format(epoch)
    np.savez(f, x=x, y=y, true_label=label)



def tsne_and_save_pic(data, label, new_class, acc):
    '''
    Args:
        data: ndarray (num_examples, dims)
        label: ndarray (num_examples)
    '''
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    def mscatter(x,y,ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax=plt.gca()
        sc = ax.scatter(x,y,**kw)
        if (m is not None) and (len(m)==len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data) # (num_examples, n_components)
    fig, ax = plt.subplots()
    x, y, c = result[:, 0], result[:, 1], label
    '''
    m = {1:'o',2:'s',3:'D',4:'+'}
    cm = list(map(lambda x:m[x],c))#将相应的标签改为对应的marker
    '''
    #scatter = mscatter(x, y, c=c, m=cm, ax=ax,cmap=plt.cm.RdYlBu)
    scatter = mscatter(x, y, c=c, ax=ax,cmap=plt.cm.Paired, s = 9.0)
    if new_class:
        ax.set_title("t-sne for new class, result:{}".format(acc))
        plt.savefig( 't-sne-new-class.svg' ) 
    else:
        ax.set_title("t-sne for known class, result:{}".format(acc))
        plt.savefig( 't-sne-known-class.svg' ) 


def k_means(data, clusters): # range from 0 to clusters - 1
    #return KMeans(n_clusters=clusters,random_state=0,algorithm='full').fit(data).predict(data)
    return KMeans(n_clusters=clusters,algorithm='full').fit(data).predict(data)



def maplabels(L1, L2):
    Label1 = np.unique(L1)
    Label2 = np.unique(L2)
    nClass1 = len(Label1)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2*ind_cla1)

    row_ind, col_ind = linear_sum_assignment(-G.T)
    newL2 = np.zeros(L2.shape, dtype=int)
    for i in range(nClass2):
        for j in range(len(L2)):
            if L2[j] == row_ind[i]:
                newL2[j] = col_ind[i]
    return newL2


def endless_get_next_batch(loaders, iters, batch_size):
    try:
        data = next(iters)
    except StopIteration:
        iters = iter(loaders)
        data = next(iters)
    # In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
    # so we will skip leftover batch of size < batch_size
    if len(data[0]) < batch_size:
        return endless_get_next_batch(loaders, iters, batch_size)
    return data, iters

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def _worker_init_fn_():
    import random
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // (2**32-1)
    random.seed(torch_seed)
    np.random.seed(np_seed)