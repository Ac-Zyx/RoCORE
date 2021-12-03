import math
import numpy as np

def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    from scipy.sparse import coo_matrix
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes

def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.
    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.
    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.
    sys_labels : ndarray, (n_frames,)
        System labels.
    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)
    Returns
    -------
    precision : float
        B-cubed precision.
    recall : float
        B-cubed recall.
    f1 : float
        B-cubed F1.
    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    ref_labels = np.array(ref_labels)
    sys_labels = np.array(sys_labels)
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1

def usoon_eval(label, pesudo_true_label):

    from sklearn.metrics.cluster import homogeneity_completeness_v_measure
    from sklearn.metrics import classification_report
    from sklearn.metrics.cluster import adjusted_rand_score
    
    ARI = adjusted_rand_score(label, pesudo_true_label)
    
    # res_dic = classification_report(label, pesudo_true_label,labels=name, output_dict=True)
    # return precision, recall, f1
    B3_prec, B3_rec, B3_f1 = bcubed(label, pesudo_true_label)
    # B3_f1 = res_dic["weighted avg"]['f1-score']
    # B3_prec = res_dic["weighted avg"]['precision']
    # B3_rec = res_dic["weighted avg"]['recall']
    
    v_hom, v_comp, v_f1 = homogeneity_completeness_v_measure(label, pesudo_true_label)
    return B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI
class ClusterEvaluation:
    '''
    groundtruthlabels and predicted_clusters should be two list, for example:
    groundtruthlabels = [0, 0, 1, 1], that means the 0th and 1th data is in cluster 0,
    and the 2th and 3th data is in cluster 1
    '''
    def __init__(self, groundtruthlabels, predicted_clusters):
        self.relations = {}
        self.groundtruthsets, self.assessableElemSet = self.createGroundTruthSets(groundtruthlabels)
        self.predictedsets = self.createPredictedSets(predicted_clusters)

    def createGroundTruthSets(self, labels):

        groundtruthsets= {}
        assessableElems = set()

        for i, c in enumerate(labels):
            assessableElems.add(i)
            groundtruthsets.setdefault(c, set()).add(i)

        return groundtruthsets, assessableElems

    def createPredictedSets(self, cs):

        predictedsets = {}
        for i, c in enumerate(cs):
            predictedsets.setdefault(c, set()).add(i)

        return predictedsets

    def b3precision(self, response_a, reference_a):
        # print response_a.intersection(self.assessableElemSet), 'in precision'
        return len(response_a.intersection(reference_a)) / float(len(response_a.intersection(self.assessableElemSet)))

    def b3recall(self, response_a, reference_a):
        return len(response_a.intersection(reference_a)) / float(len(reference_a))

    def b3TotalElementPrecision(self):
        totalPrecision = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalPrecision += self.b3precision(self.predictedsets[c],
                                                   self.findCluster(r, self.groundtruthsets))

        return totalPrecision / float(len(self.assessableElemSet))

    def b3TotalElementRecall(self):
        totalRecall = 0.0
        for c in self.predictedsets:
            for r in self.predictedsets[c]:
                totalRecall += self.b3recall(self.predictedsets[c], self.findCluster(r, self.groundtruthsets))

        return totalRecall / float(len(self.assessableElemSet))

    def findCluster(self, a, setsDictionary):
        for c in setsDictionary:
            if a in setsDictionary[c]:
                return setsDictionary[c]

    def printEvaluation(self):

        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
            F05B3 = 0.0
        else:
            betasquare = math.pow(0.5, 2)
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)

        m = {'F1': F1B3, 'F0.5': F05B3, 'precision': precB3, 'recall': recB3}
        return m

    def getF05(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()
        betasquare = math.pow(0.5, 2)
        if recB3 == 0.0 and precB3 == 0.0:
            F05B3 = 0.0
        else:
            F05B3 = ((1+betasquare) * recB3 * precB3)/((betasquare*precB3)+recB3)
        return F05B3

    def getF1(self):
        recB3 = self.b3TotalElementRecall()
        precB3 = self.b3TotalElementPrecision()

        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
        else:
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
        return F1B3



class ClusterRidded:
    def __init__(self, gtlabels, prelabels, rid_thres=5):
        self.gtlabels = np.array(gtlabels)
        self.prelabels = np.array(prelabels)
        self.cluster_num_dict = {}
        for item in self.prelabels:
            temp = self.cluster_num_dict.setdefault(item, 0)
            self.cluster_num_dict[item] = temp + 1
        self.NA_list = np.ones(self.gtlabels.shape) # 0 for NA, 1 for not NA
        for i,item in enumerate(self.prelabels):
            if self.cluster_num_dict[item]<=rid_thres:
                self.NA_list[i] = 0
        self.gtlabels_ridded = []
        self.prelabels_ridded = []
        for i, item in enumerate(self.NA_list):
            if item==1:
                self.gtlabels_ridded.append(self.gtlabels[i])
                self.prelabels_ridded.append(self.prelabels[i])
        self.gtlabels_ridded = np.array(self.gtlabels_ridded)
        self.prelabels_ridded = np.array(self.prelabels_ridded)
        print('NA clusters ridded, NA num is:',self.gtlabels.shape[0]-self.gtlabels_ridded.shape[0])

    def printEvaluation(self):
        return ClusterEvaluation(self.gtlabels_ridded,self.prelabels_ridded).printEvaluation()