from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import endless_get_next_batch, k_means, maplabels, eval_acc, flush

def finetune(model, unfreeze_layers):
    params_name_mapping = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'layer.12']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if params_name_mapping[ele] in name:
                param.requires_grad = True
                break
    return model

class ZeroShotModel3(nn.Module): # Pooling as relation description representation
    def __init__(self, args, config, pretrained_model, unfreeze_layers = []):
        super().__init__()
        self.IL = args.IL
        self.max_len = args.max_len
        self.num_class = args.num_class
        self.new_class = args.new_class
        self.hidden_dim = args.hidden_dim
        self.kmeans_dim = args.kmeans_dim
        self.initial_dim = config.hidden_size
        self.unfreeze_layers = unfreeze_layers
        self.pretrained_model = finetune(pretrained_model, self.unfreeze_layers) # fix bert weights
        self.layer = args.layer
        #self.similarity_classifier = Classifier(num_layers = 1, input_size = 4 * self.initial_dim, hidden_size = args.hidden_dim, output_size = 1)
        self.similarity_encoder = nn.Sequential(
                nn.Linear(2 * self.initial_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.kmeans_dim)
                #.LeakyReLU()
        )
        self.similarity_decoder = nn.Sequential(
                nn.Linear(self.kmeans_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, 2 * self.initial_dim)
        )
        self.device = torch.device("cuda" if args.cuda else "cpu")
        #self.ct_loss_u = CenterLoss(dim_hidden = self.kmeans_dim, num_classes = self.new_class)
        self.ct_loss_u = CenterLoss2(dim_hidden = self.kmeans_dim, num_classes = self.new_class)
        self.ct_loss_l = CenterLoss(dim_hidden = self.kmeans_dim, num_classes = self.num_class)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.bce_loss2 = torch.nn.BCELoss()
        #self.ce_loss2 = nn.CrossEntropyLoss(weight = torch.cat([torch.ones(args.num_class), torch.zeros(args.new_class)], dim = 0))
        self.ce_loss = nn.CrossEntropyLoss()
        if self.IL:
            self.labeled_head = nn.Linear(2 * self.initial_dim, self.num_class + self.new_class)
        else:
            self.labeled_head = nn.Linear(2 * self.initial_dim, self.num_class)
        self.unlabeled_head = nn.Linear(2 * self.initial_dim, self.new_class)
        #self.icr_head = nn.Linear(2 * self.initial_dim, self.num_class + self.new_class)
        self.bert_params = []
        for name, param in self.pretrained_model.named_parameters():
            if param.requires_grad is True:
                self.bert_params.append(param)

    def get_pretrained_feature(self, input_id, input_mask, head_span, tail_span):
        _, _, all_encoder_layers, _ = self.pretrained_model(input_id, token_type_ids=None, attention_mask=input_mask) # (13 * [batch_size, seq_len, bert_embedding_len])
        encoder_layers = all_encoder_layers[self.layer] # (batch_size, seq_len, bert_embedding)
        batch_size = encoder_layers.size(0)
        head_entity_rep = torch.stack([torch.max(encoder_layers[i, head_span[i][0]:head_span[i][1]+1, :], dim = 0)[0] for i in range(batch_size)], dim = 0)
        tail_entity_rep = torch.stack([torch.max(encoder_layers[i, tail_span[i][0]:tail_span[i][1]+1, :], dim = 0)[0] for i in range(batch_size)], dim = 0) # (batch_size, bert_embedding)
        pretrained_feat = torch.cat([head_entity_rep, tail_entity_rep], dim = 1) # (batch_size, 2 * bert_embedding)
        return pretrained_feat

    def forward(self, data, msg = 'vat', cut_gradient = False):
        if msg == 'similarity':
            with torch.no_grad():
                input_ids, input_mask, label, head_span, tail_span = data
                input_ids, input_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, label)))
                pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            sia_rep = self.similarity_encoder(pretrained_feat) # (batch_size, keamns_dim)
            return sia_rep # (batch_size, keamns_dim)

        elif msg == 'reconstruct':
            with torch.no_grad():
                input_ids, input_mask, label, head_span, tail_span = data
                input_ids, input_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, label)))
                pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            sia_rep = self.similarity_encoder(pretrained_feat) # (batch_size, kmeans_dim)
            rec_rep = self.similarity_decoder(sia_rep) # (batch_size, 2 * bert_embedding)
            rec_loss = (rec_rep - pretrained_feat).pow(2).mean(-1)
            return sia_rep, rec_loss


        elif msg == 'labeled':
            input_ids, input_mask, label, head_span, tail_span = data
            input_ids, input_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, label)))
            pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            if cut_gradient:
                pretrained_feat = pretrained_feat.detach()
            logits = self.labeled_head(pretrained_feat) 
            return logits # (batch_size, num_class + new_class)

        elif msg == 'unlabeled':
            input_ids, input_mask, label, head_span, tail_span = data
            input_ids, input_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, label)))
            pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            if cut_gradient:
                pretrained_feat = pretrained_feat.detach()
            logits = self.unlabeled_head(pretrained_feat)
            return logits # (batch_size, new_class)

        elif msg == 'feat':
            input_ids, input_mask, label, head_span, tail_span = data
            input_ids, input_mask, label = list(map(lambda x:x.to(self.device),(input_ids, input_mask, label)))
            pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span, tail_span) # (batch_size, 2 * bert_embedding)
            return pretrained_feat

        elif msg == 'vat':
            pretrained_feat = data
            logits = self.unlabeled_head(pretrained_feat)
            return logits # (batch_size, new_class)

        else:
            raise NotImplementedError('not implemented!')

class Classifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout=0,
                 batch_norm=False):
        super(Classifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            #self.net.add_module('p-relu-{}'.format(i), nn.ReLU())
            self.net.add_module('p-relu-{}'.format(i), nn.LeakyReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))

    def forward(self, input):
        return self.net(input)

class CenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c = 1.0, use_cuda = True):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.delta = 1
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.centers = None
        self.alpha = 0.1

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0) # (num_class, batch_size, hid_dim) => (batch_size, num_class, hid_dim)
        expanded_centers = self.centers.expand(batch_size, -1, -1) # (batch_size, num_class, hid_dim)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1) # (batch_size, num_class, hid_dim) => (batch_size, num_class)
        intra_distances = distance_centers.gather(1, y.unsqueeze(1)).squeeze() # (batch_size, num_class) => (batch_size, 1) => (batch_size)
        loss = 0.5 * self.lambda_c * torch.mean(intra_distances) # (batch_size) => scalar
        return loss


class CenterLoss2(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c = 1.0, use_cuda = True):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.delta = 1
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.centers = None
        self.alpha = 1.

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0) # (num_class, batch_size, hid_dim) => (batch_size, num_class, hid_dim)
        expanded_centers = self.centers.expand(batch_size, -1, -1) # (batch_size, num_class, hid_dim)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1) # (batch_size, num_class, hid_dim) => (batch_size, num_class)
        intra_distances = distance_centers.gather(1, y.unsqueeze(1)).squeeze() # (batch_size, num_class) => (batch_size, 1) => (batch_size)
        q = 1.0/(1.0+distance_centers/self.alpha) # (batch_size, num_class)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        prob = q.gather(1, y.unsqueeze(1)).squeeze() # (batch_size)
        loss = 0.5 * self.lambda_c * torch.mean(intra_distances*prob) # (batch_size) => scalar
        return loss