from dataset import FewRelDataset as BertDataset
from dataset import FewRelDatasetWithPseudoLabel as PBertDataset
from dataset import load_relation_info_dict
from torch.utils.data import DataLoader
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import *
from utils import seed_everything, data_split2, L2Reg, compute_kld, sigmoid_rampup, _worker_init_fn_
import random
import os
from evaluation import ClusterEvaluation, usoon_eval
# 先训练AUTOENCODER 10轮，再一起训
def load_pretrained(args):
    from transformers import BertTokenizer, BertModel, BertConfig
    config = BertConfig.from_pretrained(args.bert_model, output_hidden_states = True, output_attentions = True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, output_hidden_states=True)
    bert = BertModel.from_pretrained(args.bert_model, config = config)
    return config, tokenizer, bert

def update_centers_l(net, args, known_class_dataloader):
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    centers = torch.zeros(args.num_class, args.kmeans_dim, device = device)
    num_samples = [0] * args.num_class
    with torch.no_grad():
        for iteration, (input_ids, input_mask, label, head_span, tail_span) in enumerate(known_class_dataloader): # (batch_size, seq_len), (batch_size)
            data = (input_ids, input_mask, label, head_span, tail_span)
            sia_rep = net.forward(data, msg = 'similarity') # (batch_size, hidden_dim)
            for i in range(len(sia_rep)):
                vec = sia_rep[i]
                l = label[i]
                centers[l] += vec
                num_samples[l] += 1
        for c in range(args.num_class):
            centers[c] /= num_samples[c]
        net.module.ct_loss_l.centers = centers.to(device)

def update_centers_u(net, args, new_class_dataloader):
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=args.new_class,random_state=0,algorithm='full')
    true = [-1] * len(new_class_dataloader.dataset)
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        rep = []
        idxes = []
        true_label = []
        for iteration, (input_ids, input_mask, label, head_span, tail_span, idx, _) in enumerate(new_class_dataloader): # (batch_size, seq_len), (batch_size)
            data = (input_ids, input_mask, label, head_span, tail_span)
            sia_rep = net.forward(data, msg = 'similarity') # (batch_size, kmeans_dim)
            true_label.append(label)
            idxes.append(idx)
            rep.append(sia_rep)
        rep = torch.cat(rep, dim = 0).cpu().numpy() # (num_test_ins, kmeans_dim)
        idxes = torch.cat(idxes, dim = 0).cpu().numpy()
        true_label = torch.cat(true_label, dim = 0).cpu().numpy()

    label_pred = clf.fit_predict(rep)# from 0 to args.new_class - 1
    net.module.ct_loss_u.centers = torch.from_numpy(clf.cluster_centers_).to(device) # (num_class, kmeans_dim)
    for i in range(len(idxes)):
        idx = idxes[i]
        pseudo = label_pred[i]
        true[idx] = true_label[i]
        new_class_dataloader.dataset.examples[idx].pseudo = pseudo
        #pseudo_label_list[cnt][idx] = pseudo

def train_one_epoch(net, args, epoch, known_class_dataloader, new_class_dataloader, optimizer):
    net.train()
    device = torch.device("cuda" if args.cuda else "cpu")
    known_class_iter = iter(known_class_dataloader)
    new_class_iter = iter(new_class_dataloader)
    siamese_known_class_iter = iter(known_class_dataloader)
    siamese_new_class_iter = iter(new_class_dataloader)
    if args.IL:
        icr_new_class_iter = iter(new_class_dataloader)
        icr_known_class_iter = iter(known_class_dataloader)

    epoch_ce_loss = 0
    epoch_ct_loss = 0
    epoch_rec_loss = 0
    epoch_u_loss = 0
    epoch_acc = 0 

    with tqdm(total=len(known_class_dataloader), desc='training') as pbar:
        for iteration in range(len(known_class_dataloader)):
            optimizer.zero_grad()


            # Training unlabeled head
            if epoch > args.num_pretrain:
                # Training unlabeled head
                data, new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = new_class_iter, batch_size = args.b_size)
                data, pseudo_label = data[:-2], data[-1]
                pretrained_feat = net.forward(data, msg = 'feat').detach() # (batch_size, new_class)
                batch_size = pseudo_label.size(0)
                ground_truth = data[2] - args.num_class
                pseudo_label = (pseudo_label.unsqueeze(0) == pseudo_label.unsqueeze(1)).float().to(device)
                logits = net.forward(data, msg = 'unlabeled', cut_gradient = False) # (batch_size, new_class)
                expanded_logits = logits.expand(batch_size, -1, -1)
                expanded_logits2 = expanded_logits.transpose(0, 1)
                kl1 = compute_kld(expanded_logits.detach(), expanded_logits2)
                kl2 = compute_kld(expanded_logits2.detach(), expanded_logits) # (batch_size, batch_size)
                assert kl1.requires_grad
                u_loss = torch.mean(pseudo_label * (kl1 + kl2) + (1 - pseudo_label) * (torch.relu(args.sigmoid - kl1) + torch.relu(args.sigmoid - kl2)))
                u_loss.backward()
                #flush()
            else:
                u_loss = torch.tensor(0)


            # Training siamese head (exclude bert layer)
            data, siamese_known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = siamese_known_class_iter, batch_size = args.b_size)
            sia_rep1, rec_loss1 = net.forward(data, msg = 'reconstruct') # (batch_size, kmeans_dim)
            label = data[2]
            data, siamese_new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = siamese_new_class_iter, batch_size = args.b_size)
            data, pseudo = data[:-2], data[-1]
            sia_rep2, rec_loss2 = net.forward(data, msg = 'reconstruct') # (batch_size, kmeans_dim)
            rec_loss = (rec_loss1.mean() + rec_loss2.mean()) / 2
            pseudo = pseudo.to(device)
            label = label.to(device)
            ct_loss = args.ct * net.module.ct_loss_l(label, sia_rep1)
            loss = rec_loss + ct_loss + 1e-5 * (L2Reg(net.module.similarity_encoder) + L2Reg(net.module.similarity_decoder))
            loss.backward()
            #flush() 

            if not args.IL and epoch > args.num_pretrain:
                # Training labeled head and specific layer of bert
                data, known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = known_class_iter, batch_size = args.b_size)
                known_logits = net.forward(data, msg = 'labeled')
                if args.IL:
                    known_logits = known_logits[:,:args.num_class]
                label_pred = torch.max(known_logits, dim = -1)[1]
                known_label = data[2].to(device)
                acc = 1.0 * torch.sum(label_pred == known_label) / len(label_pred)
                ce_loss = net.module.ce_loss(input = known_logits, target = known_label)
                ce_loss.backward()
                #flush()
            else:
                ce_loss = torch.tensor(0)
                acc = torch.tensor(0)


            if args.IL:
                if epoch > args.num_pretrain:
                    data, icr_new_class_iter = endless_get_next_batch(loaders = new_class_dataloader, iters = icr_new_class_iter, batch_size = args.b_size)
                    data = data[:-2]
                    net.eval()
                    with torch.no_grad():
                        logits = net.forward(data, msg = 'unlabeled')
                        u_label = torch.max(logits, dim = -1)[1] + args.num_class
                    net.train()
                    u_logits = net.forward(data, msg = 'labeled', cut_gradient = False)

                    data, icr_known_class_iter = endless_get_next_batch(loaders = known_class_dataloader, iters = icr_known_class_iter, batch_size = args.b_size)
                    l_logits = net.forward(data, msg = 'labeled', cut_gradient = False)
                    l_label = data[2].to(device)
                    #icr_ce_loss = args.rampup_cof * sigmoid_rampup(epoch-args.num_pretrain, args.rampup_length) * net.module.ce_loss(input = logits, target = u_label)
                    icr_ce_loss = args.rampup_cof * sigmoid_rampup(epoch-args.num_pretrain, args.rampup_length) * net.module.ce_loss(input=u_logits, target=u_label) \
                                    + net.module.ce_loss(input=l_logits, target=l_label)
                    icr_ce_loss.backward()
                    #flush()
                else:
                    icr_ce_loss = torch.tensor(0)  

            optimizer.step()  
            epoch_ce_loss += ce_loss.item()
            epoch_ct_loss += ct_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_u_loss += u_loss.item()
            epoch_acc += acc.item()
            pbar.update(1)
            pbar.set_postfix({"acc":epoch_acc / (iteration + 1), "ce loss":epoch_ce_loss / (iteration + 1), "rec loss":epoch_rec_loss / (iteration + 1), "ct_loss":epoch_ct_loss / (iteration + 1), "u_loss":epoch_u_loss / (iteration + 1), 'lr': optimizer.state_dict()['param_groups'][0]['lr']})
    num_iteration = len(known_class_dataloader)
    print("===> Epoch {} Complete: Avg. ce Loss: {:.4f}, rec Loss: {:.4f}, ct Loss: {:.4f}, u Loss: {:.4f}, known class acc: {:.4f}".format(epoch, epoch_ce_loss / num_iteration, epoch_rec_loss / num_iteration, epoch_ct_loss / num_iteration, epoch_u_loss / num_iteration, epoch_acc / num_iteration))
    

def test_one_epoch(net, args, epoch, new_class_dataloader):
    import random
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        ground_truth = []
        label_pred = []
        with tqdm(total=len(new_class_dataloader), desc='testing') as pbar:
            for iteration, data in enumerate(new_class_dataloader):
                data = data[:-2]
                logits = net.forward(data, msg = 'unlabeled')
                ground_truth.append(data[2])
                label_pred.append(logits.max(dim = -1)[1].cpu())
                pbar.update(1)
            label_pred = torch.cat(label_pred, dim = 0).numpy()
            ground_truth = torch.cat(ground_truth, dim = 0).numpy() - args.num_class
            cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
            B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI = usoon_eval(ground_truth, label_pred)
    print("B3_f1:{}, B3_prec:{}, B3_rec:{}, v_f1:{}, v_hom:{}, v_comp:{}, ARI:{}".format(B3_f1, B3_prec, B3_rec, v_f1, v_hom, v_comp, ARI))
    print(cluster_eval)
    return cluster_eval['F1']

def test_mix(known_pred, known_truth, new_pred, new_truth):
    pred = np.concatenate([known_pred, new_pred])
    truth = np.concatenate([known_truth, new_truth])
    cluster_eval = ClusterEvaluation(truth,pred).printEvaluation()
    print("mix result:",cluster_eval)
    return cluster_eval['F1']

def test_one_epoch2(net, args, epoch, new_class_dataloader, labelled = False):
    import random
    net.eval()
    device = torch.device("cuda" if args.cuda else "cpu")
    desc = 'labelled' if labelled else 'unlabelled'
    with torch.no_grad():
        ground_truth = []
        label_pred = []
        with tqdm(total=len(new_class_dataloader), desc=desc) as pbar:
            for iteration, data in enumerate(new_class_dataloader):
                if not labelled:
                    data = data[:-2]
                logits = net.forward(data, msg = 'labeled')
                ground_truth.append(data[2])
                label_pred.append(logits.max(dim = -1)[1].cpu())
                pbar.update(1)
            label_pred = torch.cat(label_pred, dim = 0).numpy()
            ground_truth = torch.cat(ground_truth, dim = 0).numpy()
            cluster_eval = ClusterEvaluation(ground_truth,label_pred).printEvaluation()
    print(cluster_eval)
    return cluster_eval['F1'], label_pred, ground_truth

def main(args):
    print(52 * "-" + "\nBert cluster task for zero shot relation extraction:\n" + 52 * '-')
    config, tokenizer, pretrained_model = load_pretrained(args)
    relation_info_dict = load_relation_info_dict(args, tokenizer) # mapping from relation name to rel id and description words

    if args.dataset == 'fewrel':
        known_class_examples = BertDataset.preprocess(args.root + args.known_class_filename, relation_info_dict)
        new_class_examples = PBertDataset.preprocess(args.root + args.new_class_filename, relation_info_dict)
        known_class_train_examples, known_class_test_examples = data_split2(known_class_examples)
        new_class_train_examples, new_class_test_examples = data_split2(new_class_examples)
    elif args.dataset == 'fewrel_distant':
        known_class_examples = BertDataset.preprocess_distant_fewrel(args.root + args.known_class_filename, relation_info_dict)
        new_class_train_examples = BertDataset.preprocess_distant_fewrel(args.root + args.new_class_filename, relation_info_dict)
        _, new_class_test_examples = data_split2(BertDataset.preprocess(args.root + args.test_class_filename, relation_info_dict))
        known_class_train_examples, known_class_test_examples = data_split2(known_class_examples)

    elif args.dataset == 'tacred':
        known_class_examples, new_class_examples = BertDataset.preprocess_tacred(args.root, relation_info_dict, args)
        known_class_train_examples, known_class_test_examples = data_split2(known_class_examples)
        new_class_train_examples, new_class_test_examples = data_split2(new_class_examples)        
    known_class_train_examples = known_class_train_examples[:int(len(known_class_train_examples)*args.p)]
    known_class_train_dataloader = DataLoader(BertDataset(args, known_class_train_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = BertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    known_class_test_dataloader = DataLoader(BertDataset(args, known_class_test_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = BertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    print("knwon class dataloader ready...")
    new_class_train_dataloader = DataLoader(PBertDataset(args, new_class_train_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = PBertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    new_class_test_dataloader = DataLoader(PBertDataset(args, new_class_test_examples, tokenizer), batch_size = args.b_size, shuffle = True, num_workers = 0, collate_fn = PBertDataset.collate_fn, worker_init_fn=_worker_init_fn_())
    print("new class dataloader ready...")
    net = ZeroShotModel3(args, config, pretrained_model, unfreeze_layers = [args.layer])
    if args.cuda:
        net.cuda()
        net = nn.DataParallel(net)
    print("net ready...")
    print("-"*32)
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    best_result = 0
    best_test_result = 0
    wait_times = 0
    for epoch in range(1, args.epochs + 1):
        print("\n-------EPOCH {}-------".format(epoch))
        if epoch > args.num_pretrain:
            wait_times += 1
        update_centers_u(net, args, new_class_train_dataloader)
        update_centers_l(net, args, known_class_train_dataloader)
        train_one_epoch(net, args, epoch, known_class_train_dataloader, new_class_train_dataloader, optimizer)
        if args.IL:
            _, known_pred, known_truth = test_one_epoch2(net, args, epoch, known_class_test_dataloader, labelled = True)
            test_one_epoch2(net, args, epoch, new_class_train_dataloader, labelled = False)
            _, new_pred, new_truth = test_one_epoch2(net, args, epoch, new_class_test_dataloader, labelled = False)
            result = test_mix(known_pred, known_truth, new_pred, new_truth)
            test_result = result
            test_one_epoch(net, args, epoch, new_class_train_dataloader)
            test_one_epoch(net, args, epoch, new_class_test_dataloader)
        else:
            test_one_epoch2(net, args, epoch, known_class_test_dataloader, labelled = True)
            result = test_one_epoch(net, args, epoch, new_class_train_dataloader)
            test_result = test_one_epoch(net, args, epoch, new_class_test_dataloader)
        if result > best_result:
            wait_times = 0
            best_result = result
            best_test_result = test_result
            print("new class dev best result: {}, test result: {}".format(best_result, test_result))
            
        if wait_times > args.wait_times:
            print("wait times arrive: {}, stop training, best result is: {}".format(args.wait_times, best_test_result))
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Bert probe task for relation extraction')

    parser.add_argument("--load", type = str)
    parser.add_argument("--save", type = str, default = '../model/')
    parser.add_argument("--dataset", type = str, choices = ['fewrel', 'fewrel_distant', 'tacred'])
    parser.add_argument("--rel_filename", type = str, default = "relation_description.txt")
    parser.add_argument("--known_class_filename", type = str, default = "labeled.json")
    parser.add_argument("--new_class_filename", type = str, default = "unlabeled.json")
    parser.add_argument("--test_class_filename", type = str, default = "unlabeled.json")
    parser.add_argument("--root", type = str, default = "../data/fewrel/")
    parser.add_argument("--mode", type = int, default = 3)

    parser.add_argument("--p", type = float, default=1.0)
    parser.add_argument("--layer", type = int, default = 8)
    parser.add_argument("--b_size", type = int, default = 100)
    parser.add_argument("--clip_grad", type = float, default = 0)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--max_len", type = int, default = 160)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--wait_times", type = int, default = 10)

    parser.add_argument("--ct", type = float, default = 0.005)
    parser.add_argument("--num_pretrain", type = float, default = 10)
    parser.add_argument("--sigmoid", type = float, default = 2)
    parser.add_argument("--hidden_dim", type = int, default = 512)
    parser.add_argument("--kmeans_dim", type = int, default = 256)
    parser.add_argument("--num_class", type = int, default = 64)
    parser.add_argument("--new_class", type = int, default = 16)
    parser.add_argument("--rampup_cof", type = float, default = 0.05)
    parser.add_argument("--rampup_length", type = float, default = 50)
    parser.add_argument("--IL", action = 'store_true', help = 'incremental setting')
    parser.add_argument("--cuda", action = 'store_true', help = 'use CUDA')
    parser.add_argument("--seed", type = int, default = 1234)
    parser.add_argument("--bert_model", 
                          default="bert-base-uncased", 
                          type=str,
                          help="bert pre-trained model selected in the list: bert-base-uncased, "
                          "bert-large-uncased, bert-base-cased, bert-large-cased")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    seed_everything(args.seed)
    if args.dataset == 'tacred':
        args.ct=0.001
        args.num_class = 31
        args.new_class = 10
        args.root = '../data/tacred/'

    elif args.dataset == 'fewrel_distant':
        args.root = '../data/fewrel_distant/'
        args.known_class_filename = 'fewrel_distant_train.json'
        args.new_class_filename = 'fewrel_distant_val.json'

    main(args)