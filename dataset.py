from torch.utils.data import Dataset
from utils import clean_text
import torch
import json
import random

class InputExample(object):
  def __init__(self, unique_id, text, head_span, tail_span, label):
    self.unique_id = unique_id
    self.text = text # list
    self.head_span = head_span
    self.tail_span = tail_span
    self.label = label
    self.pseudo = -1


class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, unique_id, tokens, input_ids, input_mask): # , input_type_ids
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    # self.input_type_ids = input_type_ids
    self.head_span = None
    self.tail_span = None

class RelationInfo(object):
    def __init__(self, relation_id, relation_description):
        self.relation_id = relation_id
        self.text = relation_description # list


def load_relation_info_dict(args, tokenizer):
    """
        Args:
            filepath(str) : relation description file path
        Return:
            relation_info_dict(dict): mapping from relation surface form to class RelationInfo which include relation id and description
    """
    file_path = args.root + args.rel_filename
    with open(file_path, 'r', encoding = 'utf-8') as f:
        relation_info_dict = {}
        line = f.readline()
        while line:
            relation_name, relation_description = line.split("    ")
            relation_description = tokenizer.tokenize(relation_description.strip())
            if len(relation_description) > args.max_len - 2:
                relation_description = relation_description[ : (args.max_len - 2)]
            relation_description = ['[CLS]'] + relation_description + ['[SEP]']
            relation_info_dict[relation_name] = RelationInfo(len(relation_info_dict), relation_description)
            line = f.readline()
    return relation_info_dict

class FewRelDataset(Dataset):
    '''
    def __init__(self, args, examples, tokenizer):
        """
        Args:
            examples: list of InputExample
        """
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.examples = examples
        self.features = self.convert_examples_to_features(examples=self.examples, seq_length=min(2+self.get_max_seq_length(self.examples, self.tokenizer), self.max_len), tokenizer=self.tokenizer)
        self.fill_bert_entity_spans(self.examples, self.features)
    '''
    def __init__(self, args, examples, tokenizer):
        """
        Args:
            examples: list of InputExample
        """
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.examples = examples
        print(len(self.examples))
        actual_max_len = self.get_max_seq_length(self.examples, self.tokenizer)        
        print(len(self.examples))
        self.features = self.convert_examples_to_features(examples=self.examples, seq_length=min(2+actual_max_len, self.max_len), tokenizer=self.tokenizer)
        self.fill_bert_entity_spans(self.examples, self.features)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.features[index].input_ids, dtype = torch.long)
        input_mask = torch.tensor(self.features[index].input_mask, dtype = torch.long)
        label = self.examples[index].label
        head_span = self.features[index].head_span
        tail_span = self.features[index].tail_span
        return input_ids, input_mask, label, head_span, tail_span

    def __len__(self):
        return len(self.examples)

    def collate_fn(data):
        data = list(zip(*data))
        input_ids, input_mask, label, head_span, tail_span = data
        input_ids = torch.stack(input_ids, dim = 0) # (batch_size, L)
        input_mask = torch.stack(input_mask, dim = 0)
        label = torch.LongTensor(label) # (batch_size)
        head_span = torch.LongTensor(head_span) # (batch_size, 2)
        tail_span = torch.LongTensor(tail_span) # (batch_size, 2)
        return input_ids, input_mask, label, head_span, tail_span

    def preprocess(file_path, relation_info_dict):
        """
        Args:
            file_path(str): file path to dataset
            relation_info_dict: mapping from relation surface to RelationInfo
        Return:
            data: list of InputExample
        """
        data = []
        with open(file_path, 'r', encoding = 'utf-8') as f: 
            ori_data = json.load(f)
            unique_id = 0
            for relation_name, ins_list in ori_data.items():
                relation_id = relation_info_dict[relation_name].relation_id              
                for instance in ins_list:
                    text = clean_text(instance['tokens'])
                    head_span = [instance['h'][2][0][0], instance['h'][2][0][-1]]
                    tail_span = [instance['t'][2][0][0], instance['t'][2][0][-1]]
                    data.append(InputExample(unique_id, text, head_span, tail_span, relation_id))
                    unique_id += 1
        return data


    def preprocess_tacred(file_path, relation_info_dict, args):
        data = [[] for i in range(len(relation_info_dict))]
        data1 = json.load(open(file_path + '/train.json'))
        data2 = json.load(open(file_path + '/dev.json'))
        data3 = json.load(open(file_path + '/test.json'))
        ori_data = data1 + data2 + data3
        cut_cnt = 0
        unique_id = 0
        for sample in ori_data:
            if sample['relation'] == 'no_relation': # 过滤掉所有的no relation
                continue
            text = clean_text(sample['token'])
            head_span = [sample['subj_start'], sample['subj_end']]
            tail_span = [sample['obj_start'], sample['obj_end']]
            if len(text) >= 80:#如果句子过长，就只要两个实体及其中间的部分
                cut_cnt += 1
                if head_span[1] < tail_span[0]:# 头实体在前
                    text = text[head_span[0]:tail_span[1]+1]
                    num_remove = head_span[0]

                else:#尾实体在前
                    text = text[tail_span[0]:head_span[1]+1]
                    num_remove = tail_span[0]
                head_span = [head_span[0]-num_remove, head_span[1]-num_remove]
                tail_span = [tail_span[0]-num_remove, tail_span[1]-num_remove]
            relation_id = relation_info_dict[sample['relation']].relation_id
            data[relation_id].append(InputExample(unique_id, text, head_span, tail_span, relation_id))
            unique_id += 1
        print("cut numbert:{}".format(cut_cnt))
        train_data = data[:args.num_class]
        test_data = data[args.num_class:]
        ret_train_data, ret_test_data = [], []
        for data in train_data:
            ret_train_data.extend(data)
        for data in test_data:
            ret_test_data.extend(data)
        return ret_train_data, ret_test_data

    def preprocess_distant_fewrel(file_path, relation_info_dict):
        data = []
        with open(file_path, 'r', encoding = 'utf-8') as f:
            ins_list = json.load(f)
            unique_id = 0
            for instance in ins_list:
                text = clean_text(instance['sentence'])
                head_span = [instance['head']['e1_begin'], instance['head']['e1_end']]
                tail_span = [instance['tail']['e2_begin'], instance['tail']['e2_end']]
                relation_id = relation_info_dict[instance['relation']].relation_id
                data.append(InputExample(unique_id, text, head_span, tail_span, relation_id))
                unique_id += 1
        return data

        
    def data_split(examples1, examples2, ratio = 0.7):
        from sklearn.utils import shuffle
        examples = examples1 + examples2
        examples = shuffle(examples)
        train_num = int(len(examples) * ratio)
        train_examples = examples[:train_num]
        test_examples = examples[train_num:]
        return train_examples, test_examples
    '''
    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            cur_len = len(bert_tokens)
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        return max_seq_len
    '''
    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        remove_cnt = 0
        new_examples_list = []
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            cur_len = len(bert_tokens)
            if cur_len <= self.max_len-2:#超过最大长度的句子直接丢弃掉(tacred中只有1个这样的句子)
                new_examples_list.append(example)
            else:
                remove_cnt += 1
                continue
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        print("removed sentence number:{}".format(remove_cnt))
        self.examples = new_examples_list
        return max_seq_len



    def convert_examples_to_features(self, examples, seq_length, tokenizer):
        features = []
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            # Account for [CLS] and [SEP] with "- 2"
            if len(bert_tokens) > seq_length - 2:
                bert_tokens = bert_tokens[0 : (seq_length - 2)]
            tokens = []
            # input_type_ids = [
            tokens.append("[CLS]")
            # input_type_ids.append(0) # input type id is not needed in relation extraction task
            for token in bert_tokens:
                tokens.append(token)
                #input_type_ids.append(0)
            tokens.append("[SEP]")
            # input_type_ids.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                # input_type_ids.append(0)
            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            # assert len(input_type_ids) == seq_length
            features.append(
                InputFeatures(
                    unique_id = example.unique_id,
                    tokens = tokens, # bert_token
                    input_ids = input_ids,
                    input_mask = input_mask,
                    # input_type_ids = input_type_ids
                    )
                )
        return features

    def fill_bert_entity_spans(self, examples, features):
        for example, feature in zip(examples, features):
            bert_tokens = feature.tokens[1:]
            actual_tokens = example.text
            head_span = example.head_span
            tail_span = example.tail_span
            actual_token_index, bert_token_index, actual_to_bert = 0, 0, []
            while actual_token_index < len(actual_tokens):
                start, end = bert_token_index, bert_token_index
                actual_token = actual_tokens[actual_token_index]
                token_index = 0
                while token_index < len(actual_token):
                    bert_token = bert_tokens[bert_token_index]
                    if bert_token.startswith('##'):
                        bert_token = bert_token[2:]
                    assert(bert_token.lower()==actual_token[token_index:token_index+len(bert_token)].lower())
                    end = bert_token_index
                    token_index += len(bert_token)
                    bert_token_index += 1
                actual_to_bert.append([start, end])
                actual_token_index += 1

            feature.head_span = (1 + actual_to_bert[head_span[0]][0], 1 + actual_to_bert[head_span[1]][1])
            feature.tail_span = (1 + actual_to_bert[tail_span[0]][0], 1 + actual_to_bert[tail_span[1]][1])


class FewRelSiameseDataset(Dataset):
    def __init__(self, args, examples_list, tokenizer, pos_ratio = 0.5, is_new_class = False):
        """
        Args:
            examples: list of InputExample
        """
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.examples_list = examples_list
        self.num_class = args.num_class
        self.new_class = args.new_class
        self.pos_ratio = pos_ratio
        self.is_new_class = is_new_class
        self.features_list = self.convert_examples_to_features(examples_list = self.examples_list, seq_length = min(2 + self.get_max_seq_length(self.examples_list, self.tokenizer), self.max_len), tokenizer = self.tokenizer)
        self.num_samples = tuple([len(examples) for examples in self.examples_list])
        self.fill_bert_entity_spans(self.examples_list, self.features_list)

    def __getitem__(self, index):
        rel_id = 0
        while index >= self.num_samples[rel_id]:
            index -= self.num_samples[rel_id]
            rel_id += 1
        features = self.features_list[rel_id]
        examples = self.examples_list[rel_id]
        input_ids1 = torch.tensor(features[index].input_ids, dtype = torch.long)
        input_mask1 = torch.tensor(features[index].input_mask, dtype = torch.long)
        label1 = examples[index].label
        head_span1 = features[index].head_span
        tail_span1 = features[index].tail_span
        rnd = random.uniform(0,1)
        if rnd > self.pos_ratio: # neg sample
            if self.is_new_class:
                new_rel_id = random.randint(0, self.new_class - 2) 
            else:
                new_rel_id = random.randint(0, self.num_class - 2)
            if new_rel_id >= rel_id:
                new_rel_id += 1
            label = 0
        else:
            new_rel_id = rel_id
            label = 1
        new_sample_id = random.randint(0, self.num_samples[new_rel_id] - 1)
        features = self.features_list[new_rel_id]
        examples = self.examples_list[new_rel_id]
        input_ids2 = torch.tensor(features[new_sample_id].input_ids, dtype = torch.long)
        input_mask2 = torch.tensor(features[new_sample_id].input_mask, dtype = torch.long)
        label2 = examples[new_sample_id].label # 0-63 for known class, 64-79 for unknown class
        head_span2 = features[new_sample_id].head_span
        tail_span2 = features[new_sample_id].tail_span
        if label == 0:
            assert label1 != label2
        else:
            assert label1 == label2

        return input_ids1, input_mask1, label1, head_span1, tail_span1, input_ids2, input_mask2, label2, head_span2, tail_span2, label

    def __len__(self):
        return sum(self.num_samples)

    def collate_fn(data):
        data = list(zip(*data))
        input_ids1, input_mask1, label1, head_span1, tail_span1, input_ids2, input_mask2, label2, head_span2, tail_span2, label = data
        input_ids1 = torch.stack(input_ids1, dim = 0) # (batch_size, L)
        input_mask1 = torch.stack(input_mask1, dim = 0)
        label1 = torch.LongTensor(label1)
        head_span1 = torch.LongTensor(head_span1) # (batch_size, 2)
        tail_span1 = torch.LongTensor(tail_span1) # (batch_size, 2)
        input_ids2 = torch.stack(input_ids2, dim = 0) # (batch_size, L)
        input_mask2 = torch.stack(input_mask2, dim = 0)
        label2 = torch.LongTensor(label2)
        head_span2 = torch.LongTensor(head_span2) # (batch_size, 2)
        tail_span2 = torch.LongTensor(tail_span2) # (batch_size, 2)
        label = torch.LongTensor(label) # (batch_size)
        return input_ids1, input_mask1, label1, head_span1, tail_span1, input_ids2, input_mask2, label2, head_span2, tail_span2, label


    def convert_examples_to_features(self, examples_list, seq_length, tokenizer):
        features_list = []
        for examples in examples_list:
            features = []
            for example in examples:
                bert_tokens = tokenizer.tokenize(' '.join(example.text))
                # Account for [CLS] and [SEP] with "- 2"
                if len(bert_tokens) > seq_length - 2:
                    bert_tokens = bert_tokens[0 : (seq_length - 2)]
                tokens = []
                # input_type_ids = [
                tokens.append("[CLS]")
                # input_type_ids.append(0) # input type id is not needed in relation extraction task
                for token in bert_tokens:
                    tokens.append(token)
                    #input_type_ids.append(0)
                tokens.append("[SEP]")
                # input_type_ids.append(0)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length
                while len(input_ids) < seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    # input_type_ids.append(0)
                assert len(input_ids) == seq_length
                assert len(input_mask) == seq_length
                # assert len(input_type_ids) == seq_length
                features.append(
                    InputFeatures(
                        unique_id = example.unique_id,
                        tokens = tokens, # bert_token
                        input_ids = input_ids,
                        input_mask = input_mask,
                        # input_type_ids = input_type_ids
                        )
                    )               
            features_list.append(features)
        return features_list

    def preprocess(args, file_path, relation_info_dict, is_new_class = False):
        """
        Args:
            file_path(str): file path to dataset
            relation_info_dict: mapping from relation surface to RelationInfo
        Return:
            data: list of InputExample
        """
        if is_new_class:
            data = [[] for i in range(args.new_class)]
        else:
            data = [[] for i in range(args.num_class)]
        with open(file_path, 'r', encoding = 'utf-8') as f:
            ori_data = json.load(f)
            unique_id = 0
            for relation_name, ins_list in ori_data.items():
                relation_id = relation_info_dict[relation_name].relation_id
                for instance in ins_list:
                    text = clean_text(instance['tokens'])
                    head_span = [instance['h'][2][0][0], instance['h'][2][0][-1]]
                    tail_span = [instance['t'][2][0][0], instance['t'][2][0][-1]]
                    if is_new_class:
                        data[relation_id - args.num_class].append(InputExample(unique_id, text, head_span, tail_span, relation_id))
                    else:
                        data[relation_id].append(InputExample(unique_id, text, head_span, tail_span, relation_id))
                    unique_id += 1
        return data           

    def convert_siamese_format(args, examples, is_new_class = False):
        if is_new_class:
            data = [[] for i in range(args.new_class)]
        else:
            data = [[] for i in range(args.num_class)]
        for example in examples:
            if is_new_class:
                data[example.label - args.num_class].append(example)
            else:
                data[example.label].append(example)
        return data


    def get_max_seq_length(self, examples_list, tokenizer):
        max_seq_len = -1
        for examples in examples_list:
            for example in examples:
                bert_tokens = tokenizer.tokenize(' '.join(example.text))
                cur_len = len(bert_tokens)
                if cur_len > max_seq_len:
                    max_seq_len = cur_len
        return max_seq_len                

    def fill_bert_entity_spans(self, examples_list, features_list):
        assert(len(examples_list) == len(features_list))
        for i in range(len(examples_list)):
            examples = examples_list[i]
            features = features_list[i]
            for example, feature in zip(examples, features):
                bert_tokens = feature.tokens[1:]
                actual_tokens = example.text
                head_span = example.head_span
                tail_span = example.tail_span
                actual_token_index, bert_token_index, actual_to_bert = 0, 0, []
                while actual_token_index < len(actual_tokens):
                    start, end = bert_token_index, bert_token_index
                    actual_token = actual_tokens[actual_token_index]
                    token_index = 0
                    while token_index < len(actual_token):
                        bert_token = bert_tokens[bert_token_index]
                        if bert_token.startswith('##'):
                            bert_token = bert_token[2:]
                        assert(bert_token.lower()==actual_token[token_index:token_index+len(bert_token)].lower())
                        end = bert_token_index
                        token_index += len(bert_token)
                        bert_token_index += 1
                    actual_to_bert.append([start, end])
                    actual_token_index += 1

                feature.head_span = (1 + actual_to_bert[head_span[0]][0], 1 + actual_to_bert[head_span[1]][1])
                feature.tail_span = (1 + actual_to_bert[tail_span[0]][0], 1 + actual_to_bert[tail_span[1]][1])            
        assert features_list[-1][-1].head_span is not None

class FewRelDatasetWithPseudoLabel(Dataset):
    def __init__(self, args, examples, tokenizer):
        """
        Args:
            examples: list of InputExample
        """
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.examples = examples
        print(len(self.examples))
        actual_max_len = self.get_max_seq_length(self.examples, self.tokenizer)        
        print(len(self.examples))
        self.features = self.convert_examples_to_features(examples=self.examples, seq_length=min(2+actual_max_len, self.max_len), tokenizer=self.tokenizer)
        #self.features = self.convert_examples_to_features(examples=self.examples, seq_length=min(2+self.get_max_seq_length(self.examples, self.tokenizer), self.max_len), tokenizer=self.tokenizer)
        self.fill_bert_entity_spans(self.examples, self.features)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.features[index].input_ids, dtype = torch.long)
        input_mask = torch.tensor(self.features[index].input_mask, dtype = torch.long)
        label = self.examples[index].label
        head_span = self.features[index].head_span
        tail_span = self.features[index].tail_span
        pseudo = self.examples[index].pseudo
        return input_ids, input_mask, label, head_span, tail_span, index, pseudo

    def __len__(self):
        return len(self.examples)

    def collate_fn(data):
        data = list(zip(*data))
        input_ids, input_mask, label, head_span, tail_span, index, pseudo = data
        input_ids = torch.stack(input_ids, dim = 0) # (batch_size, L)
        input_mask = torch.stack(input_mask, dim = 0)
        label = torch.LongTensor(label) # (batch_size)
        head_span = torch.LongTensor(head_span) # (batch_size, 2)
        tail_span = torch.LongTensor(tail_span) # (batch_size, 2)
        index = torch.LongTensor(index)
        pseudo = torch.LongTensor(pseudo)
        return input_ids, input_mask, label, head_span, tail_span, index, pseudo

    def preprocess(file_path, relation_info_dict):
        """
        Args:
            file_path(str): file path to dataset
            relation_info_dict: mapping from relation surface to RelationInfo
        Return:
            data: list of InputExample
        """
        data = []
        with open(file_path, 'r', encoding = 'utf-8') as f: 
            ori_data = json.load(f)
            unique_id = 0
            for relation_name, ins_list in ori_data.items():
                relation_id = relation_info_dict[relation_name].relation_id              
                for instance in ins_list:
                    text = clean_text(instance['tokens'])
                    head_span = [instance['h'][2][0][0], instance['h'][2][0][-1]]
                    tail_span = [instance['t'][2][0][0], instance['t'][2][0][-1]]
                    data.append(InputExample(unique_id, text, head_span, tail_span, relation_id))
                    unique_id += 1
        return data

    def data_split(examples1, examples2, ratio = 0.7):
        from sklearn.utils import shuffle
        examples = examples1 + examples2
        examples = shuffle(examples)
        train_num = int(len(examples) * ratio)
        train_examples = examples[:train_num]
        test_examples = examples[train_num:]
        return train_examples, test_examples
    '''
    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            cur_len = len(bert_tokens)
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        return max_seq_len
    '''
    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        remove_cnt = 0
        new_examples_list = []
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            cur_len = len(bert_tokens)
            if cur_len <= self.max_len-2:#超过最大长度的句子直接丢弃掉(tacred中只有1个这样的句子)
                new_examples_list.append(example)
            else:
                remove_cnt += 1
                continue
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        print("removed sentence number:{}".format(remove_cnt))
        self.examples = new_examples_list
        return max_seq_len

    def convert_examples_to_features(self, examples, seq_length, tokenizer):
        features = []
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            # Account for [CLS] and [SEP] with "- 2"
            if len(bert_tokens) > seq_length - 2:
                bert_tokens = bert_tokens[0 : (seq_length - 2)]
            tokens = []
            # input_type_ids = [
            tokens.append("[CLS]")
            # input_type_ids.append(0) # input type id is not needed in relation extraction task
            for token in bert_tokens:
                tokens.append(token)
                #input_type_ids.append(0)
            tokens.append("[SEP]")
            # input_type_ids.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                # input_type_ids.append(0)
            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            # assert len(input_type_ids) == seq_length
            features.append(
                InputFeatures(
                    unique_id = example.unique_id,
                    tokens = tokens, # bert_token
                    input_ids = input_ids,
                    input_mask = input_mask,
                    # input_type_ids = input_type_ids
                    )
                )
        return features

    def fill_bert_entity_spans(self, examples, features):
        for example, feature in zip(examples, features):
            bert_tokens = feature.tokens[1:]
            actual_tokens = example.text
            head_span = example.head_span
            tail_span = example.tail_span
            actual_token_index, bert_token_index, actual_to_bert = 0, 0, []
            while actual_token_index < len(actual_tokens):
                start, end = bert_token_index, bert_token_index
                actual_token = actual_tokens[actual_token_index]
                token_index = 0
                while token_index < len(actual_token):
                    bert_token = bert_tokens[bert_token_index]
                    if bert_token.startswith('##'):
                        bert_token = bert_token[2:]
                    assert(bert_token.lower()==actual_token[token_index:token_index+len(bert_token)].lower())
                    end = bert_token_index
                    token_index += len(bert_token)
                    bert_token_index += 1
                actual_to_bert.append([start, end])
                actual_token_index += 1

            feature.head_span = (1 + actual_to_bert[head_span[0]][0], 1 + actual_to_bert[head_span[1]][1])
            feature.tail_span = (1 + actual_to_bert[tail_span[0]][0], 1 + actual_to_bert[tail_span[1]][1])
