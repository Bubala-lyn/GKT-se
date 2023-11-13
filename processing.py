import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import build_dense_graph
import torchvision.transforms as transforms

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

transform = transforms.Compose([
    transforms.ToTensor(),
])

class KTDataset(Dataset):
    def __init__(self, features, questions, answers):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    (features, questions, answers) = zip(*batch)
    features = [torch.FloatTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.FloatTensor(ans) for ans in answers]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1.0)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1.0)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1.0)
    return feature_pad, question_pad, answer_pad


def load_dkt_dataset(file_path, kc_id_path, qt_kc_path, batch_size, graph_type, dkt_graph_path=None,
                     train_ratio=0.7, val_ratio=0.2, shuffle=True, model_type='GKT',
                     use_binary=True, res_len=2, use_cuda=True):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    df = pd.read_csv(file_path)
    df_kc = pd.read_excel(kc_id_path)
    df_qt_kc = pd.read_excel(qt_kc_path)
    df_qt_kc = df_qt_kc.fillna(method='ffill')

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['Score'], inplace=True)
    df.dropna(subset=['ProblemID'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('SubjectID').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['ProblemID'], sort=True)  # we can also use problem_id to represent exercises

    # Step 3 - Cross skill id with answer to form a synthetic feature
    # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
    df['correct'] = df['Score']

    qt_ids = df_qt_kc['question_id'].dropna().unique()
    kc_ids = df_kc['kc_id'].unique()
    one_hot_matrix = pd.DataFrame(0, index=qt_ids, columns=kc_ids)
    # grouped_kc_texts = df_qt_kc.groupby('question_id')['kc_text']
    df_kc = df_kc.set_index('kc_text')
    for key, value in df_qt_kc.groupby('question_id'):
        kc_text = value['kc_text']
        for each in kc_text:
            print(each)
            new_id = df_kc.loc[each, 'kc_id']
            one_hot_matrix.at[key, new_id] = 1


    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []
    kc_id = []
    #feature 问答情况答对打错
    #question 答题skill 号
    #answer
    def get_data(series):
        feature_list.append(series['Score'].astype(float).tolist())
        question_list.append(series['skill'].tolist())
        answer_list.append(series['correct'].eq(1).astype('int').tolist())
        seq_len_list.append(series['correct'].shape[0])
    df.groupby('SubjectID').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    feature_dim = int(df['Score'].max() + 1)
    print('feature_dim: ', feature_dim)
    question_dim = int(df['skill'].max() + 1)
    print('question_dim: ', question_dim)
    concept_num = one_hot_matrix.values.shape[1]
    # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
    # assert feature_dim == res_len * question_dim

    kt_dataset = KTDataset(feature_list, question_list, answer_list)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, train_dataset.indices, student_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        if use_cuda and graph_type in ['Dense', 'Transition', 'DKT']:
            graph = graph.cuda()
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader, one_hot_matrix

def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph