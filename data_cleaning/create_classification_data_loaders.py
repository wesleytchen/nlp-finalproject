import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import moh_x_cleaning
import trofi_cleaning
import vua_cleaning
import util
from util import TextDatasetWithGloveElmoSuffix as TextDataset
import random
import h5py


def create_vua_embeddings():
    train_tups, val_tups, test_tups = vua_cleaning.get_vua_tuples()
    vocab = util.get_vocab(train_tups + val_tups + test_tups)
    # two dictionaries. <PAD>: 0, <UNK>: 1
    word2idx, idx2word = util.get_word2idx_idx2word(vocab)
    # glove_embeddings a nn.Embeddings
    glove_embeddings = util.get_embedding_matrix(word2idx, idx2word, normalization=False)
    # elmo_embeddings
    elmos_train_vua = h5py.File('../elmo/VUA_train.hdf5', 'r')
    elmos_val_vua = h5py.File('../elmo/VUA_val.hdf5', 'r')
    elmos_test_vua = h5py.File('../elmo/VUA_test.hdf5', 'r')

    # suffix_embeddings: number of suffix tag is 2, and the suffix embedding dimension is 50
    suffix_embeddings = nn.Embedding(2, 50)

    embedded_train_vua = [[util.embed_sequence(example[0], example[1], word2idx,
                                          glove_embeddings, elmos_train_vua, suffix_embeddings), example[2]]
                          for example in train_tups]
    embedded_val_vua = [[util.embed_sequence(example[0], example[1], word2idx,
                                        glove_embeddings, elmos_val_vua, suffix_embeddings), example[2]]
                        for example in val_tups]
    embedded_test_vua = [[util.embed_sequence(example[0], example[1], word2idx,
                                         glove_embeddings, elmos_test_vua, suffix_embeddings), example[2]]
                         for example in test_tups]
    return embedded_train_vua, embedded_val_vua, embedded_test_vua


def create_embeddings(data_type):
    randomize = False
    if data_type == 'trofi':
        tups = trofi_cleaning.get_trofi_tuples()
        randomize = True
        elmo_path_name = '../elmo/TroFi3737.hdf5'
    elif data_type == 'moh-x':
        tups = moh_x_cleaning.get_mohx_tuples()
        elmo_path_name = '../elmo/MOH-X_cleaned.hdf5'
    else:
        return create_vua_embeddings()

    vocab = util.get_vocab(tups)
    word2idx, idx2word = util.get_word2idx_idx2word(vocab)
    glove_embeddings = util.get_embedding_matrix(word2idx, idx2word, normalization=False)
    elmos = h5py.File(elmo_path_name, 'r')
    suffix_embeddings = nn.Embedding(2, 50)

    if randomize:
        random.seed(0)
        random.shuffle(tups)
    embedded_data = [[util.embed_sequence(example[0], example[1], word2idx, glove_embeddings, elmos, suffix_embeddings), example[2]]
                     for example in tups]
    return embedded_data


def ten_fold_cv(embeddings, data_type):
    sentences = [example[0] for example in embeddings]
    labels = [example[1] for example in embeddings]
    ten_folds = []
    if data_type == 'trofi':
        fold_size = int(3737/10)
    else:
        fold_size = 65

    for i in range(10):
        ten_folds.append((sentences[i * fold_size:(i + 1) * fold_size], labels[i * fold_size:(i + 1) * fold_size]))

    for i in range(10):
        training_sentences = []
        training_labels = []
        for j in range(10):
            if j != i:
                training_sentences.extend(ten_folds[j][0])
                training_labels.extend(ten_folds[j][1])
        training_dataset = TextDataset(training_sentences, training_labels)
        val_dataset = TextDataset(ten_folds[i][0], ten_folds[i][1])

    batch_size = 10
    train_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, collate_fn=TextDataset.collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, collate_fn=TextDataset.collate_fn)
    return train_dataloader, val_dataloader


def trofi_data_loaders():
    embeddings = create_embeddings('trofi')
    return ten_fold_cv(embeddings, 'trofi')


def moh_x_data_loaders():
    embeddings = create_embeddings('moh-x')
    return ten_fold_cv(embeddings, 'moh-x')


def vua_data_loaders():
    train_embeddings, val_embeddings, test_embeddings = create_embeddings('vua')
    train_dataset_vua = TextDataset([example[0] for example in train_embeddings],
                                    [example[1] for example in train_embeddings])
    val_dataset_vua = TextDataset([example[0] for example in val_embeddings],
                                  [example[1] for example in val_embeddings])
    test_dataset_vua = TextDataset([example[0] for example in test_embeddings],
                                   [example[1] for example in test_embeddings])

    batch_size = 64
    # Set up a DataLoader for the training, validation, and test dataset
    train_dataloader_vua = DataLoader(dataset=train_dataset_vua, batch_size=batch_size, shuffle=True,
                                      collate_fn=TextDataset.collate_fn)
    val_dataloader_vua = DataLoader(dataset=val_dataset_vua, batch_size=batch_size,
                                    collate_fn=TextDataset.collate_fn)
    test_dataloader_vua = DataLoader(dataset=test_dataset_vua, batch_size=batch_size,
                                     collate_fn=TextDataset.collate_fn)
    return train_dataloader_vua, val_dataloader_vua, test_dataloader_vua
