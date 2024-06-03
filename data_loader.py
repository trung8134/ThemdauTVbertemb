# -*- coding: utf-8 -*-

import pickle
import random

import tensorflow as tf
from tensorflow.keras import layers
from transformers import BertTokenizer

def get_text_pairs(file_path, limit=None):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()

    lines = [line.replace('\t', '\\t').strip() for line in lines]

    text_pairs = []

    if limit is None:
        limit = len(lines)
    i = 0
    for line in lines:
        stripped, original = line.split('\\t')
        original = '[start] ' + original
        text_pairs.append((stripped, original))
        i += 1
        if i >= limit:
            break

    return text_pairs


def split_pairs(text_pairs, ratio=.10, shuffle=False):
    if shuffle:
        random.shuffle(text_pairs)

    num_val_samples = int(ratio * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples:num_train_samples+num_val_samples]
    test_pairs = text_pairs[num_train_samples+num_val_samples:]

    return train_pairs, val_pairs, test_pairs

# func load data từ file path
def load_data(file_path, limit=None, ratio=.10, shuffle=False):
    text_pairs = get_text_pairs(file_path, limit)
    return split_pairs(text_pairs, ratio, shuffle)

# model vector hóa dữ liệu 
def create_vectorizations(train_pairs, sequence_length=50, vocab_size=15000, standardize=True):
    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length,
        standardize=standardize
    )
    train_stripped_texts = [n[0] for n in train_pairs]
    source_vectorization.adapt(train_stripped_texts)

    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length+1,
        standardize=standardize
    )
    train_original_texts = [n[1] for n in train_pairs]
    target_vectorization.adapt(train_original_texts)

    return source_vectorization, target_vectorization

def create_bert_tokenizations(train_pairs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_stripped_texts = [n[0] for n in train_pairs]
    train_original_texts = [n[1] for n in train_pairs]

    source_tokenization = tokenizer(train_stripped_texts, truncation=True, padding=True)
    target_tokenization = tokenizer(train_original_texts, truncation=True, padding=True)

    return source_tokenization, target_tokenization

# func sử dụng để lưu model weights vector hóa dữ liệu
def save_vectorization(vectorization, file_path):
    '''
    Save the config and weights of a vectorization to disk as joblib file,
    so that we can reuse it when making inference.
    '''

    with open(file_path, 'wb') as f:
        f.write(pickle.dump({'config': vectorization.get_config(),
            'weights': vectorization.get_weights()}), file_path(file_path))


# func sử dụng để load model weights vector hóa dữ liệu
def load_vectorization_from_disk(vectorization_path):
    '''
    Load a saved vectorization from disk.
    This method is based on the following answer on Stackoverflow.
    https://stackoverflow.com/a/65225240/4510614
    '''

    with open(vectorization_path, 'rb') as f:
        from_disk = pickle.load(f)
        new_v = layers.TextVectorization(max_tokens=from_disk['config']['max_tokens'],
            output_mode='int',
            output_sequence_length=from_disk['config']['output_sequence_length'])

        # You have to call `adapt` with some dummy data (BUG in Keras)
        new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        new_v.set_weights(from_disk['weights'])
    return new_v


def format_dataset(stripped, source_vectorization, original, target_vectorization):
    strip = source_vectorization(stripped)
    origin = target_vectorization(original)

    return ({
        'stripped': strip,
        'original': origin[:, :-1]
    }, origin[:, 1:])

# func tạo các tập train_ds, val_ds, test_ds
def make_dataset(pairs, source_vectorization, target_vectorization, batch_size):
    stripped_texts, original_texts = zip(*pairs)
    stripped_texts = list(stripped_texts)
    original_texts = list(original_texts)
    dataset = tf.data.Dataset.from_tensor_slices((stripped_texts, original_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: format_dataset(x, source_vectorization, y, target_vectorization), num_parallel_calls=8)
    return dataset.shuffle(2048).prefetch(16).cache()

# def main():
#     print(get_text_pairs('C:/Users/caotr/D. Computer Science/Data Science/DL/NLP/Project/ThemdauTVver2/test.txt'))
    
# if __name__ == "__main__":
#     main()
