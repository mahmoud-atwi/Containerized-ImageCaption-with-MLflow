import torch
import numpy as np
from random import choice


def as_matrix(sequences, word_to_index, unk_ix, pad_ix, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    max_len = max_len or max(map(len, sequences))

    matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad_ix
    for i, seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix


def generate_batch(img_codes, captions, batch_size, word_to_index, unk_ix, pad_ix, max_caption_len=None):

    # sample random numbers for image/caption indicies
    random_image_ix = np.random.randint(0, len(img_codes), size=batch_size)

    # get images
    batch_images = img_codes[random_image_ix]

    # 5-7 captions for each image
    captions_for_batch_images = captions[random_image_ix]

    # pick one from a set of captions for each image
    batch_captions = list(map(choice, captions_for_batch_images))

    # convert to matrix
    batch_captions_ix = as_matrix(
        batch_captions, word_to_index, unk_ix, pad_ix, max_len=max_caption_len)

    return torch.tensor(batch_images, dtype=torch.float32), \
        torch.tensor(batch_captions_ix, dtype=torch.int64)
