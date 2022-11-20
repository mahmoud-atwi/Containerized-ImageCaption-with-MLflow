from collections import Counter
import json
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from beheaded_inception3 import beheaded_inception_v3

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import mlflow
import mlflow.sklearn
import mlflow.pytorch

import argparse

from utils import *

mlflow.set_tracking_uri('http://localhost:5000')

# Load dataset (vectorized images and captions)
img_codes = np.load("data/image_codes.npy")
captions = json.load(open('data/captions_tokenized.json'))

# split descriptions into tokens
for img_i in range(len(captions)):
    for caption_i in range(len(captions[img_i])):
        sentence = captions[img_i][caption_i]
        captions[img_i][caption_i] = ["#START#"] + \
            sentence.split(' ') + ["#END#"]

# Build a Vocabulary
word_counts = Counter()

# Compute word frequencies for each word in captions. See code above for data structure
for caption in captions:
    for words in caption:
        word_counts.update(words)

vocab = ['#UNK#', '#START#', '#END#', '#PAD#']
vocab += [k for k, v in word_counts.items() if v >= 5 if k not in vocab]
n_tokens = len(vocab)

with open('vocab.txt', 'w') as f:
    for e in vocab:
        f.write(e+'\n')

word_to_index = {w: i for i, w in enumerate(vocab)}

eos_ix = word_to_index['#END#']
unk_ix = word_to_index['#UNK#']
pad_ix = word_to_index['#PAD#']


class CaptionNet(nn.Module):

    def __init__(self, n_tokens=n_tokens, emb_size=128, lstm_units=256, cnn_feature_size=2048):
        """ A recurrent 'head' network for image captioning. See scheme above. """
        super().__init__()

        # a layer that converts conv features to initial_h (h_0) and initial_c (c_0)
        self.cnn_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_feature_size, lstm_units)

        self.emb = nn.Embedding(n_tokens, emb_size)

        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True)

        # linear layer that takes lstm hidden state as input and computes one number per token
        self.logits = nn.Linear(lstm_units, n_tokens)

    def forward(self, image_vectors, captions_ix):
        """ 
        Apply the network in training mode. 
        :param image_vectors: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]
        :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i]. 
            padded with pad_ix
        :returns: logits for next token at each tick, shape: [batch, word_i, n_tokens]
        """

        self.lstm.flatten_parameters()

        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)

        # compute embeddings for captions_ix
        captions_emb = self.emb(captions_ix)

        lstm_out, hidden = self.lstm(
            captions_emb, (initial_cell[None], initial_hid[None]))

        # compute logits from lstm_out
        logits = self.logits(lstm_out)

        return logits


criterion = CrossEntropyLoss(ignore_index=pad_ix)


def compute_loss(network, image_vectors, captions_ix):
    """
    :param image_vectors: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]
    :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i]. 
        padded with pad_ix
    :returns: crossentropy (neg llh) loss for next captions_ix given previous ones. Scalar float tensor
    """

    # captions for input - all except last.
    captions_ix_inp = captions_ix[:, :-1].contiguous()
    captions_ix_next = captions_ix[:, 1:].contiguous()

    # apply the network, get predictions for captions_ix_next
    logits_for_next = network.forward(image_vectors, captions_ix_inp)

    loss = criterion(logits_for_next.reshape(-1, n_tokens),
                     captions_ix_next.reshape(-1))

    return loss


DEVICE = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')


inception = beheaded_inception_v3().eval()


def train_model(model, optimizer, bs, bs_per_epoch, imgs, caps, word_to_index, unk_ix, pad_ix):
    train_loss = 0

    for it in tqdm(range(bs_per_epoch)):
        images, captions = generate_batch(imgs, caps, bs, word_to_index, unk_ix, pad_ix)
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        loss_t = compute_loss(model, images, captions)

        # clear old gradients
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        train_loss += loss_t.detach().cpu().numpy()

    train_loss /= bs_per_epoch
    mlflow.log_metric("train_loss", train_loss)

    return model


def test_model(model, batch_size, imgs, caps, vb, word_to_index, unk_ix, pad_ix):
    val_loss = 0
    model.eval()
    for _ in range(vb):
        images, captions = generate_batch(imgs, caps, batch_size, word_to_index, unk_ix, pad_ix)
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        with torch.no_grad():
            loss_t = compute_loss(model, images, captions)

        val_loss += loss_t.detach().cpu().numpy()
    val_loss /= vb
    mlflow.log_metric("validation_loss", val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image Caption Torchscripted model")

    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to run (default: 10)"
    )

    parser.add_argument(
        "--bs", type=int, default=128, help="batch size (default: 128)"
    )

    parser.add_argument(
        "--bpe", type=int, default=50, help="number of batches per epoch (default: 50)"
    )

    parser.add_argument(
        "--vb", type=int, default=50, help="number of validation batches (default: 5)"
    )

    args = parser.parse_args()

    model = CaptionNet(n_tokens).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    captions = np.array(captions, dtype=object)

    train_img_codes, val_img_codes, train_captions, val_captions = train_test_split(
        img_codes, captions, test_size=0.1, random_state=42)

    with mlflow.start_run() as run:
        for epoch in tqdm(range(args.epochs)):
            train_model(model, optimizer, args.bs, args.bpe,
                        train_img_codes, train_captions, word_to_index, unk_ix, pad_ix)
            test_model(model, args.bs, val_img_codes, val_captions, args.vb, word_to_index, unk_ix, pad_ix)
        
        mlflow.pytorch.log_model(model, "ImageCaption")
