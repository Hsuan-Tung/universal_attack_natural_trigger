"""
"""
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import argparse
import os
from allennlp.common.util import lazy_groups_of
from allennlp.data import Instance
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, Field
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.training.trainer import Trainer
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
import numpy as np
from allennlp.nn.util import get_text_field_mask, move_to_device


class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def forward_with_trigger(self, tri_emb, tokens, label):
        # calculate the loss with trigger append to the sentences
        trigger_len = tri_emb.size(1)
        tirgger_sz = tri_emb.size(0)
        if tokens is not None:
            mask = get_text_field_mask(tokens)
            mask = torch.cat((torch.ones(mask.size(0), trigger_len, dtype=mask.dtype, device=mask.device), mask), dim=1)
            # get embeddings of original sentence
            embeddings = self.word_embeddings(tokens)
            # add trigger to the sentence here
            out_emb = tri_emb.repeat(int(np.ceil(embeddings.size(0) / tirgger_sz)), 1, 1)
            embeddings = torch.cat((out_emb[:embeddings.size(0), :, :], embeddings), dim=1)
        else:
            mask = torch.ones(tirgger_sz, trigger_len, device=tri_emb.device)
            embeddings = tri_emb

        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output


    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}


