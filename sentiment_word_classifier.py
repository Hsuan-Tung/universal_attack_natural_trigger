"""
This file use sentences labelled as containing sentiment words and not containing sentiment words to
train a classifier. This classifier is intended to be used in searching process so that we can avoid
the sentences containing sentiment words.
@author: Xinwei
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


class SentimentWordDatasetReader(DatasetReader):
    """
    DatasetReader for sentimenet word sentence data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        fields["label"] = LabelField(tags)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as f:
            for line in f.readlines():
                pairs = line.strip("\n")
                sentence, tags = pairs.split("###")[0], pairs.split("###")[-1]
                sentence = sentence.split(' ')
                tags = tags[-1:]
                instance = self.text_to_instance(sentence, tags)
                if instance is not None:
                    yield instance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
    parser.add_argument('--batch_sz', type=int, default=128,
                        help='training batch size')
    parser.add_argument('--model_save_path', type=str, default='./model_dir',
                        help='path to save model')
    parser.add_argument('--train_data_path', type=str, default='./data/senti_sentence_20_train.txt',
                        help='path of train data')
    parser.add_argument('--dev_data_path', type=str, default='./data/senti_sentence_20_dev.txt',
                        help='path of dev data')
    parser.add_argument('--max_len', type=int, default=5,
                        help='maximum length of sentence')
    args = parser.parse_args()

    reader = SentimentWordDatasetReader()
    train_dataset = reader.read(args.train_data_path)
    validation_dataset = reader.read(args.dev_data_path)


    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True)  # word tokenizer

    # the model is sentiment classifier
    word_embedding_dim = 300
    EMBEDDING_TYPE = "w2v"
    vocab_path = "./model_dir/" + EMBEDDING_TYPE + "_" + "vocab"

    sst_vocab = Vocabulary.from_files(vocab_path)
    weight = torch.load('weight.pt')
    token_embedding = Embedding(num_embeddings=sst_vocab.get_vocab_size('tokens'),
                                embedding_dim=word_embedding_dim,
                                weight=weight,
                                trainable=False)

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    model = LstmClassifier(word_embeddings, encoder, sst_vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1


    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    iterator = BasicIterator(batch_size=args.batch_sz)
    iterator.index_with(sst_vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=1,
                      cuda_device=cuda_device)

    trainer.train()
    #
    # for batch in lazy_groups_of(iterator(validation_dataset, num_epochs=int(1e4), shuffle=False), group_size=1):
    #     batch = move_to_device(batch[0], cuda_device=0)
    #     tokens = batch['tokens']
    #     label = batch['label']
    #     print('tokens:{}'.format(tokens))
    #     print('label:{}'.format(label))
    #     with torch.no_grad():
    #         loss = model(tokens, label)
    #         print('loss:{}'.format(loss["loss"].item()))
    # Here's how to save the model.
    with open(os.path.join(args.model_save_path, "sentiment_word_model_{}.th".format(args.max_len)), 'wb') as f:
        torch.save(model.state_dict(), f)
        f.close()
