import sys
import os
import torch
from transformers import *
import torch.optim as optim
from sst_classifier import LstmClassifier
import numpy as np
import random
import argparse
import json
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, move_to_device
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.common.util import lazy_groups_of
from allennlp.data.token_indexers import SingleIdTokenIndexer
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
sys.path.append('../')
from ARAE_utils import Seq2Seq, MLP_D, MLP_G, generate
from utils import get_embedding_weight, get_accuracy
from attack_util import project_noise, one_hot_prob, GPT2_LM_loss, select_fluent_trigger

def load_ARAE_models(load_path, args):
    # function to load ARAE model.
    if not os.path.exists(load_path):
        print('Please download the pretrained ARAE model first')
        
    ARAE_args = json.load(open(os.path.join(load_path, 'options.json'), 'r'))
    vars(args).update(ARAE_args)
    autoencoder = Seq2Seq(emsize=args.emsize,
                          nhidden=args.nhidden,
                          ntokens=args.ntokens,
                          nlayers=args.nlayers,
                          noise_r=args.noise_r,
                          hidden_init=args.hidden_init,
                          dropout=args.dropout,
                          gpu=args.cuda)
    gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
    gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)

    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()

    ARAE_word2idx = json.load(open(os.path.join(args.load_path, 'vocab.json'), 'r'))
    ARAE_idx2word = {v: k for k, v in ARAE_word2idx.items()}

    print('Loading models from {}'.format(args.load_path))
    loaded = torch.load(os.path.join(args.load_path, "model.pt"))
    autoencoder.load_state_dict(loaded.get('ae'))
    gan_gen.load_state_dict(loaded.get('gan_g'))
    gan_disc.load_state_dict(loaded.get('gan_d'))
    return ARAE_args, ARAE_idx2word, ARAE_word2idx, autoencoder, gan_gen, gan_disc
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='../ARAE/oneb_pretrained',
                        help='directory to load models from')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--sample', action='store_true',
                        help='sample when decoding for generation')
    parser.add_argument('--len_lim', type=int, default=5,
                        help='maximum length of sentence')
    parser.add_argument('--r_lim', type=float, default=1,
                        help='lim of radius of z')
    parser.add_argument('--sentiment_path', type=str, default='./opinion_lexicon_English',
                        help='directory to load sentiment word from')
    parser.add_argument('--z_seed', type=float, default=6.,
                        help='noise seed for z')
    parser.add_argument('--avoid_l', type=int, default=4,
                        help='length to avoid repeated pattern')
    parser.add_argument('--lr', type=float, default=1e3,
                        help='learn rate')
    parser.add_argument('--attack_class', type=str, default='1',
                        help='the class label to attack')
    parser.add_argument('--noise_n', type=int, default=256,
                        help='number of generated noise vectors')
    parser.add_argument('--tot_runs', type=int, default=1,
                        help='number of attack runs')
    args = parser.parse_args()

    

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # initialize ARAE model.
    ARAE_args, ARAE_idx2word, ARAE_word2idx, autoencoder, gan_gen, gan_disc = load_ARAE_models(args.load_path, args)

    # load pretrained sentiment analysis model.
    word_embedding_dim = 300
    EMBEDDING_TYPE = "w2v"
    vocab_path = "./model_dir/" + EMBEDDING_TYPE + "_" + "vocab"

    sst_vocab = Vocabulary.from_files(vocab_path)
    weight = torch.load('sst_emb_weight.pt')
    token_embedding = Embedding(num_embeddings=sst_vocab.get_vocab_size('tokens'),
                                embedding_dim=word_embedding_dim,
                                weight=weight,
                                trainable=False)

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim,
                                                  hidden_size=512,
                                                  num_layers=2,
                                                  batch_first=True))
    model_path = "./model_dir/" + EMBEDDING_TYPE + "_" + "model.th"
    model = LstmClassifier(word_embeddings, encoder, sst_vocab)

    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        f.close()
    embedding_weight = get_embedding_weight(model)
    model.train().cuda()

    # arange ARAE word embedding in consistent with sst model.
    ARAE_weight_embedding = []
    for num in range(len(ARAE_idx2word)):
        ARAE_weight_embedding.append(embedding_weight[sst_vocab.get_token_index(ARAE_idx2word[num])].numpy())
    ARAE_weight_embedding = torch.from_numpy(np.array(ARAE_weight_embedding)).cuda()

    ### collect positive/negative sentences
    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True)  # word tokenizer
    reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class",
                                                    token_indexers={"tokens": single_id_indexer})
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')


    # For sentiment analysis, get rid of positive and negative.
    pos_path = os.path.join(args.sentiment_path, 'positive_words.txt')
    neg_path = os.path.join(args.sentiment_path, 'negative_words.txt')

    pos_words = list()
    with open(cached_path(pos_path), "r") as data_file:
        for line in data_file.readlines():
            if line[0] != ';':
                line = line.strip("\n")
                if not line:
                    continue
                else:
                    pos_words.append(line)

    neg_words = list()
    with open(cached_path(neg_path), "r", encoding = "ISO-8859-1") as data_file:
        for line in data_file.readlines():
            if line[0] != ';':
                line = line.strip("\n")
                if not line:
                    continue
                else:
                    neg_words.append(line)

    my_list = ['missing', 'rapes']
    sentiment_words = pos_words + neg_words + my_list

    mask_word_ARAE = list()
    for word in sentiment_words:
        if word in ARAE_word2idx:
            mask_word_ARAE.append(ARAE_word2idx[word])

    # mask words that are unknown for sentiment words.
    ARAE_words = list(ARAE_word2idx.keys())
    for word in ARAE_words:
        if sst_vocab.get_token_index(word) == 1:
            mask_word_ARAE.append(ARAE_word2idx[word])
    mask_word_ARAE = list(set(mask_word_ARAE))
    sent_word_ARAE = np.array(mask_word_ARAE)

    mask_sentiment_logits = np.zeros((1, 1, len(ARAE_words)))
    mask_sentiment_logits[:, :, sent_word_ARAE] = -float("Inf")
    mask_sentiment_logits = torch.tensor(mask_sentiment_logits, requires_grad=False)
    mask_sentiment_logits = mask_sentiment_logits.float().cuda()
    mask_sentiment = mask_sentiment_logits[0]

    dataset_label_filter = args.attack_class
    targeted_dev_data = []
    
    for instance in dev_data:
        if instance['label'].label == dataset_label_filter:
            targeted_dev_data.append(instance)



    model.get_metrics(reset=True)

    iterator = BucketIterator(batch_size=256, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(sst_vocab)
    get_accuracy(model, targeted_dev_data, sst_vocab, trigger_token_ids=None)
    
    
    maxlen = args.len_lim
    # initialize noise
    noise_n = args.noise_n  # this should be a factor of batch_size
    tot_runs = args.tot_runs
    n_repeat = 1

      
    r_threshold = args.r_lim
    step_bound = r_threshold / 100
    max_iterations = 1000

    patience_lim = 3
    patience = 0 
    max_trial = 3
    all_output = list()
    log_loss = int(1e2)
    
    for tmp in range(tot_runs):
        model.get_metrics(reset=True)
        step_size = args.lr
        step_scale = 0.1 
        patience = 0
        old_noise = None
        old_loss = float('-Inf')
        loss_list = list()
        update = False
        i_trial = 0

        torch.manual_seed(args.z_seed + tmp)
        print('z_seed:{}'.format(args.z_seed + tmp))
        noise = torch.randn(noise_n, ARAE_args['z_size'], requires_grad=True).cuda()
        noise = Variable(noise, requires_grad=True)
        start_noise_data = noise.data.clone()
        iter = 0
        for batch in lazy_groups_of(iterator(targeted_dev_data, num_epochs=int(5e5), shuffle=True), group_size=1):
            # evaluate_batch(model, batch, trigger_token_ids, snli)
            # generate sentence with ARAE, output the word embedding instead of index.
            batch = move_to_device(batch[0], cuda_device=0)
            tokens = batch['tokens']
            label = batch['label']

            model.train()
            autoencoder.train()
            gan_gen.eval()
            gan_disc.eval()

            hidden = gan_gen(noise)


            max_indices, decoded = autoencoder.generate_decoding(hidden=hidden, maxlen=maxlen, sample=False,
                                                                 mask=mask_sentiment, avoid_l=args.avoid_l)

            decoded = torch.stack(decoded, dim=1).float()
            if n_repeat > 1:
                decoded = torch.repeat_interleave(decoded, repeats=n_repeat, dim=0)

            decoded_prob = F.softmax(decoded, dim=-1)
            decoded_prob = one_hot_prob(decoded_prob, max_indices)
            out_emb = torch.matmul(decoded_prob, ARAE_weight_embedding)
            output = model.forward_with_trigger(out_emb, tokens, label)

            loss = output["loss"]
            iter += 1

            loss_list.append(output["loss"].item())
            zero_gradients(noise)
            loss.backward()

            noise_diff = step_size * noise.grad.data
            noise_diff = project_noise(noise_diff, r_threshold=step_bound)

            noise.data = noise.data + noise_diff

            whole_diff = noise.data - start_noise_data
            whole_diff = project_noise(whole_diff, r_threshold=r_threshold)
            noise.data = start_noise_data + whole_diff

            if iter % log_loss == 0:
                cur_loss = np.mean(loss_list)
                print('current iter:{}'.format(iter))
                print('current loss:{}'.format(cur_loss))

                loss_list = list()
                if cur_loss > old_loss:
                    patience = 0
                    old_loss = cur_loss
                    old_noise = noise.data.clone()
                    update = True
                else:
                    patience += 1

                print('current patience:{}'.format(patience))
                print('\n')

                if patience >= patience_lim:
                    patience = 0
                    step_size *= step_scale
                    noise.data = old_noise
                    print('current step size:{}'.format(step_size))
                    i_trial += 1
                    print('current trial:{}'.format(i_trial))
                    print('\n')

            if i_trial >= max_trial or iter >= max_iterations:
                if update:
                    with torch.no_grad():
                        noise_new = torch.ones(noise_n, ARAE_args['z_size'], requires_grad=False).cuda()
                        noise_new.data = old_noise
                        hidden = gan_gen(noise_new)  # [:1, :]
                        max_indices, decoded = autoencoder.generate_decoding(hidden=hidden, maxlen=maxlen, sample=False,
                                                                             mask=mask_sentiment, avoid_l=args.avoid_l)

                        decoded = torch.stack(decoded, dim=1).float()
                        if n_repeat > 1:
                            decoded = torch.repeat_interleave(decoded, repeats=n_repeat, dim=0)

                        decoded_prob = F.softmax(decoded, dim=-1)
                        decoded_prob = one_hot_prob(decoded_prob, max_indices)

                    sen_idxs = torch.argmax(decoded_prob, dim=2)
                    sen_idxs = sen_idxs.cpu().numpy()

                    output_s = list()
                    glue = ' '
                    sentence_list = list()
                    for ss in sen_idxs:
                        sentence = [ARAE_idx2word[s] for s in ss]
                        trigger_token_ids = list()
                        last_word = None
                        last_word2 = None
                        contain_sentiment_word = False
                        new_sentence = list()
                        for word in sentence:
                            cur_idx = sst_vocab.get_token_index(word)
                            if cur_idx != last_word and cur_idx != last_word2:
                                trigger_token_ids.append(cur_idx)
                                new_sentence.append(word)
                                last_word2 = last_word
                                last_word = cur_idx

                                if word in sentiment_words:
                                    contain_sentiment_word = True

                        threshold = 0.5
                        num_lim = 20
                        s_str = glue.join(new_sentence)
                        if not (s_str in sentence_list):
                            accuracy = get_accuracy(model, targeted_dev_data, sst_vocab, trigger_token_ids)
                            if accuracy < threshold:
                                sentence_list.append(s_str)
                                output_s.append((s_str, accuracy, contain_sentiment_word))

                    if len(output_s) > 0:
                        all_output = all_output + output_s
                    update = False
                break



    # use GPT2 for post selection.
    GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT2_model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

    triggers = all_output
    select_fluent_trigger(triggers, GPT2_model, GPT2_tokenizer)
