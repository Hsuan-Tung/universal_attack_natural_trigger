import sys
import os
import torch
from transformers import *
import torch.optim as optim
import numpy as np
import random
import argparse
import json
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.common.file_utils import cached_path
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
from allennlp.nn.util import get_text_field_mask, move_to_device, weighted_sum, replace_masked_values, masked_log_softmax
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.models import load_archive
from allennlp.data.tokenizers import WordTokenizer
from allennlp.nn.util import masked_softmax, masked_max
sys.path.append('../')
from utils import get_embedding_weight, get_accuracy
from ARAE_utils import Seq2Seq, MLP_D, MLP_G, generate
from attack_util import select_fluent_trigger, project_noise, one_hot_prob

def SNLI_hypo_attack(model, premise, hypothesis, label, trigger_len, trigger_emb):
    """
    # Parameters
    premise : TextFieldTensors
        From a `TextField`
    hypothesis : TextFieldTensors
        From a `TextField`
    label : torch.IntTensor, optional (default = None)
        From a `LabelField`
    metadata : `List[Dict[str, Any]]`, optional, (default = None)
        Metadata containing the original tokenization of the premise and
        hypothesis with 'premise_tokens' and 'hypothesis_tokens' keys respectively.
    # Returns
    An output dictionary consisting of:
    label_logits : torch.FloatTensor
        A tensor of shape `(batch_size, num_labels)` representing unnormalised log
        probabilities of the entailment label.
    label_probs : torch.FloatTensor
        A tensor of shape `(batch_size, num_labels)` representing probabilities of the
        entailment label.
    loss : torch.FloatTensor, optional
        A scalar loss to be optimised.
    """

    embedded_premise = model._text_field_embedder(premise)
    embedded_hypothesis = model._text_field_embedder(hypothesis)
    premise_mask = get_text_field_mask(premise)
    hypothesis_mask = get_text_field_mask(hypothesis)
    # print(hypothesis_mask.shape, embedded_hypothesis.shape)
    if trigger_len > 0:
        trigger_mask = torch.ones((hypothesis_mask.shape[0], trigger_len), dtype=hypothesis_mask.dtype).cuda()
        embedded_trigger = trigger_emb.repeat(int(np.ceil(embedded_hypothesis.shape[0] / trigger_emb.shape[0])), 1, 1)
        embedded_hypothesis = torch.cat((embedded_trigger[:embedded_hypothesis.shape[0]], embedded_hypothesis), dim=1)
        hypothesis_mask = torch.cat((trigger_mask, hypothesis_mask), dim=1)
    # print(hypothesis_mask.shape, embedded_hypothesis.shape)

    # apply dropout for LSTM
    if model.rnn_input_dropout:
        embedded_premise = model.rnn_input_dropout(embedded_premise)
        embedded_hypothesis = model.rnn_input_dropout(embedded_hypothesis)

    # encode premise and hypothesis
    encoded_premise = model._encoder(embedded_premise, premise_mask)
    encoded_hypothesis = model._encoder(embedded_hypothesis, hypothesis_mask)

    # Shape: (batch_size, premise_length, hypothesis_length)
    similarity_matrix = model._matrix_attention(encoded_premise, encoded_hypothesis)

    # Shape: (batch_size, premise_length, hypothesis_length)
    p2h_attention = masked_softmax(similarity_matrix, hypothesis_mask)
    # Shape: (batch_size, premise_length, embedding_dim)
    attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

    # Shape: (batch_size, hypothesis_length, premise_length)
    h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
    # Shape: (batch_size, hypothesis_length, embedding_dim)
    attended_premise = weighted_sum(encoded_premise, h2p_attention)

    # the "enhancement" layer
    premise_enhanced = torch.cat(
        [
            encoded_premise,
            attended_hypothesis,
            encoded_premise - attended_hypothesis,
            encoded_premise * attended_hypothesis,
        ],
        dim=-1,
    )
    hypothesis_enhanced = torch.cat(
        [
            encoded_hypothesis,
            attended_premise,
            encoded_hypothesis - attended_premise,
            encoded_hypothesis * attended_premise,
        ],
        dim=-1,
    )

    # The projection layer down to the model dimension.  Dropout is not applied before
    # projection.
    projected_enhanced_premise = model._projection_feedforward(premise_enhanced)
    projected_enhanced_hypothesis = model._projection_feedforward(hypothesis_enhanced)

    # Run the inference layer
    if model.rnn_input_dropout:
        projected_enhanced_premise = model.rnn_input_dropout(projected_enhanced_premise)
        projected_enhanced_hypothesis = model.rnn_input_dropout(projected_enhanced_hypothesis)
    v_ai = model._inference_encoder(projected_enhanced_premise, premise_mask)
    v_bi = model._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)

    # The pooling layer -- max and avg pooling.
    # (batch_size, model_dim)
    v_a_max = masked_max(v_ai, premise_mask.unsqueeze(-1), dim=1)
    v_b_max = masked_max(v_bi, hypothesis_mask.unsqueeze(-1), dim=1)

    v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1) / torch.sum(
        premise_mask, 1, keepdim=True
    )
    v_b_avg = torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1) / torch.sum(
        hypothesis_mask, 1, keepdim=True
    )

    # Now concat
    # (batch_size, model_dim * 2 * 4)
    v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

    # the final MLP -- apply dropout to input, and MLP applies to output & hidden
    if model.dropout:
        v_all = model.dropout(v_all)

    output_hidden = model._output_feedforward(v_all)
    label_logits = model._output_logit(output_hidden)
    label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

    output_dict = {"label_logits": label_logits, "label_probs": label_probs}

    if label is not None:
        loss = model._loss(label_logits, label.long().view(-1))
        model._accuracy(label_logits, label)
        output_dict["loss"] = loss

    return output_dict, label_logits

def select_subset_snli(dataset, data_type='entailment'):
    # Subsample the dataset to one class to do a universal attack on that class
    dataset_label_filter = data_type 
    subset_dev_dataset = []
    count = 0
    for instance in dataset:
        if instance['label'].label == dataset_label_filter:
            subset_dev_dataset.append(instance)
            count += 1
    print(str(count)+' examples in total')
    return subset_dev_dataset

def load_ARAE_models(load_path, args):
    # function to load ARAE model.
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
    parser.add_argument('--attack_class', type=str, default='contradiction',
                        help='the class label to attack')
    parser.add_argument('--r_lim', type=float, default=1,
                        help='lim of radius of z')
    parser.add_argument('--z_seed', type=float, default=6.,
                        help='noise seed for z')
    parser.add_argument('--avoid_l', type=int, default=2,
                        help='length to avoid repeated pattern')
    parser.add_argument('--lr', type=float, default=1e3,
                        help='learn rate')
    parser.add_argument('--noise_n', type=int, default=32,
                        help='number of generated noise vectors')
    parser.add_argument('--tot_runs', type=int, default=8,
                        help='number of attack runs')
    args = parser.parse_args()

    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    ARAE_args, ARAE_idx2word, ARAE_word2idx, autoencoder, gan_gen, gan_disc = load_ARAE_models(args.load_path, args)

    single_id_indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
    # use_subtrees gives us a bit of extra data by breaking down each example into sub sentences.

    tokenizer = WordTokenizer(end_tokens=["@@NULL@@"]) # add @@NULL@@ to the end of sentences

    reader = SnliReader(token_indexers={'tokens': single_id_indexer}, tokenizer=tokenizer)

    dev_dataset = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/snli/snli_1.0_dev.jsonl')
    # Load model and vocab
    model = load_archive('https://allennlp.s3-us-west-2.amazonaws.com/models/esim-glove-snli-2019.04.23.tar.gz').model
    model.train().cuda()
    snli_vocab = model.vocab


    mask_word_ARAE = []
    ARAE_words = list(ARAE_word2idx.keys())
    for word in ARAE_words:
        if snli_vocab.get_token_index(word) == 1:

            mask_word_ARAE.append(ARAE_word2idx[word])

    mask_word_ARAE = np.array(list(set(mask_word_ARAE)))

    mask_ARAE_logits = np.zeros((1, 1, len(ARAE_words)))
    mask_ARAE_logits[:, :, mask_word_ARAE] = -float("Inf")
    mask_ARAE_logits = torch.tensor(mask_ARAE_logits, requires_grad=False)
    mask_ARAE_logits = mask_ARAE_logits.float().cuda()
    mask_ARAE = mask_ARAE_logits[0]

    embedding_weight = get_embedding_weight(model)
    ARAE_weight_embedding = []
    for num in range(len(ARAE_idx2word)):
        ARAE_weight_embedding.append(embedding_weight[snli_vocab.get_token_index(ARAE_idx2word[num])].numpy())
    ARAE_weight_embedding = torch.from_numpy(np.array(ARAE_weight_embedding)).cuda()

    universal_perturb_batch_size = 256  # 157
    iterator = BasicIterator(batch_size=universal_perturb_batch_size)
    iterator.index_with(snli_vocab)


    target_data_type = args.attack_class
    targeted_dev_data = select_subset_snli(dev_dataset, data_type=target_data_type)


    maxlen = args.len_lim
    step_size = args.lr 
    step_scale = 0.1 

    patience_lim = 2
    max_trial = 3

    # initialize noise
    noise_n = args.noise_n  # this should be a factor of batch_size
    n_repeat = 1

    r_threshold = args.r_lim
    step_bound = r_threshold / 100
    max_iterations = 1000

    tot_runs = args.tot_runs
    all_output = list()
    log_loss = int(1e2)
    
    for tmp in range(tot_runs):
        model.get_metrics(reset=True)
        step_size = args.lr
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
            batch_data = move_to_device(batch[0], cuda_device=0)

            model.train()
            autoencoder.train()
            gan_gen.eval()
            gan_disc.eval()

            hidden = gan_gen(noise)

            max_indices, decoded = autoencoder.generate_decoding(hidden=hidden, maxlen=maxlen, sample=False,
                                                                 mask=mask_ARAE, avoid_l=args.avoid_l)

            decoded = torch.stack(decoded, dim=1).float()
            if n_repeat > 1:
                decoded = torch.repeat_interleave(decoded, repeats=n_repeat, dim=0)
            decoded_prob = F.softmax(decoded, dim=-1)
            decoded_prob = one_hot_prob(decoded_prob, max_indices)
            out_emb = torch.matmul(decoded_prob, ARAE_weight_embedding)

            model.get_metrics(reset=True)
            output, logits = SNLI_hypo_attack(model, batch_data['premise'], batch_data['hypothesis'],
                                              batch_data['label'], maxlen, out_emb)

            loss = output["loss"]

            iter += 1
            loss_list.append(loss.item())

            zero_gradients(noise)
            loss.backward()

            noise_diff = step_size * (noise.grad.data)
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
                                                                             mask=mask_ARAE, avoid_l=args.avoid_l)

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
                            cur_idx = snli_vocab.get_token_index(word)
                            if cur_idx != last_word and cur_idx != last_word2:
                                trigger_token_ids.append(cur_idx)
                                new_sentence.append(word)
                                last_word2 = last_word
                                last_word = cur_idx

                        threshold = 0.8
                        num_lim = 20
                        s_str = glue.join(new_sentence)
                        if not (s_str in sentence_list):
                            accuracy = get_accuracy(model, targeted_dev_data, snli_vocab, trigger_token_ids,
                                                       snli=True)
                            if accuracy < threshold:
                                sentence_list.append(s_str)
                                output_s.append((s_str, accuracy))

                    if len(output_s) > 0:
                        all_output = all_output + output_s
                    update = False
                break

    # use gpt2 for post analysis.
    GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    GPT2_model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()

    select_fluent_trigger(all_output, GPT2_model, GPT2_tokenizer)