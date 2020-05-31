import torch
import numpy as np
import json
import os



def one_hot_prob(y, ind):
    # covert the probability to one-hot coding in a differentiable way.
    shape = y.size()
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y

    return y_hard

def project_noise(noise, r_threshold=2):
    # project the noise into the ball with radius r_threshold
    # since radius is correlated with dimension of noise, we treat the r_threshold as
    # effective radius for each dimension.
    # noise is a tensor of dimension
    noise_dim = noise.size(1)
    r_threshold_alldim = r_threshold ** 2 * noise_dim
    with torch.no_grad():
        noise_radius = torch.sum(noise ** 2, dim=1)
        mask_proj = noise_radius > r_threshold_alldim
        noise_radius = torch.sqrt(noise_radius)
        noise[mask_proj, :] = noise[mask_proj, :]/torch.unsqueeze(noise_radius[mask_proj], dim=1) * np.sqrt(r_threshold_alldim)

    return noise

def GPT2_LM_loss(GPT2_model, GPT2_tokenizer, text):
    # print(text, text+'.')
    tokenize_input = GPT2_tokenizer.tokenize(text)
    tensor_input = torch.tensor(
        [[GPT2_tokenizer.bos_token_id] + GPT2_tokenizer.convert_tokens_to_ids(tokenize_input)
         + [GPT2_tokenizer.eos_token_id]]).cuda()
    with torch.no_grad():
        outputs = GPT2_model(tensor_input, labels=tensor_input)
        loss, logits = outputs[:2]
    return loss.data.cpu().numpy() * len(tokenize_input) / len(text.split())

def select_fluent_trigger(all_candidates, GPT2_model, GPT2_tokenizer, acc_range=0.1, top_ratio=0.05):
    if len(all_candidates) > 0:
        triggers_acc_sort = sorted(all_candidates, key=lambda x: x[1])
        trigger_list, acc_list, gpt_list = [], [], []
        num_tmp = 0
        
        for trigger in triggers_acc_sort:
            gpt_loss = GPT2_LM_loss(GPT2_model, GPT2_tokenizer, trigger[0])
            print(trigger[0], trigger[1], gpt_loss)
            trigger_list.append(trigger[0])
            acc_list.append(trigger[1])
            gpt_list.append(gpt_loss)
            if  num_tmp == 0:
                acc_bound = trigger[1] + acc_range
            num_tmp += 1
            if trigger[1] >= acc_bound:
                break
            if num_tmp >= top_ratio*len(all_candidates):
                break
        trigger_list, acc_list, gpt_list = np.array(trigger_list), np.array(acc_list), np.array(gpt_list)
        overall_perform = acc_list+gpt_list/20
        best_index = np.argmin(overall_perform)
        print('=== best trigger is " ', trigger_list[best_index], ' " with model accuracy ', acc_list[best_index])
        return
    else:
        print('no trigger below threshold.')
        return
