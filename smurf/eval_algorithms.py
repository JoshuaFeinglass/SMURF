import torch
from transformers import *
from transformers import AutoTokenizer
import numpy as np
from nltk.corpus import stopwords
import string
import nltk

STOP_WORDS = stopwords.words('english')

def att_MI_torch(attention, out_shape):
    attention_MI = torch.zeros((len(attention),attention[0].shape[0],attention[0].shape[1]))
    full_att = torch.zeros((len(attention),attention[0].shape[0],
                            attention[0].shape[1],out_shape[0],out_shape[0]))

    for iter_layer, layer in enumerate(attention):
        full_att[iter_layer,:,:,:,:] = layer
    for ex_num in range(0,len(out_shape)):
        attention_MI[:, ex_num, :] = mutual_info_torch(full_att[:,ex_num, :,
                                                       0:out_shape[ex_num], 0:out_shape[ex_num]])

    return list(torch.quantile(torch.max(attention_MI, dim=-1)[0],
                               q=0.5,dim=0).cpu().detach().numpy().astype(np.float))


def mutual_info_torch(inp):
    joint_entropy = disc_joint_entropy_torch(inp)
    marg_pdf_Y = torch.sum(inp / torch.sum(inp,dim=(-2,-1)).view(
        inp.shape[0],inp.shape[1],1,1).repeat(1,1,inp.shape[-2],inp.shape[-1]), dim=-2)
    marg_pdf_X = torch.sum(inp / torch.sum(inp,dim=(-2,-1)).view(
        inp.shape[0],inp.shape[1],1,1).repeat(1,1,inp.shape[-2],inp.shape[-1]), dim=-1)

    entropy_Y = disc_joint_entropy_torch(marg_pdf_Y, axis=-1)
    entropy_X = disc_joint_entropy_torch(marg_pdf_X, axis=-1)
    return (2 * (entropy_Y + entropy_X - joint_entropy) / (entropy_Y + entropy_X))

def disc_joint_entropy_torch(inp, axis=None):
    if axis == None:
        pdf = torch.div(inp,torch.sum(inp, dim=(-2,-1)).view(
            inp.shape[0],inp.shape[1],1,1).repeat(1,1,inp.shape[-2],inp.shape[-1]))
        log_pdf = torch.log2(pdf)
        log_pdf[torch.isinf(log_pdf)] = 0
        log_pdf[torch.isnan(log_pdf)] = 0
        pdf[torch.isinf(pdf)] = 0
        pdf[torch.isnan(pdf)] = 0
        return torch.sum(-pdf * log_pdf, dim=(-2,-1))
    else:
        sum_shape = np.asarray(inp.shape)
        sum_shape[axis] = 1
        sums = torch.sum(inp, dim=axis)
        sums = sums.view(tuple(sum_shape.astype(int)))
        tile = np.ones_like(inp.shape)
        tile[axis] = inp.shape[axis]
        pdfs = torch.div(inp,sums.repeat(tuple(tile.astype(int))))
        log_pdfs = torch.log2(pdfs)
        log_pdfs[torch.isinf(log_pdfs)] = 0
        log_pdfs[torch.isnan(log_pdfs)] = 0
        pdfs[torch.isinf(pdfs)] = 0
        pdfs[torch.isnan(pdfs)] = 0
        return torch.sum(-pdfs * log_pdfs, dim=axis)


##############################################################################################################
class compute_semantic():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.stemmer = nltk.stem.PorterStemmer()
    def method(self):
        return "SPARCS"

    def coalesce_to_concepts(self,sent_in):
        toks = nltk.word_tokenize(sent_in)
        return [self.stemmer.stem(tok) for tok in toks if tok not in list(STOP_WORDS) and
                       len(set(string.punctuation).intersection(set(tok)))==0]

    def create_val_dict(self,all_sent):
        sing_sents = []
        full_set = set()
        for sent in all_sent:
            non_stop_stemmed = self.coalesce_to_concepts(sent)
            sing_sents.append(non_stop_stemmed)
            full_set = full_set.union(non_stop_stemmed)
        stem_dict = dict()
        for concept in full_set:
            relevance = sum(concept in sent for sent in sing_sents)
            stem_dict[concept] = relevance/len(sing_sents)

        full_detail = sum(stem_dict.values())
        return stem_dict,full_detail,full_set

    def sent_seq_metric(self,sent,val_dict):
        non_stop_stemmed = self.coalesce_to_concepts(sent)
        cand = {}
        for y in non_stop_stemmed:
            if y in val_dict:
                cand[y] = val_dict[y]
            else:
                cand[y] = 1
        return cand

    def sim(self,cand,full_detail,full_ref_set):
        matches = 0
        match_list = []

        for sing_cand in cand.keys():
            if sing_cand in full_ref_set:
                matches += cand[sing_cand]
                match_list += [sing_cand]
        full_cand = sum(cand.values())

        if matches>0:
            prec = matches/full_cand
            rec = matches/full_detail
            return 2*prec*rec/(prec+rec)
        else:
            return 0

    def compute_score(self,all_cands,all_refs):
        scores = []
        for ex in zip(all_cands,all_refs):
            cand = ex[0]
            ref_set = ex[1]
            if len(cand)>0:
                ref_set_tok = []
                val_dict,full_detail,full_ref_set = self.create_val_dict(ref_set)
                for ref in ref_set:
                    if len(ref)>0:
                        ref_set_tok.append(self.sent_seq_metric(ref,val_dict))
                test_tok = self.sent_seq_metric(cand,val_dict)
                score = self.sim(test_tok, full_detail, full_ref_set)
                scores.append(score)
            else:
                scores.append(0)
        return scores

#computes SPURTS if distinctness set to True and MIMA grammar quality metric otherwise
class compute_quality():
    def __init__(self, use_roberta=1, distinctness=True, batch_size=64):
        self.use_roberta = use_roberta
        self.distinctness = distinctness
        self.batch_size = batch_size
        if (self.use_roberta):
            self.config = AutoConfig.from_pretrained('distilroberta-base', output_hidden_states=False,
                                                     output_attentions=True)
            self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', do_lower_case=True)
            self.model = AutoModel.from_pretrained('distilroberta-base', config=self.config)
            self.pad_tok = 1
            self.start_tok = 0
            self.end_tok = 2
            self.unk_tok = 3

        else:
            self.config = AutoConfig.from_pretrained("distilbert-base-uncased", output_hidden_states=False,
                                                     output_attentions=True)
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
            self.model = AutoModel.from_pretrained("distilbert-base-uncased", config=self.config)
            self.pad_tok = 0
            self.start_tok = 101
            self.end_tok = 102
            self.unk_tok = 100

    def method(self):
        if self.distinctness == 1:
            return "SPURTS"
        else:
            return "MIMA"

    def rm_stop_words(self,tok_in):
        tok_copy = []
        for tok in tok_in:
            check_tok= tok.replace('Ġ', '')
            if check_tok in list(STOP_WORDS):
                tok_copy.append(self.tokenizer.convert_ids_to_tokens(self.pad_tok))
            elif check_tok not in list(string.punctuation):
               tok_copy.append(tok)
        return tok_copy

    def compute_MIMA(self, tok_in):
        tok_in = [torch.tensor([self.start_tok] + self.tokenizer.convert_tokens_to_ids(tok_sent) + [self.end_tok]) \
                  for tok_sent in tok_in]
        out_shape = [len(tok_sent) for tok_sent in tok_in]
        net_in = torch.zeros((len(tok_in), len(tok_in[0])),dtype=torch.int64)

        for i,tok_sent in enumerate(tok_in):
           pad_len = len(tok_in[0])-len(tok_sent)
           padding = torch.tensor([self.pad_tok]*pad_len)
           net_in[i, :] = torch.cat((tok_sent, padding))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net_in = net_in.to(device)
        self.model.to(device)
        self.model.eval()

        attention_mask = torch.ones(net_in.shape)
        attention_mask[net_in == self.pad_tok] = 0
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model(net_in, attention_mask=attention_mask)
            attention = outputs[-1]
        del outputs
        metric_out = att_MI_torch(attention, out_shape)
        del attention
        return metric_out

    def compute_score(self,sent_in,ref_set=None):
        scores = []
        check_zero_len = []
        sent_toks = []
        for sent in sent_in:
            sent_toks.append(self.tokenizer.tokenize(sent))
            if len(sent)>0:
                check_zero_len.append(True)
            else:
                check_zero_len.append(False)

        if self.distinctness == True:
            sent_toks = [self.rm_stop_words(sent) for sent in sent_toks]

        sort_ind = sorted(range(len(sent_toks)), key=lambda k: len(sent_toks[k]), reverse=True)
        sentences = [sent_toks[i] for i in sort_ind]
        iter_range = range(0, len(sentences), self.batch_size)
        for batch_start in iter_range:
            sen_batch = sentences[batch_start: batch_start + self.batch_size]
            result = self.compute_MIMA(sen_batch)
            scores.extend(result)

        if self.distinctness == False:
            scores = [1-scores[i] for i in np.argsort(sort_ind)]
        else:
            scores = [scores[i] for i in np.argsort(sort_ind)]

        for i in range(0,len(scores)):
            if check_zero_len[i] == False:
                scores[i] = 0

        return scores