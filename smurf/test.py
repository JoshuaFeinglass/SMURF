import numpy as np
import torch
import pickle as pkl
import time
#2D joint only
def att_MI_np(attention,out_shape):
    ex_attention_MI = []
    for ex_num in range(0, len(out_shape)):
        heads = np.zeros((attention[0].shape[1],len(attention)))
        for layer_iter,layer in enumerate(attention):
            for head_iter in range(0, layer.shape[1]):
                heads[head_iter,layer_iter] = mutual_info_np(layer[ex_num, head_iter, 0:out_shape[ex_num], 0:out_shape[ex_num]].to("cpu")).reshape(1,-1)
        ex_attention_MI.append(heads)
    return [np.median(np.max(ex_attention_MI[i], axis=0)) for i in range(0, len(ex_attention_MI))]

def mutual_info_np(inp):
    inp = np.asarray(inp)
    joint_entropy = disc_joint_entropy_np(inp)
    marg_pdf_Y = mat_sum(inp / np.sum(inp), axis=0)
    marg_pdf_X = mat_sum(inp / np.sum(inp), axis=1)

    entropy_Y = disc_joint_entropy_np(marg_pdf_Y)
    entropy_X = disc_joint_entropy_np(marg_pdf_X)
    return (2 * (entropy_Y + entropy_X - joint_entropy) / (entropy_Y + entropy_X))

#customized mutliplication function for performance
def mat_sum(inp,axis):
    vec = np.ones(inp.shape[axis], dtype=np.uint8)
    if axis==1:
        res = np.matmul(inp, vec)
    elif axis==0:
        res = np.matmul(vec,inp)
    else:
        print('axis must be 1 or 0')
        res = None
    return res

def disc_joint_entropy_np(inp,axis=None):
    if axis == None:
        pdf = inp/np.sum(inp)
        return np.sum(-pdf[pdf > 0]*np.log2(pdf[pdf > 0]))
    else:
        sum_shape = np.asarray(inp.shape)
        sum_shape[axis] = 1
        sums = mat_sum(inp, axis=axis)
        sums = sums.reshape(tuple(sum_shape.astype(int)))

        tile = np.ones_like(inp.shape)
        tile[axis] = inp.shape[axis]
        pdfs = inp / np.tile(sums,tuple(tile.astype(int)))
        log_pdfs = np.log2(pdfs)
        log_pdfs[np.isinf(log_pdfs)] = 0
        log_pdfs[np.isnan(log_pdfs)] = 0
        return mat_sum(-pdfs*log_pdfs, axis=axis)

#############################################################################################################
def att_MI_torch(attention, out_shape):
    attention_MI = torch.zeros((len(attention),attention[0].shape[0],attention[0].shape[1]))
    full_att = torch.zeros((len(attention),attention[0].shape[0],attention[0].shape[1],out_shape[0],out_shape[0]))
    for iter_layer, layer in enumerate(attention):
        full_att[iter_layer,:,:,:,:] = layer
    for ex_num in range(0,len(out_shape)):
        attention_MI[:, ex_num, :] = mutual_info_torch(full_att[:,ex_num, :, 0:out_shape[ex_num], 0:out_shape[ex_num]])
    return list(torch.quantile(torch.max(attention_MI, dim=-1)[0], q=0.5,dim=0).to("cpu"))


def mutual_info_torch(inp):
    joint_entropy = disc_joint_entropy_torch(inp)
    marg_pdf_Y = torch.sum(inp / torch.sum(inp,dim=(-2,-1)).view(inp.shape[0],inp.shape[1],1,1).repeat(1,1,inp.shape[-2],inp.shape[-1]), dim=-2)
    marg_pdf_X = torch.sum(inp / torch.sum(inp,dim=(-2,-1)).view(inp.shape[0],inp.shape[1],1,1).repeat(1,1,inp.shape[-2],inp.shape[-1]), dim=-1)

    entropy_Y = disc_joint_entropy_torch(marg_pdf_Y, axis=-1)
    entropy_X = disc_joint_entropy_torch(marg_pdf_X, axis=-1)
    return (2 * (entropy_Y + entropy_X - joint_entropy) / (entropy_Y + entropy_X))

def disc_joint_entropy_torch(inp, axis=None):
    if axis == None:
        pdf = torch.div(inp,torch.sum(inp, dim=(-2,-1)).view(inp.shape[0],inp.shape[1],1,1).repeat(1,1,inp.shape[-2],inp.shape[-1]))
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('test_data.pkl', 'rb') as handle:
    in_data = pkl.load(handle)
attention,out_shape = in_data
hi = time.time()
np_out = att_MI_np(attention,out_shape)
print(time.time()-hi)
hi = time.time()
torch_out = att_MI_torch(attention,out_shape)
print(time.time()-hi)
hi = np.asarray(np_out)-np.asarray(torch_out)
print(hi)

