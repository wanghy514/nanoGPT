import torch
from enum import Enum


class ARHeadChoice(Enum):
    ALL = "ALL"
    FIRST = "FIRST"
    CLOSEST = "CLOSEST"



def batch_find_closest_match(att_mat, att_scales):

    """
    Find indices so that att_mat[b, indices[b], :, :].sum(dim=0) resembles att_scales[b,:] for each b
    att_mat.shape = (B, nh, T, T)
    att_scales = (B, T,)
    """
    B = att_scales.size(0)
    T = att_mat.size(-1)
    # print (att_mat.shape)
    similarities = torch.nn.functional.cosine_similarity(att_mat.sum(dim=2), att_scales.view(B, 1, T), dim=-1)
    indices = torch.argmax(similarities, dim=-1)
    # print ("similarities.shape=", similarities.shape)
    # print ("idx=", idx)
    return indices


def find_closest_match(att_mat, att_scales):

    """
    Find idx so that att_mat[idx, :, :].sum(dim=0) resembles att_scales
    att_mat.shape = (nh, T, T)
    att_scales = (T,)
    """

    T = att_mat.size(-1)
    similarities = torch.nn.functional.cosine_similarity(att_mat.sum(dim=1), att_scales.view(1, T))
    idx = torch.argmax(similarities)
    # print ("similarities=", similarities)
    # print ("idx=", idx)
    return idx.item()


def apply_att_scaling(att: torch.Tensor, att_scales: torch.Tensor, ar_head_choice: ARHeadChoice):

    """
    In place operation on att.

    att.shape = (B, nh, T, T)
    att_scales.shape = (B, T)
    """

    # print ("att before = ", att.shape, att) # (B, nh, T, T)
    # Each row of att corresponds to a query. Each column of att corresponds to a key.
    # Therefore att_scales should be multiplied to columns (keys)
    # print ("att_scales.shape=", att_scales.shape)
    # print ("att.shape=", att.shape)

    B = att.size(0)
    T = att.size(-1)
    if ar_head_choice == "CLOSEST":
        for b in range(B):
            head_idx = find_closest_match(att[b,:,:,:], att_scales[b,:])
            att[b,head_idx,:,:] *= att_scales[b].view(1,T)
    elif ar_head_choice == "FIRST":
        for b in range(B):            
            att[b,0,:,:] *= att_scales[b].view(1,T)    
    elif ar_head_choice == "ALL":
        att *= att_scales.view(B, 1, 1, T)
    else:
        raise ValueError

    #print ("att after = ", att.shape, att)
    att /= att.sum(dim=-1, keepdim=True)
    #print ("att normalized = ", att.shape, att)

    return att

def batch_apply_att_scaling(att: torch.Tensor, att_scales: torch.Tensor, ar_head_choice: ARHeadChoice):

    """
    In place operation on att.

    att.shape = (B, nh, T, T)
    att_scales.shape = (B, T)
    """    

    B = att.size(0)
    nh = att.size(1)
    T = att.size(-1)
    if ar_head_choice == "CLOSEST":
        head_indices = batch_find_closest_match(att, att_scales) # shape = (B,)
        att_scales_scattered = torch.ones((B, nh, 1, T))
        att_scales_scattered.scatter_(
            dim=1, 
            index=torch.tile(head_indices.view(B,1,1,1), dims=(1,1,1,T)),
            src=att_scales.view(B,1,1,T)
        )
        rescaled_att = att * att_scales_scattered      
    elif ar_head_choice == "FIRST":        
        att_scales_scattered = torch.cat(
            [att_scales.view(B, 1, 1, T), torch.ones((B, nh-1, 1, T))],
            dim=1
        )
        rescaled_att = att * att_scales_scattered
    elif ar_head_choice == "ALL":
        rescaled_att = att * att_scales.view(B, 1, 1, T)
    else:
        raise ValueError
    
    renormalized_att = rescaled_att / rescaled_att.sum(dim=-1, keepdim=True)        

    return renormalized_att
