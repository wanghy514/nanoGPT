import torch
import torch.nn.functional as F

from model_util import (
    find_closest_match,
    batch_find_closest_match,
    apply_att_scaling,
    batch_apply_att_scaling,    
)

def test_find_closest_match():
    nh = 16
    d = 64
    att_mat = 1000 * torch.rand(nh, d, d)
    att_mat = F.softmax(att_mat, dim=-1)
    idx = 13
    attenuation_factor = 1
    att_scales = 1.0 + (att_mat.sum(dim=1)[idx] + torch.randn(d) * 0.1) / attenuation_factor
    idx1 = find_closest_match(att_mat, att_scales)
    assert idx == idx1
    print ("test_find_closest_match PASS")

def test_batch_find_closest_match():
    batch_size = 8
    nh = 16
    d = 64
    att_mat = 1000 * torch.rand(batch_size, nh, d, d)
    att_mat = F.softmax(att_mat, dim=-1)    
    attenuation_factor = 1   
    att_scales = 1.0 + torch.rand(batch_size, d) / attenuation_factor 
    # att_scales = torch.stack([
    #     1.0 + (att_mat[b,:,:,:].sum(dim=1)[idx] + torch.randn(d) * 0.1) / attenuation_factor
    #     for b in range(batch_size)
    # ])
    indices = batch_find_closest_match(att_mat, att_scales)
    # print ("indices=", indices)
    for b in range(batch_size):
        assert indices[b].item() == find_closest_match(att_mat[b], att_scales[b])

    print ("test_batch_find_closest_match PASS")

def test_batch_apply_att_scaling():
    
    d = 64
    att_mat = torch.rand(16, 8, d, d)
    att_mat = F.softmax(att_mat, dim=-1)
    attenuation_factor = 1
    att_scales = 1.0 + torch.rand(16, 64) / attenuation_factor

    indices = batch_find_closest_match(att_mat, att_scales)

    for apply_sca_to_one_head in [False, True]:
        att_mat0 = apply_att_scaling(att_mat.clone(), att_scales, apply_sca_to_one_head)
        att_mat1 = batch_apply_att_scaling(att_mat.clone(), att_scales, apply_sca_to_one_head)    
        err = torch.abs(att_mat0 - att_mat1).max()
        assert err == 0
            
    # print (att_mat[-1,:,:5,0])
    # print (att_mat1[-1,:,:5,0])        
    # print (diff.max())
    # print (diff[-1,:,:5,0])
    # print (diff[-1,:,0,:5])    
    diff = torch.abs(att_mat - att_mat1)
    # print (indices)    
    assert torch.all(indices == diff.sum(dim=(2,3)).argmax(dim=1))

    print ("test_batch_apply_att_scaling PASS")

if __name__ == "__main__": 

    test_find_closest_match()
    test_batch_find_closest_match()    
    test_batch_apply_att_scaling()