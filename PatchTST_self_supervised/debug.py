import torch
import IPython
from torch.nn.functional import normalize
# [bs x num_patch x n_vars x embedding_dim]
test_tensor = torch.Tensor(
    [
        [
            [
                [ dim*0.001 + k*0.01 + j*0.1 + i for dim in range(5)]
                for k in range(1,3)
            ] # two channels
            for j in range(1,6)
        ] # 5 patches; 
        for i in range(1,4)
    ] # batch size 3; 
) 

def get_neg_samples(num_neg_samples,contrastive_pred):
    '''
        Returns negative samples :: [ bs x num_patch - 1 x num_neg_samples x n_vars x embedding_dim ]
    '''
    
    bs, num_patch,n_vars,embedding_dim = contrastive_pred.shape
    
    # negative pair different backbone random patch
    probs = torch.log(torch.ones(bs,bs).fill_diagonal_(0))
    bs_index = torch.distributions.categorical.Categorical(logits=probs).sample([num_neg_samples*(num_patch-1)]) # [ num_neg_samples*num_patch x bs ]
    bs_index = bs_index.transpose(0,1) # [ bs x num_neg_samples*num_patch ]

    # sample negative by batch dim
    bs_neg_samples_idx = bs_index.view(bs,(num_patch-1),num_neg_samples,1,1).repeat(1,1,1,n_vars,embedding_dim) # [ bs x num_patch - 1 x num_neg_samples x n_vars x embedding_dim ]

    probs = torch.log(torch.ones(num_patch,num_patch)[:-1]) # 4 numbers uniform [0,patch_num]
    patch_index = torch.distributions.categorical.Categorical(logits=probs).sample([bs*num_neg_samples]) # [ bs*num_neg_samples x num_patch - 1]
    patch_index = patch_index.reshape(bs,num_patch -1, num_neg_samples) # [ bs x num_patch -1 x num_neg_samples ]

    # sample negatives by patch index from batch samples
    patch_neg_samples_idx = patch_index.unsqueeze(3).unsqueeze(4)
    patch_neg_samples_idx = patch_neg_samples_idx.repeat(1,1,1,n_vars,embedding_dim)

    neg_samples_idx = bs_neg_samples_idx * num_patch + patch_neg_samples_idx


    contrastive_pred_view = contrastive_pred.reshape(-1,n_vars,embedding_dim)
    contrastive_pred_view = contrastive_pred_view.unsqueeze(1).repeat(1,num_neg_samples,1,1)
    neg_samples_idx_view = neg_samples_idx.reshape(-1,num_neg_samples ,n_vars,embedding_dim)

    neg_samples = torch.gather(contrastive_pred_view,0,neg_samples_idx_view)

    neg_samples = neg_samples.reshape(bs,num_patch-1,num_neg_samples,n_vars,embedding_dim)
    
    assert neg_samples.shape == (bs, num_patch - 1,num_neg_samples, n_vars,embedding_dim)
    
    return neg_samples

def get_pos_sample(contrastive_pred):
    '''
        Return positive sample :: [bs x num_patch - 1 x num_vars x embedding_dim]
    '''
    bs, num_patch,n_vars,embedding_dim = contrastive_pred.shape
    is_future_mask = torch.triu(torch.ones(num_patch,num_patch), diagonal=1) 

    discount = torch.stack([torch.arange(num_patch)-i for i in range(num_patch)])
    discount = torch.pow(torch.ones(num_patch,num_patch)*0.99,discount)
    probs = is_future_mask * discount
    probs = torch.log(probs[:-1]) # last one has no future :(

    index = torch.distributions.categorical.Categorical(logits=probs).sample([bs])
    index = index.unsqueeze(2).unsqueeze(3).repeat(1,1,n_vars,embedding_dim) # match dims
    pos_sample = torch.gather(contrastive_pred,1,index)
    assert pos_sample.shape == (bs, num_patch - 1 , n_vars,embedding_dim)
    return pos_sample

def InfoNCE(pos_pair,neg_pair):
    pred, pos_sample = pos_pair
    pred, pos_sample = normalize(pred,dim=3), normalize(pos_sample,dim=3)
    pos = torch.einsum('ijkl,ijkl->ijk', pred,pos_sample) # [ bs x num_patch x n_dim ]
    pos = pos.unsqueeze(3) # [ bs x num_patch x n_dim x 1]
    
    neg_pred, neg_samples = neg_pair
    neg_pred, neg_samples = normalize(neg_pred,dim=4), normalize(neg_samples,dim=4)
    neg = torch.einsum('ijklm,ijklm->ijkl', neg_pred,neg_samples) # [ bs x num_patch x num_neg_samples x n_dim ]
    neg = neg.permute(0,1,3,2) # [ bs x num_patch x n_dim x num_neg_samples  ]
    
    target = torch.cat([pos,neg], dim = 3)
    target = target.reshape(-1, target.shape[-1])

    label = torch.zeros(target.shape[0])
    
    return target, label
    

num_neg_samples = 20
tau = 0.1

contrastive_pred = test_tensor
# positive pair same backbone [bs x num_patch x n_vars x embedding_dim]
prediction = contrastive_pred[:,:-1, :, :] 
pos_sample = get_pos_sample(contrastive_pred)
pos_pair = prediction, pos_sample

neg_samples = get_neg_samples(num_neg_samples, contrastive_pred)
neg_prediction = prediction.unsqueeze(2).repeat(1,1,num_neg_samples,1,1)
neg_pair = neg_prediction , neg_samples



cos = torch.nn.CosineSimilarity(dim=3) # channelwise cos similarity
# rho = cos(pos_pair[0], pos_pair[1]) # [ bs x num_patch - 1 x n_vars x embedding_dim ]

target,label = InfoNCE(pos_pair,neg_pair)



