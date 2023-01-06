
import torch
from torch import nn
from torch.nn.functional import normalize, cross_entropy

from .core import Callback

# Cell
class PatchCB(Callback):

    def __init__(self, patch_len, stride ):
        """
        Callback used to perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): self.set_patch()
       
    def set_patch(self):
        """
        take xb from learner and convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb: [bs x seq_len x n_vars]
        # learner get the transformed input
        self.learner.xb = xb_patch                              # xb_patch: [bs x num_patch x n_vars x patch_len]           


class PatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
                        mask_when_pred:bool=False):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio        

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss        
        device = self.learner.device       
 
    def before_forward(self): self.patch_masking()
        
    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor    
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
 
    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        return loss
    
class ContrastivePatchMaskCB(Callback):
    def __init__(self, patch_len, stride, mask_ratio,
                        mask_when_pred:bool=False, 
                        discount=0.99, beta=0.5, tau = 0.01, num_neg_samples = 20):
        """
        Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.
        Args:
            patch_len:        patch length
            stride:           stride
            mask_ratio:       mask ratio
        """
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio   
        self.output_embed=True 
        self.discount = discount
        self.beta = beta # relative weight between reconstruction and contrast
        self.tau = tau # temperature parameter for infoNCE
        self.criterion = cross_entropy
        self.num_neg_samples = num_neg_samples

    def before_fit(self):
        # overwrite the predefined loss function
        self.learner.loss_func = self._loss        
        device = self.learner.device     
 
    def before_forward(self): self.patch_masking()
        
    def patch_masking(self):
        """
        xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(self.xb, self.patch_len, self.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
        xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
        self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
        self.learner.xb = xb_mask       # learner.xb: masked 4D tensor    
        self.learner.yb = xb_patch      # learner.yb: non-masked 4d tensor
 
    def _loss(self, preds, target):        
        """
        preds:   [bs x num_patch x n_vars x embedding_dim],[bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len] 
        """
        reconstructive_preds, contrastive_preds = preds
        loss = (reconstructive_preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.mask).sum() / self.mask.sum()
        
        contrastive_loss = self.contrast(contrastive_preds,target)
        
        return loss + self.beta * contrastive_loss
    
    def get_neg_samples(self,num_neg_samples,contrastive_pred):
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

    def get_pos_sample(self,contrastive_pred):
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

    def InfoNCE(self,pos_pair,neg_pair):
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
        target = target / self.tau

        label = torch.zeros(target.shape[0])
        
        return target, label
    
    def contrast(self,contrastive_pred):


        # positive pair same backbone [bs x num_patch x n_vars x embedding_dim]
        prediction = contrastive_pred[:,:-1, :, :] 
        pos_sample = self.get_pos_sample(contrastive_pred)
        pos_pair = prediction, pos_sample

        neg_samples = self.get_neg_samples(self.num_neg_samples, contrastive_pred)
        neg_prediction = prediction.unsqueeze(2).repeat(1,1,self.num_neg_samples,1,1)
        neg_pair = neg_prediction , neg_samples

        target,label = self.InfoNCE(pos_pair,neg_pair)
        
        loss = self.criterion(target,label)
        return loss     

def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch


class Patch(nn.Module):
    def __init__(self,seq_len, patch_len, stride):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
        tgt_len = patch_len  + stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x):
        """
        x: [bs x seq_len x n_vars]
        """
        x = x[:, self.s_begin:, :]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)                 # xb: [bs x num_patch x n_vars x patch_len]
        return x


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]                                                 # ids_keep: [bs x len_keep]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))        # x_kept: [bs x len_keep x dim]
   
    # removed x
    x_removed = torch.zeros(bs, L-len_keep, D, device=xb.device)                        # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))    # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)                                          # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


if __name__ == "__main__":
    bs, L, nvars, D = 2,20,4,5
    xb = torch.randn(bs, L, nvars, D)
    xb_mask, mask, ids_restore = create_mask(xb, mask_ratio=0.5)
    breakpoint()


