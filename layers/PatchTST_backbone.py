__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
       
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

        self.Linear=nn.Linear(in_features=patch_len, out_features=d_model)
        self.activation=nn.SELU()
        self.normlaizaion=nn.InstanceNorm1d(num_features=self.head_nf)

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)   
                                                         # z: [bs x nvars x patch_len x patch_num]
        z = self.backbone(z)   #z: [bs x nvars x d_model x patch_num]
        z = self.head(z)


                                                                # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        self.pos_enc=0
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        if self.pos_enc:
            u = self.dropout(u + self.W_pos)
            print(1)
        else:
            u = self.dropout(u)
            print(2)                                     # u: [bs * nvars x patch_num x d_model]
        print(stop)
        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head 
        
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn

        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q


        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights

        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


# class PatchTST_backbone_new(nn.Module):
#     def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
#                  n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
#                  d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
#                  padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
#                  pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
#                  pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
#                  verbose:bool=False, **kwargs):
        
#         super().__init__()
       

#         # RevIn
#         self.revin = revin
#         if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
#         # Patching
#         self.patch_len = patch_len
#         self.stride = stride
#         self.padding_patch = padding_patch
#         patch_num = int((context_window*0.5 - patch_len)/stride + 1)

#         if padding_patch == 'end': # can be modified to general case
#             self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
#             patch_num += 1
        
#         # Backbone 
#         # self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
#         #                         n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
#         #                         attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
#         #                         attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
#         #                         pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        


#         # Head
#         # self.head_nf = d_model * patch_num
#         self.head_nf = context_window*(context_window//2+1)
#         self.n_vars = c_in
#         self.pretrain_head = pretrain_head
#         self.head_type = head_type
#         self.individual = individual

#         if self.pretrain_head: 
#             self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
#         elif head_type == 'flatten': 
#             self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

#         # self.Linear=nn.Linear(in_features=patch_len, out_features=d_model)
#         # self.activation=nn.SELU()
#         # self.normlaizaion=nn.InstanceNorm1d(num_features=self.head_nf)

#         sr=context_window
#         ts = 1.0/sr
#         t = np.arange(0,1,ts)
#         t=torch.tensor(t).cuda()
#         for i in range(context_window//2+1):
#             if i==0:
#                 cos=0.5*torch.cos(2*math.pi*i*t).unsqueeze(0)
#                 sin=-0.5*torch.sin(2*math.pi*i*t).unsqueeze(0)
#             else:
#                 cos=torch.vstack([cos,torch.cos(2*math.pi*i*t).unsqueeze(0)])
#                 sin=torch.vstack([sin,-torch.sin(2*math.pi*i*t).unsqueeze(0)]) 

#         # for i in range(context_window//2+1):
#         #     if i==0:
#         #         cos=0.5*torch.cos(((context_window+target_window)/context_window)*2*math.pi*i*t).unsqueeze(0)
#         #         sin=-0.5*torch.sin(((context_window+target_window)/context_window)*2*math.pi*i*t).unsqueeze(0)
#         #     else:
#         #         cos=torch.vstack([cos,torch.cos(((context_window+target_window)/context_window)*2*math.pi*i*t).unsqueeze(0)])
#         #         sin=torch.vstack([sin,-torch.sin(((context_window+target_window)/context_window)*2*math.pi*i*t).unsqueeze(0)]) 

#         self.cos = nn.Parameter(cos, requires_grad=False)
#         self.sin = nn.Parameter(sin, requires_grad=False)

#         # self.weight = nn.Parameter(torch.ones(169).cuda(), requires_grad=True)
#         # self.linear=nn.Linear(context_window,target_window)
#         # self.encoder = TSTEncoder(context_window, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
#         #                            pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
#         self.dropout = nn.Dropout(dropout)

#         # self.linears=nn.Linear(336+991, target_window)
#         # self.head_dropout = nn.Dropout(head_dropout)

#         # self.head_2 = Flatten_Head(self.individual, self.n_vars, 169*336, target_window, head_dropout=head_dropout)
#         # self.inference = Inference(context_window, 169, 169,7,individual)
#         # self.temp=6
#         # self.binaryConcrete=BinaryConcrete(self.temp, 64,c_in)
#         # self.weight_1 = nn.Parameter(torch.randn([7,169]),requires_grad=False)

        
     
#     def common(self,N):
#         out=[]
#         out_2=[]
#         effective=[]
#         a=np.arange(N)
#         for i in range(N//2+1):
#             if N%a[i]==0 and a[i]!=0:
#                 out.append(a[i])
#                 effective.append(N/a[i])
#             else:
#                 out_2.append(a[i])

#         out=np.array(out)
#         out_2=np.array(out_2)

#         out=torch.LongTensor(out).cuda()
#         out_2=torch.LongTensor(out_2).cuda()
#         effective=torch.LongTensor(effective).cuda()
                
#         return out,out_2, effective


#     # def sample(self, mu, log_var):
#     #     std = torch.exp(0.5*log_var)
#     #     eps = torch.randn_like(std)
#     #     return mu + eps*std

#     def sample(self, alpha, temp=None):
#         residual=self.binaryConcrete(alpha)
#         return residual
      
    
#     def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
#         # norm
#         if self.revin: 
#             z = z.permute(0,2,1)
#             z = self.revin_layer(z, 'norm')
#             z = z.permute(0,2,1)

#         norm=z.size()[-1]
#         frequency=rfft(z,axis=-1)
#         X_oneside=frequency/(norm)*2
#         # store=torch.sort(abs(X_oneside[0,0,1:]), dim=-1, descending=True)
#         # indice1=store.indices[store.values>0.2]
#         # mask = (store.values <= 0.2) & (store.values >= 0.1)
#         # indice2=store.indices[mask]
#         # mask2 = (store.values <= 0.1) & (store.values >= 0.05)
#         # indice3=store.indices[mask2]
#         # indice4=store.indices[store.values <= 0.05]
#         # print(torch.sort(abs(X_oneside[0,0,1:]), dim=-1, descending=True))
#         # print(torch.sort(abs(X_oneside[0,1,1:]), dim=-1, descending=True))
#         # print(torch.sum(abs(X_oneside[0,0,1:]), axis=-1))
#         # print(stop)
#         basis_cos=torch.einsum('bkp,pt->bkpt', X_oneside.real, self.cos)
#         basis_sin=torch.einsum('bkp,pt->bkpt', X_oneside.imag, self.sin)
#         # z=torch.cat([basis_cos[:,:,1:,:],basis_sin[:,:,1:,:]],axis=2)
#         # z=basis_cos[:,:,1:,:]+basis_sin[:,:,1:,:]
#         z=basis_cos+basis_sin

#         # z=torch.einsum('bkpt,p->bkt', z, self.weight)
#         # z=self.linear(z)

#         # real=z.detach().cpu().numpy()
#         # pred=z_in.detach().cpu().numpy()
#         # folder_path = './basis/' + 'electricity.csv' + '/'
#         # np.save(folder_path + 'actual', real)
#         # np.save(folder_path + 'pred', pred)
#         # print(step)



#         # alpha=self.linear(self.weight_1)
#         # sample = self.sample(alpha, self.temp)
#         # out=torch.einsum('bkpt,kp->bkpt', z_in, sample)
#         # out=torch.sum(out,axis=-2)

#         # index=[14, 28, 42, 56, 70, 84]*2
#         # index2= [4, 8, 12, 16, 20, 24, 32, 36 ,40 ,44 ,48 ,52]

#         # z_1=torch.sum(z_in[:,:,index,:],axis=-2)
#         # z_2=torch.sum(z_in[:,:,index2,:],axis=-2)


#         # z=z-z_1[:,:,:672]


        

#         # a,b,c=self.common(336)
#         # z_in= torch.index_select(w, 2, a) 
#         # z=z-torch.sum(z_in,axis=-2)  
#         # z_in=torch.cat([z_in,z.unsqueeze(-2)],axis=-2)

#         # data=z
#         # data=data.detach().cpu().numpy()
#         # basis_cos=basis_cos.detach().cpu().numpy()
#         # basis_sin=basis_sin.detach().cpu().numpy()
#         # folder_path = './basis/' + 'electricity.csv' + '/'
#         # np.save(folder_path + 'input', data)
#         # np.save(folder_path + 'basis_cos', basis_cos)
#         # np.save(folder_path + 'basis_sin', basis_sin)
#         # print(stop)
#         # z = self.backbone(z)
#         z = self.head(z)  
#         # for i in range(len(a)):
#         #     if i==0:
#         #         z_new=z_in[:,:,i,:c[i]]
#         #     else:
#         #         z_add=z_in[:,:,i,:c[i]]
#         #         z_new=torch.cat([z_new,z_add],axis=-1)

    
#         # z_out=torch.index_select(z, 2, b) 
#         # z_out=torch.sum(z_out[:,:,1:,:],axis=-2)
#         # z=torch.cat([z_new,z_out],axis=-1)
#         # z=self.linears(z)
#         # z=self.head_dropout(z)
                

#         # w=basis_cos+basis_sin
#         # index=[14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168]
#         # index2=np.arange(168)
#         # index2=index2[::2][:8]
#         # z_1=torch.sum(w[:,:,:14,:],axis=-2)

#         # z_1=torch.sum(w[:,:,index,:],axis=-2)
#         # z_2=torch.sum(w[:,:,index2,:],axis=-2)
#         # z_3=z-z_1-z_2
#         # if self.training:
#         #     noise=0.1*torch.normal(mean=torch.zeros(z_3.size()),std=torch.zeros(z_3.size())).cuda()
#         #     z_1=z_1+noise
#         #     z_2=z_2+noise
#         #     z_3=z_3-2*noise

#         # z_1=z_1.unsqueeze(-2)
#         # z_2=z_2.unsqueeze(-2)
#         # z=z_3
#         # z_add=torch.cat([z_1,z_2],axis=-2) 

#         # a,b,c=self.common(336)
#         # for i in range(len(a)-1):
#         #     if i==0:
#         #         z_1=torch.sum(z[:,:,:a[i],:],axis=-2).unsqueeze(-2)
#         #     else:
#         #         z_add=torch.sum(z[:,:,a[i-1]:a[i],:],axis=-2).unsqueeze(-2)
#         #         z_1=torch.cat([z_1,z_add],axis=-2)
                


#         # z_1=torch.sum(z[:,:,:14,:],axis=-2).unsqueeze(-2)


#         # z_1=torch.sum(z[:,:,:14,:],axis=-2).unsqueeze(-2)
#         # z_2=z[:,:,14,:].unsqueeze(-2)
#         # z_3=torch.sum(z[:,:,14:,:],axis=-2).unsqueeze(-2)

#         # z=torch.cat([z_1,z_2,z_3],axis=-2)



#         # z_in= torch.index_select(z, 2, a) 
#         # z_out=torch.index_select(z, 2, b) 

#         # z_in= torch.index_select(z, 2, a) 
#         # z_out=torch.index_select(z, 2, b) 
#         # # z_out=torch.sum(z_out[:,:,1:,:],axis=-2).unsqueeze(-2)
#         # # z=torch.cat([z_in,z_out],axis=-2)
#         # z=torch.sum(z_out[:,:,1:,:],axis=-2)

#         # z1=torch.sum(z[:,:,indice1,:],axis=2)
#         # z2=torch.sum(z[:,:,indice2,:],axis=2)
#         # z3=torch.sum(z[:,:,indice3,:],axis=2)
#         # z4=torch.sum(z[:,:,indice4,:],axis=2)
#         # z=torch.stack([z1,z2,z3,z4],axis=-1)

#         # # do patching
#         # if self.padding_patch == 'end':
#         #     z = self.padding_patch_layer(z)

#         # z = z.unfold(dimension=-1, size=24, step=24)                   # z: [bs x nvars x patch_num x patch_len]
#         # z=z.permute(0,1,2,4,3) 
#         # z= torch.reshape(z, (z.shape[0],z.shape[1],-1,z.shape[-1]))

#         # z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
#         # z = self.backbone(z)   #z: [bs x nvars x d_model x patch_num]

#         # n_vars = z.shape[1]
#         # z = z.permute(0,1,3,2)  
#         # # z= self.linear(z)
#         # u = torch.reshape(z, (z.shape[0]*z.shape[1],z.shape[2],z.shape[3]))  # z: [bs x nvars x patch_num x patch_len]
#         # u = self.dropout(u)
#         # z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
#         # z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
#         # z = z.permute(0,1,3,2)                                                  # z: [bs x nvars x d_model x patch_num]
#         # z=torch.cat([basis_cos[:,:,0:1,:],z],axis=2)


#         # if self.padding_patch == 'end':
#         #     z = self.padding_patch_layer(z)
#         # z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
#         # z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
#         # z = self.backbone(z)
#         # z = self.head(z) # z: [bs x nvars x target_window
#         # denorm
#         if self.revin: 
#             z = z.permute(0,2,1)
#             z = self.revin_layer(z, 'denorm')
#             z = z.permute(0,2,1)
            
#         return z
    

# class Flatten_Head(nn.Module):
#     def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
#         super().__init__()
        
#         self.individual = individual
#         self.n_vars = n_vars
        
#         if self.individual:
#             self.linears = nn.ModuleList()
#             self.dropouts = nn.ModuleList()
#             self.flattens = nn.ModuleList()
#             for i in range(self.n_vars):
#                 self.flattens.append(nn.Flatten(start_dim=-2))
#                 self.linears.append(nn.Linear(nf, target_window))
#                 self.dropouts.append(nn.Dropout(head_dropout))
#         else:
#             self.flatten = nn.Flatten(start_dim=-2)
#             # self.linear = nn.Sequential(
#             #                                 nn.Linear(nf,720*2),      
#             #                                 nn.Dropout(p=0.15),
#             #                                 nn.ReLU(),
#             #                                 nn.Linear(720*2, 720*2),
#             #                                 nn.Dropout(p=0.15),
#             #                                 nn.ReLU(),
#             #                                 nn.Linear(720*2, target_window)
#             #                             ) 
#             self.linear = nn.Linear(nf, target_window)
#     def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
#         if self.individual:
#             x_out = []
#             for i in range(self.n_vars):
#                 z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
#                 z = self.linears[i](z)                    # z: [bs x target_window]
#                 z = self.dropouts[i](z)
#                 x_out.append(z)
#             x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
#         else:
#             x = self.flatten(x)
#             x = self.linear(x)
#         return x
        


# # class Flatten_Head(nn.Module):
# #     def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
# #         super().__init__()
        
# #         self.individual = individual
# #         self.n_vars = n_vars
        
# #         if self.individual:
# #             self.linears = nn.ModuleList()
# #             self.dropouts = nn.ModuleList()
# #             self.flattens = nn.ModuleList()
# #             for i in range(self.n_vars):
# #                 self.flattens.append(nn.Flatten(start_dim=-2))
# #                 self.linears.append(nn.Linear(nf, target_window))
# #                 self.dropouts.append(nn.Dropout(head_dropout))
# #         else:
# #             self.flatten = nn.Flatten(start_dim=-2)
# #             self.linear = nn.Linear(nf//16, target_window)
# #             self.dropout = nn.Dropout(head_dropout)
# #             self.linear_middle = nn.Linear(nf, nf//16)
# #             self.relu=nn.ReLU()
# #     def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
# #         if self.individual:
# #             x_out = []
# #             for i in range(self.n_vars):
# #                 z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
# #                 z = self.linears[i](z)                    # z: [bs x target_window]
# #                 z = self.dropouts[i](z)
# #                 x_out.append(z)
# #             x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
# #         else:
# #             x = self.flatten(x)
# #             x=self.linear_middle(x)
# #             x = self.dropout(x)
# #             x = self.relu(x)
# #             x = self.linear(x)
# #         return x
        
        
    
    
# # class TSTiEncoder(nn.Module):  #i means channel-independent
# #     def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
# #                  n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
# #                  d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
# #                  key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
# #                  pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
# #         super().__init__()
        
# #         self.patch_num = patch_num
# #         self.patch_len = patch_len
        
# #         # Input encoding
# #         q_len = patch_num
# #         self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
# #         self.seq_len = q_len

# #         # Positional encoding
# #         self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

# #         # Residual dropout
# #         self.dropout = nn.Dropout(dropout)

# #         # Encoder
# #         self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
# #                                    pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
# #     def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
# #         n_vars = x.shape[1]
# #         # Input encoding
# #         x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]

# #         x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

# #         u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
# #         u = self.dropout(u)
# #         # + self.W_pos                                         # u: [bs * nvars x patch_num x d_model]

# #         # Encoder
# #         z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
# #         z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
# #         z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        
# #         return z    
            
            
    
# # # Cell
# # class TSTEncoder(nn.Module):
# #     def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
# #                         norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
# #                         res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
# #         super().__init__()

# #         self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
# #                                                       attn_dropout=attn_dropout, dropout=dropout,
# #                                                       activation=activation, res_attention=res_attention,
# #                                                       pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
# #         self.res_attention = res_attention

# #     def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
# #         output = src
# #         scores = None
# #         if self.res_attention:
# #             for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
# #             return output
# #         else:
# #             for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
# #             return output



# # class TSTEncoderLayer(nn.Module):
# #     def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
# #                  norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
# #         super().__init__()
# #         assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
# #         d_k = d_model // n_heads if d_k is None else d_k
# #         d_v = d_model // n_heads if d_v is None else d_v


# #         # Multi-Head attention
# #         self.res_attention = res_attention
# #         self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

# #         # Add & Norm
# #         self.dropout_attn = nn.Dropout(dropout)
# #         if "batch" in norm.lower():
# #             self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
# #         else:
# #             self.norm_attn = nn.LayerNorm(d_model)

# #         # Position-wise Feed-Forward
# #         self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
# #                                 get_activation_fn(activation),
# #                                 nn.Dropout(dropout),
# #                                 nn.Linear(d_ff, d_model, bias=bias))

# #         # Add & Norm
# #         self.dropout_ffn = nn.Dropout(dropout)
# #         if "batch" in norm.lower():
# #             self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
# #         else:
# #             self.norm_ffn = nn.LayerNorm(d_model)

# #         self.pre_norm = pre_norm
# #         self.store_attn = store_attn


# #     def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

# #         # Multi-Head attention sublayer
# #         if self.pre_norm:
# #             src = self.norm_attn(src)
# #         ## Multi-Head 

# #         if self.res_attention:
# #             src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
# #         else:
# #             src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
# #         if self.store_attn:
# #             self.attn = attn

# #         ## Add & Norm
# #         src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
# #         if not self.pre_norm:
# #             src = self.norm_attn(src)

# #         # Feed-forward sublayer
# #         if self.pre_norm:
# #             src = self.norm_ffn(src)
# #         ## Position-wise Feed-Forward
# #         src2 = self.ff(src)
# #         ## Add & Norm
# #         src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
# #         if not self.pre_norm:
# #             src = self.norm_ffn(src)

# #         if self.res_attention:
# #             return src, scores
# #         else:
# #             return src




# # class _MultiheadAttention(nn.Module):
# #     def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
# #         """Multi Head Attention Layer
# #         Input shape:
# #             Q:       [batch_size (bs) x max_q_len x d_model]
# #             K, V:    [batch_size (bs) x q_len x d_model]
# #             mask:    [q_len x q_len]
# #         """
# #         super().__init__()
# #         d_k = d_model // n_heads if d_k is None else d_k
# #         d_v = d_model // n_heads if d_v is None else d_v

# #         self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

# #         self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
# #         self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
# #         self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

# #         # Scaled Dot-Product Attention (multiple heads)
# #         self.res_attention = res_attention
# #         self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

# #         # Poject output
# #         self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


# #     def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
# #                 key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

# #         bs = Q.size(0)
# #         if K is None: K = Q
# #         if V is None: V = Q


# #         # Linear (+ split in multiple heads)

# #         q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
# #         k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
# #         v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]


# #         # Apply Scaled Dot-Product Attention (multiple heads)
# #         if self.res_attention:
# #             output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
# #         else:
# #             output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
# #         # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

# #         # back to the original inputs dimensions
# #         output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
# #         output = self.to_out(output)

# #         if self.res_attention: return output, attn_weights, attn_scores
# #         else: return output, attn_weights


# # class _ScaledDotProductAttention(nn.Module):
# #     r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
# #     (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
# #     by Lee et al, 2021)"""

# #     def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
# #         super().__init__()
# #         self.attn_dropout = nn.Dropout(attn_dropout)
# #         self.res_attention = res_attention
# #         head_dim = d_model // n_heads
# #         self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
# #         self.lsa = lsa

# #     def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
# #         '''
# #         Input shape:
# #             q               : [bs x n_heads x max_q_len x d_k]
# #             k               : [bs x n_heads x d_k x seq_len]
# #             v               : [bs x n_heads x seq_len x d_v]
# #             prev            : [bs x n_heads x q_len x seq_len]
# #             key_padding_mask: [bs x seq_len]
# #             attn_mask       : [1 x seq_len x seq_len]
# #         Output shape:
# #             output:  [bs x n_heads x q_len x d_v]
# #             attn   : [bs x n_heads x q_len x seq_len]
# #             scores : [bs x n_heads x q_len x seq_len]
# #         '''



# #         # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
# #         attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]
        

# #         # Add pre-softmax attention scores from the previous layer (optional)
# #         if prev is not None: attn_scores = attn_scores + prev

# #         # Attention mask (optional)
# #         if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
# #             if attn_mask.dtype == torch.bool:
# #                 attn_scores.masked_fill_(attn_mask, -np.inf)
# #             else:
# #                 attn_scores += attn_mask

# #         # Key padding mask (optional)
# #         if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
# #             attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

# #         # normalize the attention weights
# #         attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
# #         attn_weights = self.attn_dropout(attn_weights)

# #         # compute the new values given the attention weights
# #         output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]
# #         if self.res_attention: return output, attn_weights, attn_scores
# #         else: return output, attn_weights
