import os
import math


import torch
from flash_attn import flash_attn_func
from torch.cuda.amp import autocast
import torch.nn.functional as F

def main():
    torch.manual_seed(1)

    head = 8
    batch = 5
    key = 16
    node = 20
    q = torch.rand((head, batch, 1, 1, key)).cuda()
    k = torch.rand(head, batch, 1, node, key).cuda()
    v = torch.rand(head, batch, 1, node, key).cuda()
    mask = torch.zeros((batch, 1, node)).bool().cuda()
    mask = torch.randint(0, 2, (batch, 1, node), dtype=torch.int).bool().cuda()
    with autocast():
        out1 = heavy_part(q, k, v, mask, flash=True)
        out2 = heavy_part(q, k, v, mask, flash=False)
    print(f'{out1[0, :3, :4, :5]=}\n{out2[0, :3, :4, :5]=}')



def heavy_part(q, k, v, mask, flash):
    """
    q: (head, batch, 1, 1, key)
    k: (head, batch, 1, key, node)
    v: (head, batch, 1, key, node)
    mask: (batch, 1, node)
    """

    embed_dim = q.size(0) * q.size(-1)
    k = k.transpose(-2, -1)  # (head, batch, 1, node, key)

    if flash:
        q = q.squeeze(-2).permute(1, 2, 0, 3).half()  # (batch, 1, head, key)
        k = k.squeeze(2).permute(1, 3, 0, 2).half()  # (batch, node, head, key)
        v = v.squeeze(2).permute(1, 2, 0, 3).half()  # (batch, node, head, key)

        heads = flash_attn_func(q, k, v)  # (batch, 1, head, key)
        heads = heads.permute(2, 0, 1, 3).unsqueeze(2).float()  # (head, batch, node, key)
    else:

        compatibility = torch.matmul(q, k) / math.sqrt(q.size(-1))
        compatibility[mask[None, :, None, :].expand_as(compatibility)] = -math.inf
        compatibility = F.softmax(compatibility, dim=-1)
        heads = torch.matmul(compatibility, v)  # (head, batch, 1, 1, key)

    heads = heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, 1, 1, embed_dim)
    return heads


main()


