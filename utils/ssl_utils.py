import torch
import torch.nn as nn

def ssl_step(W, W_k, optimizer, img_ssl_q, img_ssl_k, queue, queue_ptr, K):
    q = W(img_ssl_q) #query
    k = W_k(img_ssl_k) #k
    k = k.detach()
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= 0.07
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    loss_ssl = nn.CrossEntropyLoss(logits, labels)
    optimizer.step()
    dequeue_and_enqueue(k, queue, queue_ptr, K)

def momentum_update_key_encoder(encoder_q, encoder_k, m):
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

def dequeue_and_enqueue(keys,queue, queue_ptr, K):
    batch_size = keys.shape[0]
    ptr = int(queue_ptr)
    assert K % batch_size == 0  # for simplicity
    queue[:, ptr:ptr + batch_size] = keys.T
    ptr = (ptr + batch_size) % K  # move pointer
    queue_ptr[0] = ptr
