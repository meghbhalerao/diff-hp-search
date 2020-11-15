import torch
import torch.nn as nn

def ssl_step(W, W_k, img_ssl_q, img_ssl_k, queue = None, queue_ptr = None, K = None, A_ssl = None):
    q = W(img_ssl_q) #query
    k = W_k(img_ssl_k) #k
    k = k.detach()
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= 0.07
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    # Here mask needs to be added
    loss_ssl = nn.CrossEntropyLoss(logits, labels)
    if queue is not None:
        momentum_update_key_encoder(W,W_k, 0.99)
        dequeue_and_enqueue(k, queue, queue_ptr, K)
    return loss_ssl

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
