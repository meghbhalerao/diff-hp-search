import torch
from utils.ssl_utils import *

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def hessian_vector_product(model, vector, input, target, weights, r=1e-2):
    R = r / _concat(vector).norm()

    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)

    loss = model._loss(input, target, weights)
    grads_p = torch.autograd.grad(loss,weights)

    for p, v in zip(model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = model._loss(input, target, weights)
    grads_n = torch.autograd.grad(loss, weights)

    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

def hessian_vector_product_ssl(W, W_k, img_ssl_q, img_ssl_k, vector, weights,  r=1e-2):
    R = r / _concat(vector).norm()

    for p, v in zip(W.parameters(), vector):
      p.data.add_(R, v)

    loss = ssl_step(W,W_k, img_ssl_q, img_ssl_k)
    grads_p = torch.autograd.grad(loss,weights)

    for p, v in zip(W.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = ssl_step(W,W_k, img_ssl_q, img_ssl_k)
    
    grads_n = torch.autograd.grad(loss, weights)

    for p, v in zip(W.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

