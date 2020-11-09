import torch

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def hessian_vector_product(model, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)
    loss = model._loss(input, target)
    grads_p = torch.autograd.grad(loss, model.arch_parameters())

    for p, v in zip(model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = model._loss(input, target)
    grads_n = torch.autograd.grad(loss, model.arch_parameters())

    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

