import torch
import torch.autograd as autograd

def contractive_penalty(self, network, inputs, penalty_coefficient):

    if penalty_coefficient == 0.:
        return

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    inputs = [inp.detach() for inp in inputs]
    # inputs that require gradient should be be 1 for computing the Frobenius norm of the Jacobian
    # inputs that do not require gradient should be 0
    inputs = [inp.requires_grad_() for inp in inputs]

    with torch.set_grad_enabled(True):
        output = network(*inputs)
    gradient = autograd.grad(output, inputs, 
                             torch.ones_like(output),
                             create_graph=True, retain_graph=True,
                             only_inputs=True, allow_unused=True)[0]
    gradient = gradient.view(gradient.size()[0], -1)
    penalty = (gradient ** 2).sum(1).mean()

    return penalty_coefficient * penalty
