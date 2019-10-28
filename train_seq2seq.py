import logging
import torch
import torch.nn.functional as F

loss_i = 0
def custom_loss(output, target):
    n_element = output.numel()

    # mse
    mse_loss = F.mse_loss(output, target)  # ~0.1

    # continuous motion
    diff = [abs(output[:, n, :] - output[:, n-1, :]) for n in range(1, output.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    cont_loss /= 100  # ~0.1 -> 0.001

    # motion variance
    norm = torch.norm(output, 2, 1)  # output shape (64, 30, 10)
    var_loss = -torch.sum(norm) / n_element
    var_loss /= 1  # ~0.1 -> 0.1

    # final loss
    loss = mse_loss + cont_loss + var_loss

    # debugging code
    global loss_i
    if loss_i == 1000:
        logging.debug('(my loss) mse %.5f, cont %.5f, var %.5f' % (mse_loss, cont_loss, var_loss))
        loss_i = 0
    loss_i += 1

    return loss


def train_iter_seq2seq(args, in_text, in_lengths, target_poses, net, optim, loss_fn):
    # zero gradients
    optim.zero_grad()

    # generation
    outputs = net(in_text, in_lengths, target_poses, None)

    # loss
    # loss = loss_fn(outputs, target_poses)
    loss = custom_loss(outputs, target_poses)
    loss.backward()

    # optimize
    optim.step()

    return {'loss': loss.item()}

