import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def log_sum_exp(x, axis=None):
    """Log sum exp function
    Args:
        x: Input.
        axis: Axis over which to perform sum.
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y
def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1', 'JSMI', 'BCE']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    elif measure == 'JSMI':
        Ep = - F.softplus(-p_samples)
    elif measure == 'BCE':
        Ep = torch.log(p_samples)
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    elif measure == 'JSMI':
        Eq = F.softplus(q_samples)
    elif measure == 'BCE':
        Eq = -torch.log(1 - q_samples)

    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq



def local_global_loss_(pred, labels, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    positives = pred.reshape(pred.shape[0])*labels
    negatives = pred.reshape(pred.shape[0])*(1-labels)
    pos_nums = torch.sum(labels)
    neg_nums = labels.shape[0] - pos_nums


    E_pos = get_positive_expectation(positives, measure, average=False).sum()
    E_pos = E_pos / pos_nums
    E_neg = get_negative_expectation(negatives, measure, average=False).sum()
    E_neg = E_neg / neg_nums

    return E_neg - E_pos

def global_loss_(g_enc, measure):
    '''
    Args:
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    pos_graphs_num = int(num_graphs/2)
    neg_graphs_num = int(num_graphs/2)

    pos_mask = torch.zeros(num_graphs).cuda()
    pos_mask[0:pos_graphs_num] = 1
    neg_mask = torch.zeros(num_graphs).cuda()
    neg_mask[neg_graphs_num:] = 1

    E_pos = get_positive_expectation(g_enc[0:pos_graphs_num], measure, average=False).sum()
    E_pos = E_pos / pos_graphs_num
    E_neg = get_negative_expectation(g_enc[neg_graphs_num:], measure, average=False).sum()
    E_neg = E_neg / neg_graphs_num

    return E_neg - E_pos


# #%%
# for measure in ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1', 'JSMI', 'BCE']:
#     best = 1000
#     pbest = 0
#     qbest = 0
#     for i in range(101):
#         for j in range(101):
#             en = get_negative_expectation(torch.tensor([i/100]), measure)
#             ep = get_positive_expectation(torch.tensor([j/100]), measure)
#             if en - ep < best:
#                 best = en - ep
#                 pbest = j/100,
#                 qbest = i/100
#     print('measure is : ', measure, 'which loss is: ', best ,' with the minimum loss when p = ', pbest, ' and q = ', qbest)
