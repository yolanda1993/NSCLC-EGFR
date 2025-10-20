import torch


def Dice_loss(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    # 获取除了batch维度之外的空间维度数量，假设输入形状为(batch_size, *spatial_dims)
    ndims = y_pred.ndim - 2
    vol_axes = list(range(2, ndims + 2))

    top = 2 * torch.sum(y_true * y_pred, dim=vol_axes)
    bottom = torch.maximum(torch.sum(y_true + y_pred, dim=vol_axes), torch.tensor(1e-5, dtype=y_true.dtype))
    dice = torch.mean(top / bottom)

    return -dice


def BinaryCross_Loss(y_ture, y_pred):
    y_ture = torch.unsqueeze(y_ture, 1)
    BC_loss = torch.nn.BCELoss()(y_pred, y_ture)

    return BC_loss


def Cox_loss(y_true, y_pred):
    '''
    Calculate the average Cox negative partial log-likelihood.
    y_pred is the predicted risk from trained model.
    y_true is event indicator, event=0 means censored
    Survival time is not requied as input
    Samples should be sorted with increasing survial time
    '''

    risk = y_pred
    event = y_true.unsqueeze(dim=1)
    # event = torch.cast(y_true, dtype=risk.dtype)

    risk_exp = torch.exp(risk)
    risk_exp_cumsum = torch.cumsum(torch.flip(risk_exp, dims=[0]), dim=0)
    likelihood = risk - torch.log(risk_exp_cumsum)
    uncensored_likelihood = torch.multiply(likelihood, event)
    n_observed = torch.sum(event)
    cox_loss = -torch.sum(uncensored_likelihood) / n_observed

    return cox_loss
