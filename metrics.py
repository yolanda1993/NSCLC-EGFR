import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

def Dice(y_true, y_pred):
    """
    Dice score for binary segmentation
    """

    ndims = y_pred.ndim - 2
    vol_axes = list(range(2, ndims + 2))

    y_pred = (y_pred > 0.5).to(y_pred.dtype)

    top = 2 * torch.sum(y_true * y_pred, dim=vol_axes)
    bottom = torch.maximum(torch.sum(y_true + y_pred, dim=vol_axes), torch.tensor(1e-5, dtype=y_true.dtype))
    dice = torch.mean(top / bottom)

    return dice


def Cindex(y_true, y_pred):
    '''
    C-index score for risk prediction.
    y_pred is the predicted risk from trained model.
    y_true is event indicator, event=0 means censored
    Survival time is not requied as input
    Samples should be sorted with increasing survial time
    '''
    risk = y_pred
    event = y_true.unsqueeze(dim=1)
    # event = torch.cast(y_true, risk.dtype)
    g = torch.subtract(risk, risk[:, 0])
    g = torch.where(g == 0.0, torch.tensor(1, dtype=torch.float32),
                    torch.tensor(0, dtype=torch.float32)) * 0.5 + torch.where(g > 0.0,
                                                                              torch.tensor(1, dtype=torch.float32),
                                                                              torch.tensor(0, dtype=torch.float32))
    # g = torch.cast(g == 0.0, risk.dtype) * 0.5 + torch.cast(g > 0.0, risk.dtype)
    f = torch.matmul(event, torch.where(torch.transpose(event, 0, 1) > -1, torch.tensor(1, dtype=torch.float32),
                                      torch.tensor(0, dtype=torch.float32)))
    # f = torch.matmul(event, torch.cast(torch.transpose(event)>-1, risk.dtype))
    f = torch.triu(f) - torch.diag(torch.diag(f))
    top = torch.sum(torch.multiply(g, f))
    bottom = torch.sum(f)
    cindex = top / bottom
    return cindex


def Auc(y_true, y_scores):
    # 将torch的tensor转换为numpy数组，因为sklearn的roc_auc_score函数需要接收numpy数组类型的数据
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_scores.detach().cpu().numpy()
    pair = list(zip(y_true, y_pred))
    pair = sorted(pair, key=lambda x: x[1])
    df = pd.DataFrame([[x[0], x[1], i + 1] for i, x in enumerate(pair)], columns=['y_true', 'y_pred', 'rank'])
    pos_df = df[df.y_true == 1]
    m = pos_df.shape[0]
    n = df.shape[0] - m
    auc_value = (pos_df['rank'].sum() - m * (m + 1) / 2) / (m * n)
    # 使用sklearn的roc_auc_score函数计算AUC
    return auc_value




