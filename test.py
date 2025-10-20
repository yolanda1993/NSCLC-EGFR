import os
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

import datagenerators
import networks

# matplotlib.use("TkAgg")


def test(train_data_dir,
         test_data_dir,
         device,
         load_model):
    train_samples = os.listdir(train_data_dir)
    test_samples = os.listdir(test_data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = networks.Muti_Task_Seg()
    print('loading', load_model)
    state_dict = torch.load(load_model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    EGFR_Truths_train = []
    EGFR_Preds_train = []
    Survival_Truths_train = []
    Survival_Preds_train = []
    Survival_Time_train = []
    for train_sample in train_samples:
        train_sample = os.path.join(train_data_dir, train_sample)
        PET, CT, Seg, label, Clinic, Text = datagenerators.load_example_by_name(train_sample)
        Text = Text.astype(np.float32)
        PET = np.transpose(PET, (0, 4, 1, 2, 3))
        CT = np.transpose(CT, (0, 4, 1, 2, 3))
        PET = torch.from_numpy(PET).to(device).float()
        CT = torch.from_numpy(CT).to(device).float()
        Clinic = torch.from_numpy(Clinic).to(device).float()
        Text = torch.from_numpy(Text).to(device).float()
        EGFR_Truths_train.append(label[1])
        Survival_Truths_train.append(label[2])
        Survival_Time_train.append(label[0])

        with torch.no_grad():
            # pred = model(PET, CT)
            pred = model(PET, CT, Clinic, Text)
            EGFR_Preds_train.append(pred[1])
            Survival_Preds_train.append(pred[2])

    EGFR_Truths_test = []
    EGFR_Preds_test = []
    Survival_Truths_test = []
    Survival_Preds_test = []
    Survival_Time_test = []
    for test_sample in test_samples:
        test_sample = os.path.join(test_data_dir, test_sample)
        PET, CT, Seg, label, Clinic, Text = datagenerators.load_example_by_name(test_sample)
        Text = Text.astype(np.float32)
        PET = np.transpose(PET, (0, 4, 1, 2, 3))
        CT = np.transpose(CT, (0, 4, 1, 2, 3))
        PET = torch.from_numpy(PET).float().to(device)
        CT = torch.from_numpy(CT).float().to(device)
        Clinic = torch.from_numpy(Clinic).float().to(device)
        Text = torch.from_numpy(Text).float().to(device)
        EGFR_Truths_test.append(label[1])
        Survival_Truths_test.append(label[2])
        Survival_Time_test.append(label[0])

        with torch.no_grad():
            # pred = model(PET, CT)
            pred = model(PET, CT, Clinic, Text)
            EGFR_Preds_test.append(pred[1])
            Survival_Preds_test.append(pred[2])

    EGFR_Truths_train = np.array(EGFR_Truths_train)
    EGFR_Preds_train = np.array([EGFR_Pred.item() for EGFR_Pred in EGFR_Preds_train])
    Survival_Preds_train = np.array([Survival_Pred.item() for Survival_Pred in Survival_Preds_train])
    Survival_Truths_train = np.array(Survival_Truths_train)
    threshold = 0.5  # 设定阈值
    y_preds_train = np.array([1 if prob >= threshold else 0 for prob in EGFR_Preds_train])
    y_trues_train = EGFR_Truths_train
    cm1 = confusion_matrix(y_trues_train, y_preds_train)
    print(cm1)
    accuracy_train = accuracy_score(y_preds_train, y_trues_train)
    cindex_train = concordance_index(Survival_Time_train, -Survival_Preds_train, Survival_Truths_train)
    print("cindex_train:", cindex_train)
    print("accuracy_train:", accuracy_train)

    fpr1, tpr1, thresholds = roc_curve(y_true=EGFR_Truths_train, y_score=EGFR_Preds_train)
    fpr2, tpr2, thresholds2 = roc_curve(y_true=EGFR_Truths_train, y_score=EGFR_Preds_train)

    roc_auc1 = roc_auc_score(EGFR_Truths_train, EGFR_Preds_train)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(fpr1, tpr1, label=f'Train-ROC curve (area = {roc_auc1:.2f})')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')

    # axes[1].plot(fpr2, tpr2, label=f'Train-ROC curve (area = {roc_auc2:.2f})')
    # axes[1].set_xlabel('False Positive Rate')
    # axes[1].set_ylabel('True Positive Rate')
    # axes[1].legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=[0, 1])
    disp.plot(
        include_values=True,  # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",  # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,  # 同上
        xticks_rotation="horizontal",  # 同上
        values_format="d"  # 显示的数值格式
    )

    EGFR_Truths_test = np.array(EGFR_Truths_test)
    EGFR_Preds_test = np.array([EGFR_Pred.item() for EGFR_Pred in EGFR_Preds_test])
    Survival_Preds_test = np.array([Survival_Pred.item() for Survival_Pred in Survival_Preds_test])
    threshold = 0.5  # 设定阈值
    y_preds_test = np.array([1 if prob >= threshold else 0 for prob in EGFR_Preds_test])
    y_trues_test = EGFR_Truths_test
    cm2 = confusion_matrix(y_trues_test, y_preds_test)
    print(cm2)
    accuracy_test = accuracy_score(y_trues_test, y_preds_test)
    cindex_test = concordance_index(Survival_Time_test, -Survival_Preds_test, Survival_Truths_test)
    print("cindex_test:", cindex_test)
    print("accuracy_test:", accuracy_test)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=[0, 1])
    disp.plot(
        include_values=True,  # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",  # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,  # 同上
        xticks_rotation="horizontal",  # 同上
        values_format="d"  # 显示的数值格式
    )

    fpr3, tpr3, thresholds = roc_curve(y_true=EGFR_Truths_test, y_score=EGFR_Preds_test)
    roc_auc1 = roc_auc_score(EGFR_Truths_test, EGFR_Preds_test)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(fpr3, tpr3, label=f'Test-ROC curve (area = {roc_auc1:.2f})')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(loc='lower right')

    # axes[1].plot(fpr4, tpr4, label=f'Test-ROC curve (area = {roc_auc2:.2f})')
    # axes[1].set_xlabel('False Positive Rate')
    # axes[1].set_ylabel('True Positive Rate')
    # axes[1].legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_data_dir", type=str,
                        dest="train_data_dir",
                        default=r'/',
                        help="data folder")
    parser.add_argument("--test_data_dir", type=str,
                        dest="test_data_dir",
                        default=r'/',
                        help="data folder")
    parser.add_argument("--device", type=str, default='/',
                        dest="device", help="cpu or gpuN")
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default='/',
                        help="load best model")

    args = parser.parse_args()
    test(**vars(args))
