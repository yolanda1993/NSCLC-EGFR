import csv
import glob
import os
import random
import time
from argparse import ArgumentParser

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import datagenerators
import networks
import losses
import metrics


def lr_scheduler(epoch):
    if epoch < 150:
        lr = 5e-6
    elif epoch < 200:
        lr = 1e-6
    elif epoch < 250:
        lr = 5e-7
    else:
        lr = 1e-7
    print('lr: %f' % lr)
    return lr


def train(train_dir,
          valid_dir,
          model_dir,
          device,
          lr,
          nb_epochs,
          steps_per_epoch,
          validation_steps,
          batch_size,
          validation_size,
          load_model_file,
          initial_epoch):
    # prepare data files
    train_vol_names = glob.glob(os.path.join(train_dir, '*.npz'))
    assert len(train_vol_names) > 0, "Could not find any training data"
    random.shuffle(train_vol_names)  # shuffle volume list

    valid_vol_names = glob.glob(os.path.join(valid_dir, '*.npz'))
    assert len(valid_vol_names) > 0, "Could not find any validation data"
    random.shuffle(valid_vol_names)  # shuffle volume list

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # device handling
    # if 'gpu' in device:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
    #     device = 'cuda'
    #     torch.backends.cudnn.deterministic = True
    # else:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #     device = 'cpu'

    model = networks.Muti_Task_SegMuti_Task_Seg()
    if load_model_file != './':
        print('loading', load_model_file)
        state_dict = torch.load(load_model_file, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters())
    Losses = [losses.Dice_loss, losses.BinaryCross_Loss, losses.Cox_loss]
    Weights = [1.0, 2.0, 2.0]

    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size=batch_size, balance_class=True)
    data_gen_train = datagenerators.gen_train(train_example_gen, Sort=True)

    valid_example_gen = datagenerators.example_gen(valid_vol_names, batch_size=validation_size, balance_class=True)
    data_gen_valid = datagenerators.gen_valid(valid_example_gen, Sort=True)

    # training/validation loops

    iteration = 0
    for epoch in range(initial_epoch, nb_epochs):
        torch.cuda.empty_cache()
        start_time = time.time()
        # adjust lr
        lr = lr_scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(f"Epoch---{epoch + 1}/{nb_epochs}")
            # training
            model.train()
            train_losses = []
            train_total_loss = []
            train_dices = []
            train_aucs = []
            train_survivals = []
            for step in range(steps_per_epoch):
                iteration_start_time = time.time()
                iteration += 1
                # generate inputs (and true outputs) and convert them to tensors

                inputs, labels = next(data_gen_train)
                inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
                labels = [torch.from_numpy(d).to(device).float() for d in labels]

                # run inputs through the model to produce a warped image and flow field
                pred = model(*inputs)
                Seg_Tumor = labels[0]
                EGFR = torch.unsqueeze(labels[1], 1)
                Survival = labels[2]
                # Seg_Tumor_pred = pred[0].detach().cpu().numpy().squeeze()
                # Survival_pred = pred[1].detach().cpu().numpy().squeeze()
                Seg_Tumor_pred = pred[0]
                EGFR_pred = pred[1]
                Survival_pred = pred[2]
                Dice_Tumor = metrics.Dice(Seg_Tumor, Seg_Tumor_pred)
                Auc = metrics.Auc(EGFR, EGFR_pred)
                Cindex = metrics.Cindex(Survival, Survival_pred)
                train_dices.append(Dice_Tumor.cpu().detach().numpy())
                train_aucs.append(Auc)
                train_survivals.append(Cindex.cpu().detach().numpy())
                # calculate total loss

                loss = 0
                loss_list = []
                for i, Loss in enumerate(Losses):
                    curr_loss = Loss(labels[i], pred[i]).to(device) * Weights[i]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss
                train_losses.append(loss_list)
                train_total_loss.append(loss.item())

                # backpropagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iteration_time = time.time() - iteration_start_time
                data_row = [iteration, iteration_time, loss, loss_list[0], loss_list[1], loss_list[2], Dice_Tumor, Auc,
                            Cindex]
                with open(os.path.join(model_dir, 'training_results.csv'), 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if iteration == 1:
                        # 在第一次循环时写入表头
                        headers = ["iteration", "iteration_time", "Loss", "Dice_loss", "BinaryCross_Loss",
                                   "C-index_loss", "Dice_tumor", "Auc", "Cindex"]
                        writer.writerow(headers)
                    writer.writerow(data_row)
                print(
                    f"第{iteration}次---iteration_time:{iteration_time:.2f}---itration-loss:{loss:.4f}--Dice_loss:{loss_list[0]:.4f}--BC_loss:{loss_list[1]:.4f}--Cox_loss:{loss_list[2]:.4f}--Dice:{Dice_Tumor:.4f}--Auc:{Auc:.4f}--Cindex:{Cindex:.4f}")

        # validation
        model.eval()
        valid_losses = []
        valid_total_loss = []
        valid_dices = []
        valid_Aucs = []
        valid_cindexs = []
        for step in range(validation_steps):
            inputs, labels = next(data_gen_valid)
            inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            labels = [torch.from_numpy(d).to(device).float() for d in labels]

            with torch.no_grad():
                pred = model(*inputs)

            Seg_Tumor = labels[0]
            EGFR = torch.unsqueeze(labels[1], 1)
            Survival = labels[2]
            Seg_Tumor_pred = pred[0]
            EGFR_pred = pred[1]
            Survival_pred = pred[2]
            Val_Dice_Tumor = metrics.Dice(Seg_Tumor, Seg_Tumor_pred)
            Val_Auc = metrics.Auc(EGFR, EGFR_pred)
            Val_Cindex = metrics.Cindex(Survival, Survival_pred)
            valid_dices.append(Val_Dice_Tumor.cpu().detach().numpy())
            valid_Aucs.append(Val_Auc)
            valid_cindexs.append(Val_Cindex.cpu().detach().numpy())
            Val_loss = 0
            Val_loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(labels[i], pred[i]).to(device) * Weights[i]
                Val_loss_list.append(curr_loss.item())
                Val_loss += curr_loss
            valid_losses.append(Val_loss_list)
            valid_total_loss.append(Val_loss.item())
        epoch_info = 'Epoch %d/%d' % (epoch + 1, nb_epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)
        valid_losses = ', '.join(['%.4f' % f for f in np.mean(valid_losses, axis=0)])
        val_loss_info = 'Valid loss: %.4f (%s)' % (np.mean(valid_total_loss), valid_losses)
        valid_dice_info = 'Valid Dice: %.4f ' % (np.mean(valid_dices))
        valid_Auc_info = 'Valid Auc: %.4f ' % (np.mean(valid_Aucs))
        valid_cindex_info = 'Valid Cindex: %.4f ' % (np.mean(valid_cindexs))
        print(' - '.join((epoch_info, time_info, val_loss_info, valid_dice_info, valid_Auc_info, valid_cindex_info)),
              flush=True)
        valid_data_row = [lr, epoch, (np.mean(train_total_loss), train_losses),
                          (np.mean(valid_total_loss), valid_losses), (np.mean(train_dices)), (np.mean(train_aucs)),
                          (np.mean(train_survivals)), (np.mean(valid_dices)), (np.mean(valid_Aucs)),
                          (np.mean(valid_cindexs))]
        with open(os.path.join(model_dir, 'valid_results.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if epoch + 1 == 1:
                # 在第一次循环时写入表头
                headers = ["lr", "epochs", "train_loss_info", "val_loss_info", "train_dice_info", "train_auc_info",
                           "train_cindex_info", "valid_dice_info", "valid_auc_info", "valid_cindex_info"]
                writer.writerow(headers)
            writer.writerow(valid_data_row)
        # save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_dir, '%4f--%4f--%4f--%02d.pt' % (np.mean(valid_dices),
                                                                                           np.mean(valid_Aucs),
                                                                                           np.mean(valid_cindexs),
                                                                                           epoch + 1)))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_dir", type=str,
                        dest="train_dir", default=r'/',
                        help="training data folder")
    parser.add_argument("--valid_dir", type=str,
                        dest="valid_dir", default=r'/',
                        help="validation data folder")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default=r'/',
                        help="models folder")
    parser.add_argument("--device", type=str, default='',
                        dest="device",
                        help="gpuN or multi-gpu")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4,
                        help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=0,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=0,
                        help="iterations of each epoch")
    parser.add_argument("--validation_steps", type=int,
                        dest="validation_steps", default=0,
                        help="iterations for validation")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=0,
                        help="batch size")
    parser.add_argument("--validation_size", type=int,
                        dest="validation_size", default=0,
                        help="validation size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default=r'./',
                        help="optional h5 model file to initialize with")

    args = parser.parse_args()
    train(**vars(args))
