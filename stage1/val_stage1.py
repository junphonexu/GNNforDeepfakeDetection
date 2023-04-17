import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.ANFL import MEFARG
from dataset_stage1 import MyData
from utils import *
from conf import get_config, set_logger, set_outdir, set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    valset = MyData(train=False)
    val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)


# Val
def val(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            update_list = statistics(outputs, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list


def main(conf):
    dataset_info = infolist

    start_epoch = 0
    # data
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    # train_weight = torch.from_numpy(np.loadtxt(os.path.join(conf.dataset_path, 'list', conf.dataset+'_weight_fold'+str(conf.fold)+'.txt')))

    # logging.info("Fold: [val_data_num: {} ]".format(val_data_num))

    net = MEFARG(num_classes=4, metric=conf.metric, backbone=conf.arc)
    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        # train_weight = train_weight.cuda()

    criterion = WeightedAsymmetricLoss()
    optimizer = optim.AdamW(net.parameters(),  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    #train and val
    for epoch in range(start_epoch, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc = val(net, val_loader, criterion)

        # log
        infostr = {'Epoch:  {} val_loss: {:.5f}  val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                .format(epoch + 1, val_loss, 100. * val_mean_f1_score, 100. * val_mean_acc)}

        logging.info(infostr)
        infostr = {'F1-score-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_f1_score)
        logging.info(infostr)
        infostr = {'Acc-list:'}
        logging.info(infostr)
        infostr = dataset_info(val_acc)
        logging.info(infostr)



# ---------------------------------------------------------------------------------


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

