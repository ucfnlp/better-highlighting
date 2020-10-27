import datetime
import os
import time
import numpy as np
import pytz
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
import torch.optim.lr_scheduler as lr_scheduler
from transformers import AdamW, WarmupLinearSchedule

from utils import save_model


def show_epoch_info(logger, info, meta_info):
    logger.tensorboard_log(info, meta_info['epoch'])


def show_iter_info(logger, meta_info, iter_info, log_interval, progress_bar=None):
    if meta_info['iter'] % log_interval == 0:
        logger.tensorboard_log(iter_info, meta_info['iter'])
    if progress_bar is not None:
        progress_bar.set_postfix(loss=f'{iter_info["loss"]:05.3f}')
        progress_bar.update()


def show_eval_info(logger, eval_data):
    eval_info, predictions, labels, classes, meta_info = eval_data
    for k, v in eval_info.items():
        logger.write(f'EvalInfo\t{k}: {v}')
    logger.tensorboard_log(eval_info, meta_info['epoch'])
    logger.tensorboard_confusion_matrix(ground_truth=labels,
                                        prediction=predictions,
                                        classes=classes,
                                        global_step=meta_info['epoch'])


def train(model, train_loader, eval_loader, num_epochs, output, logger, optim=None, s_epoch=0, batch_size=128, device='cuda', n_gpu=1, lr_init=5e-5):
    # steps
    num_train_data = len(train_loader.dataset)
    num_total_steps = (num_train_data // batch_size) * (num_epochs - s_epoch)

    # optimizer
    if optim is None:
        optim = AdamW(model.parameters(), lr=lr_init, correct_bias=False)
        optim_name = 'AdamW'
        num_warmup_steps = int(num_total_steps * .1)
    else:
        # optim from previously trained model
        optim_name = 'AdamW'
        num_warmup_steps = 0

    # scheduler
    is_step_scheduler = True
    if is_step_scheduler:
        scheduler = WarmupLinearSchedule(
            optim, warmup_steps=num_warmup_steps, t_total=num_total_steps)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim, mode='max', factor=0.5, patience=3)

    logger.write('optim: %s lr_default=%.4f, num_total_steps=%d, num_warmup_steps=%.2f' %
                 (optim_name, lr_init, num_total_steps, num_warmup_steps))

    iter_info = {}
    epoch_info = {}
    meta_info = dict(epoch=0, iter=0)
    log_interval = 20

    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        total_norm = 0
        count_norm = 0
        st_time = time.time()
        meta_info['epoch'] = epoch

        with tqdm(total=num_train_data) as tbar:
            for i, (input_ids, input_mask, segment_ids, labels) in enumerate(train_loader):
                batch_size_cur = input_ids.size(0)
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()
                labels = labels.cuda()

                outputs = model(input_ids, input_mask,
                                segment_ids, labels=labels)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean()

                optim.zero_grad()
                loss.backward()

                total_norm += 0
                count_norm += 1
                optim.step()
                if is_step_scheduler:
                    # WarmupLinearSchedule
                    scheduler.step()

                total_loss += loss.item()

                # statistics
                iter_info['loss'] = loss.item()
                iter_info['learning rate'] = optim.param_groups[0]["lr"]
                show_iter_info(logger, meta_info, iter_info, log_interval)
                meta_info['iter'] += 1

                tbar.update(batch_size_cur)

        total_loss /= count_norm

        if eval_loader is not None:
            model.train(False)
            eval_scores = evaluate(
                model, eval_loader, logger, meta_info, device=device, n_gpu=n_gpu)
            model.train(True)

        if not is_step_scheduler:
            # ReduceLROnPlateau
            scheduler.step(eval_scores)

        logger.write('epoch %d, time: %.3f' % (epoch, time.time() - st_time))
        logger.write('\tlr: %.3E' % optim.param_groups[0]['lr'])
        logger.write('\tcurrent time: {}'.format(
            datetime.datetime.now(pytz.timezone('US/Eastern')).time()))
        logger.write('\ttrain_loss: %.3E, norm: %.3E' %
                     (total_loss, total_norm / count_norm))
        epoch_info['mean_train_loss'] = total_loss
        show_epoch_info(logger, epoch_info, meta_info)

        if eval_loader is not None:
            logger.write('\taccuracy top-1: %.3f\n' % (eval_scores))

        # save all models at the end of each epoch
        model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
        save_model(model_path, model, epoch, optim)


@torch.no_grad()
def evaluate(model, dataloader, logger=None, meta_info=None, topk=[1], device='cuda', n_gpu=1):
    eval_info = dict()
    loss_value, label_list, pred_list = [], [], []
    nb_eval_steps = 0
    with tqdm(total=len(dataloader.dataset)) as tbar:
        for i, (input_ids, input_mask, segment_ids, labels) in enumerate(dataloader):
            batch_size_cur = input_ids.size(0)
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            labels = labels.cuda()

            outputs = model(input_ids, input_mask, segment_ids, labels=labels)
            loss = outputs[0]
            pred_score = outputs[1]
            if n_gpu > 1:
                loss = loss.mean()

            # statistics
            loss_value.append(loss.item())
            label_list.append(labels.to('cpu').numpy())
            pred_list.append(pred_score.detach().cpu().numpy())

            tbar.update(batch_size_cur)
            nb_eval_steps += 1

        # evaluation
        prediction_output = np.concatenate(pred_list)
        predictions = np.array(prediction_output.ravel() >= 0.5)
        labels = np.concatenate(label_list)
        acc = accuracy_score(labels, predictions)

        classes = np.unique(labels)
        eval_info['mean_loss_test'] = np.sum(loss_value) / nb_eval_steps

        if meta_info is None:
            meta_info = dict(epoch=0)
        eval_data = (eval_info, predictions, labels, classes, meta_info)
        show_eval_info(logger, eval_data)

    return acc
