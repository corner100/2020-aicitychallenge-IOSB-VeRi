# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
import numpy as np
import wandb
global ITER
ITER = 0
global ITER_EVAL
ITER_EVAL = 0
global EPOCH
EPOCH = 0
global mAP_list
mAP_list = []
global result_liste
result_liste = []

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        model.to(device)
        print('model to device: ',device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target, camids = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target,engine)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def default_score_fn_epoch(engine): # https://github.com/pytorch/ignite/issues/677
    epoch = engine.state.epoch
    return epoch

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        dataset
):
    log_period = cfg['SOLVER.LOG_PERIOD']
    checkpoint_period = cfg['SOLVER.EVAL_PERIOD']
    eval_period = cfg['SOLVER.EVAL_PERIOD']
    output_dir = cfg['OUTPUT_DIR']
    device = cfg['MODEL.DEVICE']
    epochs = cfg['SOLVER.MAX_EPOCHS']

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    if cfg['DATASETS.TRACKS']:
        trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
        evaluator = create_supervised_evaluator(model, metrics={}, device=device)
    else:
        trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
        evaluator = create_supervised_evaluator(model, metrics={}, device=device)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    checkpointer_last_epoch = ModelCheckpoint(dirname=output_dir, filename_prefix=cfg['MODEL.NAME']+'_last_epoch_', score_function=default_score_fn_epoch, n_saved=1, require_empty=False)
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.6f}, Acc: {:.5f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
            wandb.log({'avg_acc': engine.state.metrics['avg_acc'], 'avg_loss': engine.state.metrics['avg_loss'],'lr':scheduler.get_lr()[0]})
        if len(train_loader) == ITER:
            ITER = 0
    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer_last_epoch, {'model': model,'optimizer': optimizer})
    trainer.run(train_loader, max_epochs=epochs)

def create_supervised_trainer_for_regression(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, labels = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels
        # x = x.to(device) if torch.cuda.device_count() >= 1 else x
        # y = y.to(device) if torch.cuda.device_count() >= 1 else y
        # z = z.to(device) if torch.cuda.device_count() >= 1 else z
        score, feat = model(img)
        loss = loss_fn(score.unsqueeze(dim=1), feat, labels,engine)
        loss.backward()
        optimizer.step()
        # compute acc
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator_for_regression(model,loss_fn, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img, labels = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels
            score, feat = model(img)
            loss = loss_fn(score.unsqueeze(dim=1), feat, labels, engine)
            # compute acc
            return score,labels, loss.item()

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
def default_score_fn_regression(engine): # https://github.com/pytorch/ignite/issues/677
    loss = engine.state.metrics['avg_loss']
    return 100/(loss+1) # der checkpointer will, dass die groessere zahl besser ist

def do_regression_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        start_epoch,
        dataset
):
    log_period = cfg['SOLVER.LOG_PERIOD']
    eval_period = cfg['SOLVER.EVAL_PERIOD']
    output_dir = cfg['OUTPUT_DIR']
    device = cfg['MODEL.DEVICE']
    epochs = cfg['SOLVER.MAX_EPOCHS']

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_for_regression(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator_for_regression(model, loss_fn,metrics={}, device=device)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x[2]).attach(evaluator, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[2]).attach(evaluator, 'avg_acc')

    #checkpointer = ModelCheckpoint(dirname=output_dir, filename_prefix=cfg['MODEL.NAME'], save_interval=checkpoint_period, n_saved=10, require_empty=False)
    checkpointer = ModelCheckpoint(dirname=output_dir, filename_prefix=cfg['MODEL.NAME'], score_function=default_score_fn_regression, n_saved=1, require_empty=False)
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.6f}, Acc: {:.6f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
            wandb.log({'avg_acc': engine.state.metrics['avg_acc'], 'avg_loss': engine.state.metrics['avg_loss'],'lr':scheduler.get_lr()[0]})
        if len(train_loader) == ITER:
            ITER = 0
    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        global mAP_list
        global result_liste
        global EPOCH
        EPOCH += 1
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            val_loss = evaluator.state.metrics['avg_loss']

            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("val_loss: {:.6%}".format(val_loss))
            wandb.log({'val_loss': val_loss, 'epoch': EPOCH})

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,'optimizer': optimizer})
    trainer.run(train_loader, max_epochs=epochs)