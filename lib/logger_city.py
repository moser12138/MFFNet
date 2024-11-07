#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import logging

import torch.distributed as dist


def setup_logger(name, logpth):
    logfile = '{}-{}.log'.format(name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank() != 0:
        log_level = logging.WARNING
    try:
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile, force=True)
    except Exception:
        for hl in logging.root.handlers: logging.root.removeHandler(hl)
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def print_log_msg(it, max_iter, lr, time_meter, loss_meter, loss_main_meter, loss_aux_meter):
    t_intv, eta = time_meter.get()
    loss_all, _ = loss_meter.get()
    loss_main, _ = loss_main_meter.get()
    loss_aux, _ = loss_aux_meter.get()
    msg = ', '.join([
        'iter: {it}/{max_it}',
        'lr: {lr:4f}',
        'eta: {eta}',
        'time: {time:.2f}',
        'loss: {loss:.4f}',
        'loss_main:{loss_main:.4f}',
        'loss_aux:{loss_aux:.4f}',
    ]).format(
        it=it+1,
        max_it=max_iter,
        lr=lr,
        time=t_intv,
        eta=eta,
        loss=loss_all,
        loss_main=loss_main,
        loss_aux=loss_aux,
        )
    logger = logging.getLogger()
    logger.info(msg)
