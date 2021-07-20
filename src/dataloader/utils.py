# -*- coding: utf-8 -*-
import sys


def update_train_progress_bar(progress_bar, epoch, total_epoch, current_batch, total_batch, loss):
    progress_bar.set_postfix(
        epoch='{}/{}'.format(epoch + 1, total_epoch),
        batch='{}/{}'.format(current_batch + 1, total_batch),
        loss='{:.5f}'.format(float(loss)))

    progress_bar.update(1)


def update_eval_progress_bar(progress_bar, current_batch, total_batch):
    progress_bar.set_postfix(
        batch='{}/{}'.format(current_batch + 1, total_batch))

    progress_bar.update(1)


class tcolors:
    INFO = '\033[94m'
    ERROR = '\033[91m'
    OK = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def print_info(msg):
    sys.stdout.write(tcolors.INFO + msg + tcolors.ENDC)
    sys.stdout.flush()


def print_err(msg):
    sys.stdout.write(tcolors.ERROR + msg + tcolors.ENDC)
    sys.stdout.flush()


def print_ok(msg):
    sys.stdout.write(tcolors.OK + msg + tcolors.ENDC)
    sys.stdout.flush()


def print_warn(msg):
    sys.stdout.write(tcolors.WARNING + msg + tcolors.ENDC)
    sys.stdout.flush()


def print_progress(progress, msg=''):
    sys.stdout.write('\r')
    sys.stdout.write(msg + " [%-20s] %d%%" % ('=' * int(progress / 5), progress))
    sys.stdout.flush()
