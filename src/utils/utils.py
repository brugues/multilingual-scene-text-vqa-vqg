# -*- coding: utf-8 -*-
import sys


def update_progress_bar(progress_bar, epoch, total_epoch, current_batch, total_batch, loss):
    progress_bar.set_postfix(
        epoch='{}/{}'.format(epoch + 1, total_epoch),
        batch='{}/{}'.format(current_batch + 1, total_batch),
        loss='{:.5f}'.format(float(loss)))

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


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
