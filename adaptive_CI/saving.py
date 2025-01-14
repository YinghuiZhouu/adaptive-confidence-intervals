"""
This script contains helping functions for better saving format. 
"""

import numpy as np
import pandas as pd
import os
import subprocess
from time import time
from os.path import dirname, realpath, join, exists
from os import makedirs, chmod
from getpass import getuser

__all__ = [
    "compose_filename",
    'on_sherlock',
    'get_sherlock_dir'
]


def compose_filename(prefix, extension):
    """
    Creates a unique filename based on Github commit id and time.
    Useful when running in parallel on server.

    INPUT:
        - prefix: file name prefix
        - extension: file extension

    OUTPUT:
        - fname: unique filename
    """
    # Tries to find a commit hash
    try:
        commit = subprocess\
            .check_output(['git', 'rev-parse', '--short', 'HEAD'],
                          stderr=subprocess.DEVNULL)\
            .strip()\
            .decode('ascii')
    except subprocess.CalledProcessError:
        commit = ''

    # Other unique identifiers
    rnd = str(int(time() * 1e8 % 1e8))
    sid = tid = jid = ''
    ident = filter(None, [prefix, commit, jid, sid, tid, rnd])
    basename = "_".join(ident)
    fname = f"{basename}.{extension}"
    return fname

def on_sherlock():
    """ Checks if running locally or on sherlock """
    return 'GROUP_SCRATCH' in os.environ

def get_sherlock_dir(project, *tail, create=True):
    """
    Output consistent folder name in Sherlock.
    If create=True and on Sherlock, also makes folder with group permissions.
    If create=True and not on Sherlock, does not create anything.

    '/scratch/groups/athey/username/project/tail1/tail2/.../tailn'.

    >>> get_sherlock_dir('adaptive-inference')
    '/scratch/groups/athey/adaptive-inference/vitorh'

    >>> get_sherlock_dir('toronto')
    '/scratch/groups/athey/toronto/vitorh/'

    >>> get_sherlock_dir('adaptive-inference', 'experiments', 'exp_out')
    '/scratch/groups/athey/adaptive-inference/vitorh/experiments/exp_out'
    """
    base = join("/", "scratch", "groups", "athey", project, getuser())
    path = join(base, *tail)
    if not exists(path) and create and on_sherlock():
        makedirs(path, exist_ok=True)
        # Correct permissions for the whole directory branch
        chmod_path = base
        chmod(base, 0o775)
        for child in tail:
            chmod_path = join(chmod_path, child)
            chmod(chmod_path, 0o775)
    return path
