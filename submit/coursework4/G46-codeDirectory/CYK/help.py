#!/usr/bin/env python

# encoding: utf-8

                      
import os

def run_tensorboard(log_dir):
    # tensorboard --logdir=logs
    os.system('tensorboard --logdir={}'.format(log_dir))



