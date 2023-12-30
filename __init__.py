#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 17:34:27 2022

@author: system01
"""

from . import settings
from . import data
from . import mlbase
from . import util
import os
import yaml


def set_env():
    cred_path = os.path.join(os.path.dirname(__file__), 'cred.yaml')
    if os.path.exists(cred_path):
        with open(cred_path, 'r') as f:
            cred = yaml.safe_load(f)
    else:
        cred = {"slack_token": "", "line_notify_token": ""}
        print('cred.yaml is not found. Please create cred.yaml or set environment variables.')
    
    os.environ["SLACK_TOKEN"] = cred["slack_token"]
    os.environ["LINE_TOKEN"] = cred["line_notify_token"]

set_env()