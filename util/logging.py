#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:16:21 2022

@author: system01
"""

import logging
import csv
import datetime as dt
from enum import Enum


class LogLevel(Enum):
    debug = 'debug'
    info = 'info'
    warn = 'warn'
    error = 'error'


def init_logger(log_file_path):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)-9s  %(asctime)s  [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def write_log(logger, message, level):
    now = dt.datetime.now().strftime("%Y/%m/%d %H%M%S")
    msg = f'{now}: {message}'
    if level == LogLevel.debug:
        logger.debug(msg)
    elif level == LogLevel.info:
        logger.info(msg)
    elif level == LogLevel.warn:
        logger.warn(msg)
    elif level == LogLevel.error:
        logger.error(msg)
    else:
        raise ValueError('Invalid LogLevel {level._value}.')
    

def write_to_csv(data: dict, file_path: str, exist: bool):
    mode = 'a' if exist else 'w'
    with open(file_path, mode, encoding='utf-8') as file:
        if exist:
            writer = csv.writer(file)
            writer.writerow(data.values())
        else:
            field_names = list(data.keys())
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows([data])







