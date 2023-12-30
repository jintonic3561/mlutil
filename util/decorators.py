#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:03:03 2022

@author: system01
"""

import inspect
import time
import traceback
from functools import wraps
try:
    from notifier import slack_notify
except ImportError:
    from .notifier import slack_notify


def time_watcher(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__}: {round(end - start)}s')
        return result
    
    return wrapper


def method_time_watcher(ignore_method_names=[]):
    def wrapper(cls_):
        for name, func in inspect.getmembers(cls_, inspect.isfunction):
            if name.startswith('__'):
                continue
            elif name in ignore_method_names:
                continue
            else:
                setattr(cls_, name, time_watcher(func))
            
        return cls_
    
    return wrapper


def error_notify(func, channel_name):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            message = f'Error has occured in {func.__name__}:\n' 
            message += f'<{e.__class__.__name__}>\n'
            for i in e.args:
                message += f'{i}\n'
            
            slack_notify(message=message, channel_name=channel_name)
            print(f'{message}\n{traceback.format_exc()}')
        else:
            return result
    
    return wrapper
    