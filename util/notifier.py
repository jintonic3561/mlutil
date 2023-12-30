#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:16:21 2022

@author: system01
"""

import os
import asyncio
import requests
import functools
import traceback
from .. import settings as args
from enum import Enum


def slack_notify(message, channel_name):
    messages = _split_message(message)
    for i in messages:
        _ = _post_to_slack(message=i, channel_name=channel_name)


def line_notify(message):
    messages = _split_message(message)
    for i in messages:
        _ = _post_to_line(message=i)


async def async_slack_notify(message, channel_name):
    messages = _split_message(message)
    loop = asyncio.get_event_loop()
    
    for i in messages:
        # キーワード引数をrun_in_executorに渡す工夫
        post = functools.partial(_post_to_slack, message=i, channel_name=channel_name)
        _ = await loop.run_in_executor(None, post)


async def async_line_notify(message):
    messages = _split_message(message)
    loop = asyncio.get_event_loop()
    
    for i in messages:
        # キーワード引数をrun_in_executorに渡す工夫
        post = functools.partial(_post_to_line, message=i)
        _ = await loop.run_in_executor(None, post)


def error_notify(complete_msg, complete_channel_name, error_channel_name):
    def _virtual_wrapper(func):
        def _wrapper(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
                if complete_msg:
                    slack_notify(message=complete_msg, channel_name=complete_channel_name)
                return res
            except:
                message = f'Error has occurred in {func.__name__}:\n{traceback.format_exc()}'
                slack_notify(message=message, channel_name=error_channel_name)
        
        return _wrapper
    return _virtual_wrapper


def _split_message(message, max_length=40000):
    message_list = []
    while True:
        if message[max_length:] == "":
            message_list.append(message)
            break
                
        message_list.append(message[:max_length])
        message = message[max_length:]
    
    return message_list


def _post_to_slack(message, channel_name):
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": "Bearer " + os.environ['SLACK_TOKEN']}
    data  = {'channel': channel_name, 'text': message}
    
    return requests.post(url, data=data, headers=headers)


def _post_to_line(message):
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': 'Bearer ' + os.environ['LINE_TOKEN']}
    data = {'message': message}
    
    return requests.post(url, data=data, headers=headers)


