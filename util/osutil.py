# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 07:16:57 2023

@author: jintonic
"""

import os


def go_to_sleep():
    command = """
    PowerShell -Command "Add-Type -Assembly System.Windows.Forms;[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false);"
    """
    os.system(command)