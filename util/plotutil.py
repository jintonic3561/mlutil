#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:10:51 2022

@author: system01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def plot_time_series(df, title='history', savefig=False, savedir=None):
    # default: (6.0, 4.0)
    figsize = (6.0, max(len(df.columns) * 2, 4.0))
    fig, ax = plt.subplots(len(df.columns), 1, sharex='all', figsize=figsize)
    if len(df.columns) > 1:
        for index, column in enumerate(df.columns):
            ax[index].plot(df[column])
            ax[index].set_title(column)
    else:
        ax.plot(df.iloc[:, 0])
        ax.set_title(df.columns[0])
    
    fig.tight_layout()
    fig.align_labels(ax)
    plt.xticks(rotation=45)
    _show(plt, savefig=savefig, savedir=savedir, title=title)

def scatter(x, y, x_name=None, y_name=None, title=None, text=None, arrange_axis=True, savefig=False, savedir=None):
    corr_coef = round(_calc_corr_coef(x, y), 2)
    linear_model = _linear_regression(x, y)
    x_label = _get_title_name(x, x_name)
    y_label = _get_title_name(y, y_name)
    title = title if title else f'{x_label} vs {y_label}'
    text = text if text else f'r = {corr_coef}'
    plt.figure(figsize=[5.0,5.0])
    if linear_model:
        linear_x = np.linspace(start=x.min(), stop=x.max(), num=100).reshape(-1, 1)
        linear_y = linear_model.predict(linear_x)
        plt.plot(linear_x, linear_y)
    plt.scatter(x, y)
    plt.axhline(y=0, color='k', linewidth=1)
    plt.axvline(x=0, color='k', linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if arrange_axis:
        plt.xlim(min(x.min(), y.min()) - 0.001, max(x.max(), y.max()) + 0.001)
        plt.ylim(min(x.min(), y.min()) - 0.001, max(x.max(), y.max()) + 0.001)
        plt.text(x=max(x.max(), y.max()), 
                  y=min(x.min(), y.min()), 
                  s=text,
                  horizontalalignment='right',
                  verticalalignment='bottom')
    else:
        plt.text(x=x.max(), 
                 y=y.min(), 
                 s=text,
                 horizontalalignment='right',
                 verticalalignment='bottom')
    plt.tight_layout()
    _show(plt, savefig=savefig, savedir=savedir, title=title)

def hist(x, x_name=None, savefig=False, savedir=None):
    title = _get_title_name(x, x_name)
    plt.hist(x, bins=50)
    plt.title(title)
    _show(plt, savefig=savefig, savedir=savedir, title=title)
    
def bar(value_counts, title=None, savefig=False, savedir=None):
    plt.title(title)
    plt.bar(range(len(value_counts)),
            value_counts,
            tick_label=value_counts.index,
            align="center")
    _show(plt, savefig=savefig, savedir=savedir, title=title)

def boxplot(*args, title=None, savefig=False, savedir=None):
    plt.title(title)
    plt.boxplot(args)
    _show(plt, savefig=savefig, savedir=savedir, title=title)


def _get_title_name(x, x_name):
    if x_name:
        return x_name
    elif isinstance(x, pd.Series):
        return x.name
    else:
        return None

def _show(plt, savefig=False, savedir=None, title=None):
    if savefig:
        path = savedir if savedir else '.'
        path += '/'
        path += title.replace(' ', '_') if title else 'result'
        path += '.png'
        plt.savefig(path)
    
    plt.show()

def _linear_regression(x, y, dropna=True):
    model = LinearRegression()
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    if dropna:
        df = df.dropna(how='any')
        if len(df) == 0:
            return None
    elif df.isnull().sum().sum() > 0:
        return None
    
    model.fit(df[['x']], df['y'])
    return model

def _calc_corr_coef(x, y, dropna=True):
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    if dropna:
        df = df.dropna(how='any')
    elif df.isnull().sum().sum() > 0:
        return np.nan
    
    if len(df['x'].unique()) <= 1 or len(df['y'].unique()) <= 1:
        return np.nan
    else:
        return np.corrcoef(df['x'], df['y'])[0][1]









