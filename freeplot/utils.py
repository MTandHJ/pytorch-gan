



import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import json
from .config import cfg
from collections.abc import Iterable



for group, params in cfg['rc_params'].items():
    plt.rc(group, **params)



_ROOT = cfg['root']


def load(filename):
    with open(filename, encoding="utf-8") as j:
        data = json.load(j)
    return data


def style_env(style):
    def decorator(func):
        def wrapper(*arg, **kwargs):
            with plt.style.context(style + cfg.default_style, after_reset=cfg.reset):
                results = func(*arg, **kwargs)
            return results
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


class FreeAxes:

    def __init__(self, fig, shape):
        self.fig = fig
        length = shape[0] * shape[1]
        self.shape = shape
        self.axes = np.array([None] * length)


    def add_subplot(self, index):
        ax = self.axes[index]
        if ax is None:
            ax = self.fig.add_subplot(self.shape[0], self.shape[1], index + 1)
        self.axes[index] = ax
        return ax

    def __iter__(self):
        return iter(self.axes)

    def __len__(self):
        return len(self.axes)

    def __getitem__(self, index):
        ax = self.axes[index]
        if ax is None:
            ax = self.add_subplot(index)
        return ax


class FreePlot:
    """
    A simple implement is used to draw some easy figures in my sense. 
    It is actually a rewrite based on matplotlib and seaborn as the former 
    is flexible but difficult to use and the latter is eaiser but not flexible.
    Therefore, I try my best to combine the both to make it easy to draw.
    At least, in my opinion, it's helpful.
    """
    def __init__(
        self, shape, figsize, titles=None,
        **kwargs
    ):
        """
        If you are familiar with plt.subplots, you will find most of 
        kwargs can be used here directly except
        titles: a list or tuple including the subtitles for differents axes.
        You can ignore this argument and we will assign (a), (b) ... as a default setting.
        Titles will be useful if you want call a axe by the subtitles or endowing the axes 
        different titles together.
        """
        self.root = _ROOT
        self.fig = plt.figure(figsize=figsize, **kwargs)
        self.axes = FreeAxes(self.fig, shape)   
        self.titles = self.initialize_titles(titles)

    def _bound(
        self, values, 
        the_min, the_max, 
        nums, 
        need_max=False,
        need_min=False
    ):
        is_min = values >= the_min
        is_max = values <= the_max
        values = values[is_min & is_max]
        if not need_min:
            the_min = values.min()
        if not need_max:
            the_max = values.max()
        return np.linspace(the_min, the_max, nums)
        

    def initialize_titles(self, titles):
        n = len(self.axes)
        if titles is None:
            names = dict()
            for i in range(n):
                s = "(" + chr(i + 97) + ")"
                names.update({s:i})
        else:
            names = dict()
            for i in range(n):
                title = titles[i]
                names.update({title:i})
        return names

    def legend(self, x, y, ncol, ax=0, loc="lower left"):
        self[ax].legend(bbox_to_anchor=(x, y), loc=loc,
        bbox_transform=plt.gcf().transFigure, ncol=ncol)

    def subplots_adjust(
        self,
        left=None, bottom=None, right=None, 
        top=None, wspace=None, hspace=None
    ):
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    def savefig(
        self, filename, 
        bbox_inches='tight', 
        tight_layout=True,
        **kwargs
    ):
        if tight_layout:
            plt.tight_layout()
        plt.savefig(
            self.root + filename,
            bbox_inches=bbox_inches,
            **kwargs
        )


    def set(self, index=None, **kwargs):
        if index is None:
            axes = self.axes
        else:
            axes = self[index]
        if not isinstance(axes, Iterable):
            axes = [axes]
        for ax in axes:
            ax.set(**kwargs)

    def set_titles(self, y=-0.3):
        for title in self.titles:
            ax = self[title]
            ax.set_title(title, y=y)

    def show(self, tight_layout=True):
        if tight_layout:
            plt.tight_layout()
        plt.show()

    def set_xticklabels_one(
        self, index, 
        the_min=0., the_max=1., 
        nums=5, format="%.2f",
        need_max=False,
        need_min=False
    ):
        ax = self[index]
        values = ax.get_xticks()
        values = self._bound(values, the_min, the_max, nums, 
                        need_max=need_max, need_min=need_min)
        labels = [format%value for value in values]
        ax.set_xticks(values)
        ax.set_xticklabels(labels)
    
    def set_yticklabels_one(
        self, index, 
        the_min=0., the_max=1., 
        nums=5, format="%.2f",
        need_max=False,
        need_min=False
    ):
        ax = self[index]
        values = ax.get_yticks()
        values = self._bound(values, the_min, the_max, nums, 
                        need_max=need_max, need_min=need_min)
        labels = [format%value for value in values]
        ax.set_yticks(values)
        ax.set_yticklabels(labels)
    
    def set_xticklabels(self, the_min=0., the_max=1., nums=5, format="%.2f"):
        axes = self.axes
        for ax in axes:
            values = ax.get_xticks()
            values = self._bound(values, the_min, the_max, nums)
            labels = [format%value for value in values]
            ax.set_xticks(values)
            ax.set_xticklabels(labels)

    def set_yticklabels(self, the_min=0., the_max=1., nums=5, format="%.2f"):
        axes = self.axes
        for ax in axes:
            values = ax.get_yticks()
            values = self._bound(values, the_min, the_max, nums)
            labels = [format%value for value in values]
            ax.set_yticks(values)
            ax.set_yticklabels(labels)

    def _extend_index(self, index):
        if not isinstance(index, (list, tuple)):
            index = [index]
        axes = self[index]
        return axes

    @style_env(cfg.heatmap_style)
    def heatmap(
        self, data, index=0, 
        annot=True, format=".4f",
        cmap='GnBu', linewidth=.5,
         **kwargs
    ):
        """
        data: M x N dataframe.
        cmap: GnBu, Oranges are recommanded.
        annot: annotation.
        fmt: the format for annotation.
        kwargs:
            cbar: bool
        """
        ax = self[index]
        sns.heatmap(
            data, ax=ax, 
            annot=annot, fmt=format,
            cmap=cmap, linewidth=linewidth,
            **kwargs
        )

    @style_env(cfg.lineplot_style)
    def lineplot(self, x, y, index=0, seaborn=False, **kwargs):
        ax = self[index]
        if seaborn:
            sns.lineplot(x, y, ax=ax, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
        
    @style_env(cfg.scatterplot_style)
    def scatterplot(self, x, y, index=0, seaborn=False, **kwargs):
        ax = self[index]
        if seaborn:
            sns.scatterplot(x, y, ax=ax, **kwargs)
        else:
            ax.scatter(x, y, **kwargs)

    @style_env(cfg.imageplot_style)
    def imageplot(self, img, index=0, show_ticks=False, **kwargs):
        ax = self[index]
        try:
            assert img.shape[2] == 3
            ax.imshow(img, **kwargs)
        except AssertionError:
            ax.imshow(img.squeeze(), cmap="gray", **kwargs)
        if not show_ticks:
            ax.set(xticks=[], yticks=[])
        

    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            ax = []
            for i in index:
                try:
                    ind = self.titles[i]
                except KeyError:
                    ind = i
                finally:
                    ax.append(self.axes[ind])
            return ax
        else:
            try:
                ind = self.titles[index]
            except KeyError:
                ind = index
            finally:
                return self.axes[ind]



 
    

    
        




