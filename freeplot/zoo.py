


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE






def tsne(features, labels, fp, index=0, **kwargs):
    data_embedded = TSNE(n_components=2, learning_rate=10, n_iter=1000).fit_transform(features)
    fp[index].set_xticks([])
    fp[index].set_yticks([])
    data = pd.DataFrame(
        {
            "x": data_embedded[:, 0],
            "y": data_embedded[:, 1],
            "label": labels
        }
    )
    for label in np.unique(labels):
        event = data.loc[data['label']==label]
        x = event['x']
        y = event['y']
        x_mean = x.median()
        y_mean = y.median()
        plt.text(x_mean, y_mean, label)
        fp.scatterplot(x, y, index, label=label, s=1.5, edgecolors="none", **kwargs)
    sns.despine(left=True, bottom=True)


def roc_curve(
    y_pred, y_labels, 
    fp, index=0, 
    name=None,
    estimator_name=None,
    style="whitegrid",
    dict_=None,
    **kwargs
):
    """
    y_pred: the prediction
    y_labels: the corresponding labels of instances
    fp: ...
    index: ...
    name: for labelling the roc_curve, is None, use the estimator_name
    estimator_name: the name of classifier
    style: the style of seaborn
    dict_: the correspoding properties dict
    """
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, 
                            roc_auc=roc_auc, estimator_name=estimator_name)
    with sns.axes_style(style, dict_):
        display.plot(fp[index], name)
    return tpr, fpr, roc_auc
