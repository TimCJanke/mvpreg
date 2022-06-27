import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd


def plot_dm_test_matrix(data: pd.DataFrame, width=15, height=12.5, title=None, cbar=True, cmap=None, linewidths=0.1, fontsize_title=10, fontsize_labels=10, fontsize_cbar=10, annot=True, annot_fmt='.3f', annot_kws={"size": 6}, save_to_path=None):
    """ Plot heatmap of p-values """
    if cmap is None:
        cmap = _get_green_red_yellow_colormap()

    plt.figure(figsize=(width/2.54,height/2.54))
    ax = sns.heatmap(data, linewidths=linewidths, annot=annot, fmt=annot_fmt, annot_kws=annot_kws, cmap=cmap, square=True, vmin=-0.0001, vmax=0.101, cbar=cbar)
    if cbar:
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0,0.01,0.05,0.1])
        colorbar.set_ticklabels([0,0.01,0.05,0.1])
        colorbar.ax.tick_params(labelsize=fontsize_cbar)
    plt.xlabel("")
    plt.ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize_labels)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize_labels)
    ax = plt.gca()
    ax.set_facecolor('darkgrey')
    plt.title(title, fontsize=fontsize_title)
    plt.tight_layout()
    
    if save_to_path is not None:
        plt.savefig(save_to_path ,dpi=200, bbox_inches='tight',pad_inches=0)


def _get_green_red_yellow_colormap():
    """ returns custom colormap for dm matrix plots"""
    cdict1 = {'blue':  ((0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.1),
                    (0.95, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

            'green': ((0.0, 0.5, 0.5),
                    (0.5, 1.0, 1.0),
                    (0.95, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

            'red':  ((0.0, 0.1, 0.1),
                    (0.5, 1.0, 1.0),
                    (0.95, 0.8, 0.8),
                    (1.0, 0.0, 0.0))
            }

    return LinearSegmentedColormap('Green_Yellow', cdict1, gamma=1) 



def PIT_histogram_grid(pobs, nrows, ncols, n_bins=20, width=40, height=20, title=None, dim_labels=None):
    
    fig, axs = plt.subplots(nrows, ncols, sharey=True, sharex=True, figsize=(width/2.54,height/2.54))
    
    for i in range(pobs.shape[1]):
        if pobs.shape[1]==1:
            ax = axs
        else:
            ax = axs.T.flatten()[i]            
        ax.hist(pobs[:, i], bins=np.linspace(0.0,1.0,n_bins+1), rwidth=0.95)
        ax.hlines(pobs.shape[0]/n_bins, xmin=-0.1, xmax=1.1, linestyles = ':', color = 'black',lw=1.5)
        ax.set_ylim(0,pobs.shape[0]/n_bins*1.75)
        ax.set_xlim(-0.01,1.01)
        ax.set_facecolor('gainsboro')
    
    fig.suptitle(title, fontsize=12)
    #fig.tight_layout(rect=[0, 0.0, 1, 0.975])
    fig.tight_layout()

    


def fan_plot(y_true, q_pred, taus, point_forecasts=None):
    #TODO
    pass



def scenario_plot_temporal(y_true, s_pred=None, point_forecasts=None):
    #TODO
    pass


def scenario_plot_spatial(y_true, y_predict, point_forecasts=None, kde=False, dim_names=None, title=None):
    
    y_true_df = pd.DataFrame(np.reshape(y_true, (1,-1)), columns=dim_names)
    y_true_df["label"] = "realized_value"

    y_predict_df = pd.DataFrame(y_predict, columns=dim_names)
    y_predict_df["label"] = "samples"

    df = pd.concat((y_predict_df, y_true_df), axis=0, ignore_index=True)
    
    g = sns.pairplot(df, hue="label", plot_kws=dict(marker="+", linewidth=1), diag_kind="kde", kind="scatter")
    g.map_lower(sns.kdeplot, levels=10, warn_singular=False, fill=True)

    def plot_vline(data, **kwargs):
        if len(data)==1:
            plt.gca().vlines(data.values[0], *plt.gca().get_ylim(), color=sns.color_palette()[1])

    def plot_obs(x,y, **kwargs):
        if len(x)==1:
            plt.gca().scatter(x.values[0], y.values[0], color=sns.color_palette()[1], s=100, marker="o")

    g.map_diag(plot_vline)
    g.map_offdiag(plot_obs)
    