import matplotlib.pyplot as plt

# プロットの作成
# -i X軸データ(), Y軸データ平均値(list), Y軸データ標準偏差(list), 座標軸(axis), プロットの色(str), ラベル(str), ラベルを表示するか(boolean), X軸の範囲(list), Y軸の範囲(list),
#    タイトル(str), X軸ラベル(str), Y軸ラベル(str),
#    タイトルサイズ(int), X軸ラベルサイズ(int), Y軸ラベルサイズ(int), 凡例のサイズ(int), 透過度(float), 凡例の位置(str), 背景パターン(str: gray/white), その他
# -o ---
def my_plot(X, Y, Y_std, ax, color, label, title, xlabel, ylabel, show_label=True, xlim=None, ylim=None,
            title_size=20, xlabel_size=18, ylabel_size=18, legend_size=14, alpha=.3, loc='upper right', back_ground='gray'):
    """ 
    input
    --------------------
    X (list) : X-axis data
    Y (list) : Y-axis data
    Y_std (list) : Y-axis standard deviation data
    ax (axis)
    color (str)
    label (str)
    title (str)
    xlabel (str)
    ylabel (str)
    show_label (boolean)
    xlim (list)
    ylim (list)
    title_size (int)
    xlabel_size (int)
    ylabel_size (int)
    legend_size (int)
    alpha (float)
    loc (str)
    back_ground (str) : back ground color
    """
    Y_upper = [y + y_std for y,y_std in zip(Y, Y_std)]
    Y_lower = [y - y_std for y,y_std in zip(Y, Y_std)]

    ax.fill_between(X, Y_upper, Y_lower, facecolor=color, alpha=alpha)
    ax.plot(X, Y, color=color, label=label)

    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel(ylabel, fontsize=ylabel_size)
    ax.set_title(title, fontsize=title_size)
    if back_ground=='gray':
        ax.set_facecolor('gainsboro')
        ax.grid(color='white')
    elif back_ground=='white':
        ax.set_facecolor('white')
        ax.grid(color='gainsboro')
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if show_label: ax.legend(loc=loc, fontsize=legend_size)
    ax.set_axisbelow(True)