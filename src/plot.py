from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.colors as mcolors

from .test import compute_sign_test


def darken_color(color, factor=0.7):
    """
    Darkens the given color by multiplying its RGB channels by `factor`.
    E.g. factor=0.8 means the color is 20% darker.
    """
    r, g, b, a = mcolors.to_rgba(color)
    return (r * factor, g * factor, b * factor, a)


def plot_conditional(aico, var, var_delta, save_path=None):
    """
    Plots the delta metric of var_delta conditional on a variable var
    in an AICO object. Handles both categorical and continuous var by showing either:
      1) A violin plot (colored by p-values) for each category or
      2) A scatter plot if var is continuous.
    In both cases, an [ALL] violin plot is shown below for the full data distribution.
    """
    # ------------------------------------------------------------------
    # 1) Prepare the data
    # ------------------------------------------------------------------
    x = aico.x_test[aico.vars.loc[var]['columns']]

    # Flatten if multiple columns exist (e.g., one-hot-encoded categorical)
    if x.shape[1] > 1:
        x = np.array(pd.from_dummies(x))  # .get_dummies() merges columns
        x = x.flatten()
    else:
        x = x.values.flatten()

    delta = aico.delta[var_delta]

    # ------------------------------------------------------------------
    # 2) Create main figure and GridSpec
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(
        3, 2,
        width_ratios=[20, 0.3],  # narrower color bar
        height_ratios=[2, 1, 0.3],
        hspace=0.1,
        wspace=0.05
    )

    ax_front = fig.add_subplot(gs[0, 0])     # top subplot
    ax_colorbar = fig.add_subplot(gs[0:2, 1])  # color bar

    # ------------------------------------------------------------------
    # 3) Define colors, colormap, and normalization for p-values
    # ------------------------------------------------------------------
    color_sig = '#00E682'
    color_insig = '#ff6670'
    color_all_default = '#3399ff'  # fallback color for scatter

    p_min = 1e-5
    p_max = 1.0
    log_min, log_max = np.log10([p_min, p_max])

    # Determine the relative position of alpha in log space
    log_alpha = np.log10(aico.alpha)
    x_alpha = (log_alpha - log_min) / (log_max - log_min)

    # Create a custom colormap that transitions
    cmap = LinearSegmentedColormap.from_list(
        'custom_pvalue_colormap',
        [
            (0.0, color_sig),
            (x_alpha, '#eeeeee'),
            (1.0, color_insig)
        ]
    )
    norm = LogNorm(vmin=p_min, vmax=p_max)

    # ------------------------------------------------------------------
    # 4) Plot the top panel
    # ------------------------------------------------------------------
    if aico.vars.loc[var]['type'] == 'categorical':
        # --- 4a) Handle categorical var ---
        unique_classes = np.unique(x)
        delta_by_class = [delta[x == cls] for cls in unique_classes]

        # Compute p-values per category using compute_sign_test
        p_values = pd.concat(
            [compute_sign_test(delta_class, aico.alpha) for delta_class in delta_by_class]
        )
        p_values = (p_values['p_value_lower'] + p_values['p_value_upper']) / 2

        # Convert each sub-array to float32 for the violin plot
        delta_by_class = [
            np.array(delta_class, dtype='float32').flatten()
            for delta_class in delta_by_class
        ]

        # Draw the violin plot (horizontal orientation)
        parts = ax_front.violinplot(delta_by_class, vert=False, showmedians=True)

        # Apply color mapping for each violin based on its p-value
        for p_value, pc in zip(p_values, parts['bodies']):
            color = cmap(norm(max(p_value, p_min)))  # guard for log(0)
            pc.set_facecolor(color)
            pc.set_alpha(1.0)

        # Darken bar-like elements (medians, min, max, etc.)
        for partname in ('cbars', 'cmedians', 'cmins', 'cmaxes'):
            component = parts[partname]
            darker_colors = [
                darken_color(cmap(norm(max(p, p_min))), 0.8)
                for p in p_values
            ]
            component.set_color(darker_colors)
            component.set_alpha(1.0)

        # Set up y-axis labels
        ax_front.set_yticks(range(1, len(unique_classes) + 1))
        ax_front.set_yticklabels(unique_classes, fontsize=14, rotation=45)

    else:
        # --- 4b) Handle continuous var ---
        ax_front.scatter(delta, x, alpha=0.25, c=color_all_default,
                         edgecolor='none', s=25)

    # Cosmetic improvements
    ax_front.set_ylabel(f'{var}', fontsize=16, color='#222222', fontweight="bold")
    ax_front.spines['top'].set_visible(False)
    ax_front.spines['right'].set_visible(False)
    ax_front.spines['bottom'].set_visible(False)
    ax_front.spines['left'].set_color('#d3d3d3')
    ax_front.spines['left'].set_linewidth(1)
    ax_front.spines['bottom'].set_linewidth(1)
    ax_front.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_front.set_facecolor('white')

    # ------------------------------------------------------------------
    # 5) Plot the [ALL] violin in the bottom panel
    # ------------------------------------------------------------------
    ax_violin = fig.add_subplot(gs[1, 0], sharex=ax_front)

    # Compute the p-value for the entire delta distribution
    # (assuming compute_sign_test returns a DataFrame with column 'p_value')
    p_value_all_df = compute_sign_test(delta, aico.alpha)
    p_value_all = (p_value_all_df['p_value_lower'].iloc[0] + p_value_all_df['p_value_upper'].iloc[0]) / 2

    # Map that p-value to the same colormap
    color_all_violin = cmap(norm(max(p_value_all, p_min)))
    color_all_violin_dark = darken_color(color_all_violin, factor=0.8)

    # Create the [ALL] violin
    parts_all = ax_violin.violinplot(
        np.array(delta, dtype='float32'),
        vert=False,
        showextrema=True,
        showmedians=True
    )

    # Apply the color scaling
    for pc in parts_all['bodies']:
        pc.set_facecolor(color_all_violin)
        pc.set_alpha(1.0)

    # Darken bar-like elements (median, min, max)
    for partname in ('cbars', 'cmedians', 'cmins', 'cmaxes'):
        parts_all[partname].set_color(color_all_violin_dark)
        parts_all[partname].set_alpha(1.0)

    ax_violin.set_xlabel(f'Delta of {var_delta}', fontsize=14, color='#222222', fontweight="bold")
    ax_violin.set_yticks([1])
    ax_violin.set_yticklabels(['[ALL]'], fontsize=14, rotation=45, fontweight="bold")
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['right'].set_visible(False)
    ax_violin.spines['left'].set_color('#d3d3d3')
    ax_violin.spines['bottom'].set_color('#d3d3d3')
    ax_violin.spines['left'].set_linewidth(1)
    ax_violin.spines['bottom'].set_linewidth(1)
    ax_violin.tick_params(axis='both', labelsize=12, color='#555555')
    ax_violin.set_facecolor('white')

    # ------------------------------------------------------------------
    # 6) Add the color bar
    # ------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_colorbar)
    cbar.outline.set_visible(False)
    cbar.minorticks_off()
    cbar.ax.tick_params(labelsize=10, colors='#555555')
    cbar.set_label('p-value', fontsize=12, color='#555555', fontweight="bold")

    # ------------------------------------------------------------------
    # 7) Add a vertical line at x=0 (for reference) across both subplots
    # ------------------------------------------------------------------
    x_zero = ax_front.transData.transform((0, 0))[0]
    x_zero_fig = fig.transFigure.inverted().transform((x_zero, 0))[0]

    top_y = ax_front.get_position().y1
    bottom_y = ax_violin.get_position().y0

    line = Line2D(
        [x_zero_fig, x_zero_fig], [bottom_y, top_y],
        color='#999999',
        linestyle='-',
        linewidth=1,
        transform=fig.transFigure
    )
    fig.lines.append(line)

    # ------------------------------------------------------------------
    # 8) Show the final figure
    # ------------------------------------------------------------------
    if save_path is not None:
        plt.savefig(path.join(save_path, f'{var_delta}-{var}.png'), bbox_inches='tight')
    plt.show()
