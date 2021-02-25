import glob
import numpy as np
from scipy.stats import binned_statistic

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

fontsize = 16


def pseudotau2tau(ptau):
    tau = ptau.copy()
    tau /= 1000.0  # Tasha's scaling
    # cosmology
    h = 0.67321
    low_z_tau = 0.03
    tau = low_z_tau + h**2 * tau
    return tau


def read_data(filenames):
    data = np.load(filenames[0])
    for fn in filenames[1:]:
        data = np.concatenate((data, np.load(fn)))
    data["truth"] = pseudotau2tau(data["truth"])
    data["prediction"] = pseudotau2tau(data["prediction"])
    return data


def bin_center(bin_edges):
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def plot_errors():
    # read in data and compute statistics
    files_modes_removed = sorted(glob.glob("modes_removed/*.npy"))
    files_no_modes_removed = sorted(glob.glob("no_modes_removed/*.npy"))

    full = read_data(files_no_modes_removed)
    cut = read_data(files_modes_removed)

    err_full = full["truth"] - full["prediction"]
    frac_err_full = (full["truth"] - full["prediction"]) / full["truth"]

    bias_full, bias_full_bin_edges, _ = binned_statistic(
        full["truth"].squeeze(), err_full.squeeze(), statistic="mean"
    )
    std_full, std_full_bin_edges, _ = binned_statistic(
        full["truth"].squeeze(), err_full.squeeze(), statistic="std"
    )

    err_cut = cut["truth"] - cut["prediction"]
    frac_err_cut = (cut["truth"] - cut["prediction"]) / cut["truth"]

    bias_cut, bias_cut_bin_edges, _ = binned_statistic(
        cut["truth"].squeeze(), err_cut.squeeze(), statistic="mean"
    )
    std_cut, std_cut_bin_edges, _ = binned_statistic(
        cut["truth"].squeeze(), err_cut.squeeze(), statistic="std"
    )

    # make a figure
    matplotlib.rc("text", usetex=True)
    matplotlib.rc("font", family="serif")
    matplotlib.rc("lines", linewidth=2)

    figsize = (7, 7)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, hspace=0.0)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)

    ax1.axhline(y=0, color="black", linestyle="--")
    ax1.axhline(
        y=0.002, color="blue", linestyle="--", alpha=0.5, label="CMB EE cosmic variance"
    )
    ax1.axhline(y=-0.002, color="blue", linestyle="--", alpha=0.5)
    ax2.axhline(y=0, color="black", linestyle="--")
    ax2.axhline(
        y=0.002, color="blue", linestyle="--", alpha=0.5, label="CMB EE cosmic variance"
    )
    ax2.axhline(y=-0.002, color="blue", linestyle="--", alpha=0.5)

    ax1.plot(full["truth"], err_full, ".", color="gray", alpha=0.5)
    ax1.errorbar(
        bin_center(bias_full_bin_edges),
        bias_full,
        yerr=std_full,
        # capsize=3,
        color="red",
        marker="o",
        label="Mean bias and variance",
    )
    # ax1.set_xlabel(r'$\tau_\mathrm{true}$', size=fontsize)
    ax1.set_ylabel(r"$\tau_\mathrm{true} - \tau_\mathrm{pred}$", size=fontsize)
    ax1.text(0.0477, 0.003, "Full 21 cm cube", size=fontsize)
    leg = ax1.legend(prop={"size": fontsize})

    ax2.plot(cut["truth"], err_cut, ".", color="gray", alpha=0.5)
    ax2.errorbar(
        bin_center(bias_cut_bin_edges),
        bias_cut,
        yerr=std_cut,
        # capsize=3,
        color="red",
        marker="o",
        label="Mean bias and variance",
    )

    ax1.set_ylim([-0.004, 0.004])
    ax2.set_ylim([-0.004, 0.004])
    ax2.set_yticks([-0.004, -0.002, 0, 0.002, 0.004])
    yticks = ax2.get_yticks()
    ax2.set_yticks(yticks[:-1])
    ax2.set_xlim([0.0475, 0.0675])
    ax1.tick_params(labelbottom=False)
    ax1.tick_params(axis="both", labelsize=fontsize)
    ax2.tick_params(axis="both", labelsize=fontsize)

    ax2.set_xlabel(r"$\tau_\mathrm{true}$", size=fontsize)
    ax2.set_ylabel(r"$\tau_\mathrm{true} - \tau_\mathrm{pred}$", size=fontsize)
    ax2.text(0.0477, 0.003, "Wedge cut 21 cm cube", size=fontsize)
    output = "bias_variance.pdf"
    print(f"Saving {output}...")
    fig.savefig(output, bbox_inches="tight")

    return


if __name__ == "__main__":
    plot_errors()
