"""Plotting helpers for validation (overlay + scatter + IV curve)."""
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt


def plot_overlay(time_index, pred, ref, title, ylabel, output_path,
                 pred_label='pyres', ref_label='TRNSYS'):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_index, ref, label=ref_label, color='black', linewidth=1)
    ax.plot(time_index, pred, label=pred_label, color='tab:blue', linewidth=1, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_scatter(pred, ref, title, unit, output_path,
                 pred_label='pyres', ref_label='TRNSYS'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ref, pred, s=5, alpha=0.4)
    lim = max(float(max(ref)), float(max(pred)))
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='1:1')
    ax.set_xlabel(f'{ref_label} [{unit}]')
    ax.set_ylabel(f'{pred_label} [{unit}]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_diff(time_index, pred, ref, title, ylabel, output_path):
    diff = pred - ref
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_index, diff, color='tab:red', linewidth=0.8, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('time')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_iv_curve(pv, output_path, n_points=500):
    """Plot I-V and P-V characteristic curves at STC using the single-diode model.
    Notable points (Isc, Voc, MPP) are computed via PvPanels.compute_output_0."""
    il = pv.il_ref
    io = pv.io_ref
    gam = pv.gam
    rs = pv.r_serie
    qbz = pv.qbz
    tc = pv.t_cell_ref_c
    voc = pv.voc_ref

    # Model-computed points at STC
    vmp, imp, p_max, voc_model, isc, _ = pv.compute_output_0(
        I_total=np.array([pv.I_tot_ref]),
        t_amb=np.array([25.0]),
    )
    vmp, imp, p_max = float(vmp[0]), float(imp[0]), float(p_max[0])
    voc_model, isc = float(voc_model[0]), float(isc[0])

    # Manufacturer datasheet values scaled to array
    # isc_ref, imppt_ref are per-module; voc_ref, vmaxappx are already array-level
    isc_ds = pv.isc_ref * pv.n_parallel
    voc_ds = pv.voc_ref
    vmp_ds = pv.vmaxappx
    imp_ds = pv.imppt_ref * pv.n_parallel
    p_ds = pv.pmaxappx * 1000

    v_arr = np.linspace(0, voc, n_points)
    i_arr = np.zeros(n_points)

    for j, v in enumerate(v_arr):
        def f(i):
            return il - io * np.exp(qbz * (v + i * rs) / (gam * tc)) - i
        try:
            i_arr[j] = brentq(f, 0, il, xtol=1e-6)
        except ValueError:
            i_arr[j] = 0.0

    p_arr = v_arr * i_arr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(v_arr, i_arr, color='tab:blue', linewidth=1.5, label='Model')
    ax1.plot(0, isc, 'o', color='tab:green', ms=8, label=f'Model Isc = {isc:.2f} A')
    ax1.plot(voc_model, 0, 's', color='tab:red', ms=8, label=f'Model Voc = {voc_model:.1f} V')
    ax1.plot(vmp, imp, 'D', color='tab:purple', ms=8, label=f'Model MPP ({vmp:.1f} V, {imp:.2f} A)')
    ax1.plot(0, isc_ds, 'x', color='tab:green', ms=10, mew=2, label=f'Datasheet Isc = {isc_ds:.2f} A')
    ax1.plot(voc_ds, 0, 'x', color='tab:red', ms=10, mew=2, label=f'Datasheet Voc = {voc_ds:.1f} V')
    ax1.plot(vmp_ds, imp_ds, 'x', color='tab:purple', ms=10, mew=2, label=f'Datasheet MPP ({vmp_ds:.1f} V, {imp_ds:.2f} A)')
    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Current [A]')
    ax1.set_title('I-V curve (STC) — Model vs Datasheet')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)

    ax2.plot(v_arr, p_arr / 1000, color='tab:orange', linewidth=1.5, label='Model')
    ax2.plot(vmp, p_max / 1000, 'D', color='tab:purple', ms=8, label=f'Model Pmax = {p_max/1000:.2f} kW')
    ax2.plot(vmp_ds, p_ds / 1000, 'x', color='tab:purple', ms=10, mew=2, label=f'Datasheet Pmax = {p_ds/1000:.2f} kW')
    ax2.set_xlabel('Voltage [V]')
    ax2.set_ylabel('Power [kW]')
    ax2.set_title('P-V curve (STC) — Model vs Datasheet')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, None)
    ax2.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
