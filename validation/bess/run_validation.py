"""
BESS validation — standalone.

Tests the battery model (src.rec_sim.Bess) with controlled charge/discharge
profiles and verifies energy balances, SOC limits, C-rate limits,
self-discharge, and round-trip efficiency.

No external input file needed — test profiles are generated internally.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.rec_sim.Bess import Bess
from src.rec_sim.Controller import Controller

# BESS parameters — edit here
BESS_PARAMS = dict(
    id='bess_validation',
    cap=100.0,          # kWh
    c_rate=0.5,         # 0.5C → max 50 kW
    soc_in=0.5,
    soc_max=0.9,
    soc_min=0.1,
    eta_charge=0.95,
    eta_discharge=0.95,
    self_discharge_rate_per_hour=0.00004,
    cap_cost=0, opex_cost=0, inc_year=0, inc_start_end=[0, 0], tax_year=0,
)

TIME_STEP = 1.0  # hours


def run_profile(bess, power_profile, time_step):
    """Run a power profile through the BESS and collect results."""
    records = []
    for t, p_in in enumerate(power_profile):
        soc_before = bess.soc_in
        power_in, soc, power_from_source, supply, power, surplus, deficit = \
            bess.energy_performance(p_in, time_step)
        records.append({
            'step': t,
            'power_in': power_in,
            'soc_before': soc_before,
            'soc_after': soc,
            'power_from_source': power_from_source,
            'supply': supply,
            'power': power,
            'surplus': surplus,
            'deficit': deficit,
        })
        bess.soc_in = soc
    return pd.DataFrame(records)


def check_energy_balance(df, bess_cap, time_step, self_discharge_rate, soc_min):
    """Verify that SOC changes match the energy exchanged + self-discharge."""
    delta_soc = df['soc_after'] - df['soc_before']
    soc_after_sd = df['soc_before'] * (1 - self_discharge_rate * time_step)
    sd_loss = df['soc_before'] - np.maximum(soc_after_sd, soc_min)
    energy_from_power = df['power'] * time_step / bess_cap
    error = (delta_soc + sd_loss - energy_from_power).abs()
    max_err = error.max()
    print(f"  Energy balance max error: {max_err:.2e} (SOC units)")
    return max_err < 1e-10


def check_soc_limits(df, soc_min, soc_max):
    """Verify SOC stays within bounds."""
    soc_ok = (df['soc_after'] >= soc_min - 1e-12).all() and \
             (df['soc_after'] <= soc_max + 1e-12).all()
    print(f"  SOC range: [{df['soc_after'].min():.6f}, {df['soc_after'].max():.6f}]"
          f"  limits: [{soc_min}, {soc_max}]  -> {'OK' if soc_ok else 'FAIL'}")
    return soc_ok


def check_crate_limit(df, c_rate, bess_cap):
    """Verify power never exceeds C-rate."""
    p_max = c_rate * bess_cap
    power_ok = (df['power'].abs() <= p_max + 1e-10).all()
    print(f"  Max |power|: {df['power'].abs().max():.2f} kW"
          f"  limit: {p_max:.2f} kW  -> {'OK' if power_ok else 'FAIL'}")
    return power_ok


def check_self_discharge(bess_params, time_step, n_hours=100):
    """Verify self-discharge in idle."""
    bess = Bess(**bess_params)
    soc_0 = bess.soc_in
    rate = bess.self_discharge_rate_per_hour

    soc_history = [soc_0]
    for _ in range(n_hours):
        _, soc, *_ = bess.energy_performance(0.0, time_step)
        bess.soc_in = soc
        soc_history.append(soc)

    soc_history = np.array(soc_history)
    expected = soc_0 * (1 - rate * time_step) ** np.arange(n_hours + 1)
    max_err = np.abs(soc_history - expected).max()
    print(f"  Self-discharge max error over {n_hours}h: {max_err:.2e}")
    return max_err < 1e-10, soc_history, expected


def check_roundtrip_efficiency(bess_params, time_step):
    """Full charge then full discharge — verify roundtrip = eta_ch * eta_dis."""
    params = dict(bess_params)
    params['soc_in'] = params['soc_min']
    params['self_discharge_rate_per_hour'] = 0.0
    bess = Bess(**params)

    p_max = bess.c_rate * bess.cap
    energy_in = 0.0
    steps_charge = 0
    while bess.soc_in < bess.soc_max - 1e-12:
        _, soc, power_from_source, *_ = bess.energy_performance(p_max, time_step)
        energy_in += power_from_source * time_step
        bess.soc_in = soc
        steps_charge += 1
        if steps_charge > 1000:
            break

    energy_out = 0.0
    steps_discharge = 0
    while bess.soc_in > bess.soc_min + 1e-12:
        _, soc, _, supply, *_ = bess.energy_performance(-p_max, time_step)
        energy_out += supply * time_step
        bess.soc_in = soc
        steps_discharge += 1
        if steps_discharge > 1000:
            break

    eta_rt = energy_out / energy_in if energy_in > 0 else 0
    eta_expected = bess.eta_charge * bess.eta_discharge
    print(f"  Energy in:  {energy_in:.4f} kWh")
    print(f"  Energy out: {energy_out:.4f} kWh")
    print(f"  Roundtrip efficiency: {eta_rt:.6f}  expected: {eta_expected:.6f}"
          f"  error: {abs(eta_rt - eta_expected):.2e}")
    return abs(eta_rt - eta_expected) < 1e-6


def main():
    output_dir = HERE / 'Output'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("BESS VALIDATION")
    print("=" * 60)

    results = {}

    # --- Test 1: Charge/Discharge cycle with energy balance ---
    print("\n[1] Charge/Discharge cycle — energy balance, SOC limits, C-rate")
    bess = Bess(**BESS_PARAMS)
    n = 48
    profile = np.zeros(n)
    profile[0:12] = 80     # charge (above C-rate limit to test clipping)
    profile[12:24] = 0     # idle
    profile[24:36] = -80   # discharge (above C-rate limit)
    profile[36:48] = 20    # gentle charge

    df = run_profile(bess, profile, TIME_STEP)

    results['energy_balance'] = check_energy_balance(df, BESS_PARAMS['cap'], TIME_STEP,
                                                       BESS_PARAMS['self_discharge_rate_per_hour'],
                                                       BESS_PARAMS['soc_min'])
    results['soc_limits'] = check_soc_limits(df, BESS_PARAMS['soc_min'], BESS_PARAMS['soc_max'])
    results['crate_limit'] = check_crate_limit(df, BESS_PARAMS['c_rate'], BESS_PARAMS['cap'])

    # --- Test 2: Self-discharge ---
    print("\n[2] Self-discharge (idle for 100h)")
    ok, soc_hist, soc_expected = check_self_discharge(BESS_PARAMS, TIME_STEP)
    results['self_discharge'] = ok

    # --- Test 3: Roundtrip efficiency ---
    print("\n[3] Roundtrip efficiency (full charge + full discharge)")
    results['roundtrip'] = check_roundtrip_efficiency(BESS_PARAMS, TIME_STEP)

    # --- Test 4: Controller with 2 batteries ---
    print("\n[4] Controller — 2 batteries, energy balance, SOC priority")
    b1 = Bess(id='b1', cap=100.0, c_rate=0.5, soc_in=0.3, soc_max=0.9, soc_min=0.1,
              eta_charge=0.95, eta_discharge=0.95, self_discharge_rate_per_hour=0.0,
              cap_cost=0, opex_cost=0, inc_year=0, inc_start_end=[0, 0], tax_year=0)
    b2 = Bess(id='b2', cap=50.0, c_rate=0.5, soc_in=0.7, soc_max=0.9, soc_min=0.1,
              eta_charge=0.95, eta_discharge=0.95, self_discharge_rate_per_hour=0.0,
              cap_cost=0, opex_cost=0, inc_year=0, inc_start_end=[0, 0], tax_year=0)
    ctrl = Controller(bess=[b1, b2])

    n_ctrl = 48
    prod = np.zeros(n_ctrl)
    dem = np.zeros(n_ctrl)
    prod[0:12] = 100;  dem[0:12] = 20    # surplus → charge
    prod[12:24] = 10;  dem[12:24] = 10   # balanced → idle
    prod[24:36] = 20;  dem[24:36] = 100  # deficit → discharge
    prod[36:48] = 60;  dem[36:48] = 30   # moderate surplus

    pfs_ev, supply_ev, power_ev, surplus_ev, deficit_ev, soc_ev = \
        ctrl.energy_performance(prod, dem, TIME_STEP)

    # 4a: System-level energy balance at each step
    ctrl_balance_ok = True
    max_balance_err = 0.0
    for i in range(n_ctrl):
        self_cons = min(prod[i], dem[i])
        if prod[i] >= dem[i]:
            err = abs(prod[i] - (self_cons + pfs_ev[i] + surplus_ev[i]))
        else:
            err = abs(dem[i] - (self_cons + supply_ev[i] + deficit_ev[i]))
        max_balance_err = max(max_balance_err, err)
        if err > 1e-6:
            ctrl_balance_ok = False
    print(f"  System energy balance max error: {max_balance_err:.2e} kW")
    results['ctrl_energy_balance'] = ctrl_balance_ok

    # 4b: SOC priority — during charge, lower-SOC battery gets charged first
    soc_b1 = b1.en_perf_evolution['soc']
    soc_b2 = b2.en_perf_evolution['soc']
    pfs_b1 = b1.en_perf_evolution['power_from_source']
    pfs_b2 = b2.en_perf_evolution['power_from_source']
    supply_b1 = b1.en_perf_evolution['supply']
    supply_b2 = b2.en_perf_evolution['supply']

    # At step 0: b1 starts at 0.3, b2 at 0.7 → b1 should charge first
    priority_ok = pfs_b1[0] >= pfs_b2[0]
    print(f"  Charge priority (step 0): b1(soc=0.3) absorbed {pfs_b1[0]:.1f} kW,"
          f" b2(soc=0.7) absorbed {pfs_b2[0]:.1f} kW -> {'OK' if priority_ok else 'FAIL'}")
    results['ctrl_charge_priority'] = priority_ok

    # 4c: During discharge, higher-SOC battery discharges first
    # Find first discharge step where both batteries have charge
    dis_priority_ok = True
    for i in range(24, 36):
        if supply_b1[i] > 0 and supply_b2[i] > 0:
            break
    else:
        i = 24
    if soc_b1[max(0, i-1)] > soc_b2[max(0, i-1)]:
        dis_priority_ok = supply_b1[i] >= supply_b2[i]
    elif soc_b2[max(0, i-1)] > soc_b1[max(0, i-1)]:
        dis_priority_ok = supply_b2[i] >= supply_b1[i]
    print(f"  Discharge priority: -> {'OK' if dis_priority_ok else 'FAIL'}")
    results['ctrl_discharge_priority'] = dis_priority_ok

    # 4d: All individual battery SOCs stay within limits
    soc_limits_ctrl = True
    for batt in [b1, b2]:
        s = batt.en_perf_evolution['soc']
        if (s < batt.soc_min - 1e-12).any() or (s > batt.soc_max + 1e-12).any():
            soc_limits_ctrl = False
    print(f"  Individual SOC limits: b1[{soc_b1.min():.4f}, {soc_b1.max():.4f}]"
          f"  b2[{soc_b2.min():.4f}, {soc_b2.max():.4f}]"
          f"  -> {'OK' if soc_limits_ctrl else 'FAIL'}")
    results['ctrl_soc_limits'] = soc_limits_ctrl

    # --- Summary ---
    print("\n" + "=" * 60)
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 60)

    # --- Plots ---

    # Plot 1: SOC + power over time
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax1.step(df['step'], df['power_in'], where='mid', color='gray', linewidth=1, label='power_in (requested)')
    ax1.step(df['step'], df['power'], where='mid', color='tab:blue', linewidth=1.5, label='power (actual)')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('Power [kW]')
    ax1.set_title('BESS validation — Charge/Discharge cycle')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.step(df['step'], df['soc_after'], where='mid', color='tab:orange', linewidth=1.5)
    ax2.axhline(BESS_PARAMS['soc_max'], color='red', linewidth=0.8, linestyle='--', label='SOC max')
    ax2.axhline(BESS_PARAMS['soc_min'], color='red', linewidth=0.8, linestyle='--', label='SOC min')
    ax2.set_ylabel('SOC [-]')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3.step(df['step'], df['surplus'], where='mid', color='tab:green', linewidth=1, label='surplus')
    ax3.step(df['step'], df['deficit'], where='mid', color='tab:red', linewidth=1, label='deficit')
    ax3.set_ylabel('Power [kW]')
    ax3.set_xlabel('Time step [h]')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'cycle_test.png', dpi=120)
    plt.close(fig)

    # Plot 2: Self-discharge
    fig, ax = plt.subplots(figsize=(8, 4))
    t_hours = np.arange(len(soc_hist))
    ax.plot(t_hours, soc_hist, 'o-', ms=2, color='tab:blue', label='Model')
    ax.plot(t_hours, soc_expected, '--', color='tab:red', label='Expected (analytical)')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('SOC [-]')
    ax.set_title('Self-discharge test (idle)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'self_discharge.png', dpi=120)
    plt.close(fig)

    # Plot 3: Controller — 2 batteries SOC + system power
    steps = np.arange(n_ctrl)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax1.step(steps, prod, where='mid', color='tab:green', linewidth=1, label='production')
    ax1.step(steps, dem, where='mid', color='tab:red', linewidth=1, label='demand')
    ax1.step(steps, prod - dem, where='mid', color='tab:blue', linewidth=1.5, linestyle='--', label='net (prod-dem)')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('Power [kW]')
    ax1.set_title('Controller validation — 2 batteries (b1=100kWh, b2=50kWh)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.step(steps, soc_b1, where='mid', color='tab:blue', linewidth=1.5, label=f'b1 SOC (cap={b1.cap}kWh)')
    ax2.step(steps, soc_b2, where='mid', color='tab:orange', linewidth=1.5, label=f'b2 SOC (cap={b2.cap}kWh)')
    ax2.step(steps, soc_ev, where='mid', color='gray', linewidth=1, linestyle='--', label='SOC weighted avg')
    ax2.axhline(0.9, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.axhline(0.1, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_ylabel('SOC [-]')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3.step(steps, pfs_ev, where='mid', color='tab:green', linewidth=1, label='power_from_source (total)')
    ax3.step(steps, supply_ev, where='mid', color='tab:red', linewidth=1, label='supply (total)')
    ax3.step(steps, surplus_ev, where='mid', color='tab:cyan', linewidth=1, label='surplus')
    ax3.step(steps, deficit_ev, where='mid', color='tab:pink', linewidth=1, label='deficit')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel('Power [kW]')
    ax3.set_xlabel('Time step [h]')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'controller_test.png', dpi=120)
    plt.close(fig)

    # Save data
    df.to_csv(output_dir / 'cycle_results.csv', index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
