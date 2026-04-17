"""
PV validation — standalone.

Instantiates src.rec_sim.PvPanels with explicit parameters and feeds it the
same meteo inputs that were used to generate the TRNSYS reference curve.
Compares the pyres output power against TRNSYS and saves metrics + plots.

Input file (CSV or Excel) in validation/pv/Input/ — default name: pv_validation.csv
Required columns (case-insensitive, comma or semicolon separator autodetected):
    datetime
    i_beam          [W/m^2]  beam radiation on horizontal
    i_skydiff       [W/m^2]  sky-diffuse radiation on horizontal
    i_grounddiff    [W/m^2]  ground-reflected radiation on horizontal
    t_amb           [°C]
    theta           [°]      angle of incidence on tilted surface
    power_trnsys_w [W]      TRNSYS reference output
Optional:
    wind_speed      [m/s]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.rec_sim.PvPanels import PvPanels
from validation.common import metrics, plots



# PV parameters — edit here

PV_PARAMS = dict(
    id='pv_validation',
    # module electrical (STC)
    isc_ref=8.7,
    voc_ref=37.95,
    vmppt_ref=31.21,
    imppt_ref=8.33,
    mu_isc_ref=0.03,
    mu_voc_ref=-0.34,
    ser_cell=60,
    t_cell_noct_c=40,
    area=1.63,
    eg=1.12,
    # array layout
    n_series=20,
    n_parallel=39,
    # losses
    dc_ac_efficiency=1,
    mismatch_loss=0,
    wiring_loss=0,
    soiling_loss=0,
    # economics (not used here but required by constructor)
    cap_cost=0, opex_cost=0, inc_year=0, inc_start_end=[0, 0], tax_year=0,
)

# Array tilt (slope) — degrees from horizontal — edit here
SLOPE_DEG = 0

# Input file
INPUT_FILENAME = 'pv_validation.csv'


# ─────────────────────────────────────────────────────────────────────────────

def load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    else:
        # autodetect separator
        df = pd.read_csv(path, sep=None, engine='python')
    df.columns = [c.strip().lower() for c in df.columns]

    required = [ 'i_beam', 'i_skydiff', 'i_grounddiff',
                't_amb', 'theta', 'power_trnsys_w']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
    else:
        df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='1h')
        df.index.name = 'datetime'
    return df


def main():
    input_path = HERE / 'Input' / INPUT_FILENAME
    output_dir = HERE / 'Output'
    output_dir.mkdir(exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"See the docstring of this script for the required columns."
        )

    df = load_input(input_path)

    pv = PvPanels(**PV_PARAMS)

    wind = df['wind_speed'].to_numpy() if 'wind_speed' in df.columns else None

    _, _, _, p_max, *_ = pv.compute_output(
        slope=SLOPE_DEG,
        I_beam=df['i_beam'].to_numpy() / 3.6,
        I_skydiff=df['i_skydiff'].to_numpy() / 3.6,
        I_grounddiff=df['i_grounddiff'].to_numpy() / 3.6,
        t_amb=df['t_amb'].to_numpy(),
        theta=df['theta'].to_numpy(),
        wind_speed=wind,
    )
    # pyres stores production in kW
    prod_kw = np.asarray(pv.en_perf_evolution['electricity']['prod'], dtype=float)

    result = pd.DataFrame({
        'pyres_kw': prod_kw,
        'trnsys_kw': df['power_trnsys_w'].to_numpy() / 1000.0,
    }, index=df.index).dropna()

    m = metrics.summary(result['pyres_kw'], result['trnsys_kw'])
    m.to_csv(output_dir / 'metrics.csv', index=False)
    print(m.to_string(index=False))

    plots.plot_overlay(
        result.index, result['pyres_kw'].values, result['trnsys_kw'].values,
        title='PV production — pyres vs TRNSYS',
        ylabel='Power [kW]',
        output_path=output_dir / 'overlay.png',
    )
    plots.plot_diff(
        result.index, result['pyres_kw'].values, result['trnsys_kw'].values,
        title='PV production difference (pyres − TRNSYS)',
        ylabel='ΔPower [kW]',
        output_path=output_dir / 'diff.png',
    )
    plots.plot_scatter(
        result['pyres_kw'].values, result['trnsys_kw'].values,
        title='PV production — pyres vs TRNSYS',
        unit='kW',
        output_path=output_dir / 'scatter.png',
    )

    plots.plot_iv_curve(pv, output_path=output_dir / 'iv_curve.png')

    result.to_csv(output_dir / 'series_aligned.csv')
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
