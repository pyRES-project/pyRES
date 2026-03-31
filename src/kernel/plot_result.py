"""
Created on June 7 08:00:00 2025

@author: isabella pizzuti
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.kernel.run import  *
from datetime import datetime



def filter_by_period(df, period,year):
    df = df.copy()
    if period == 'Winter':
        mask = ((df.index >= f"{year}-12-21") & (df.index <= f"{year}-12-22"))
    elif period == 'Spring':
        mask = ((df.index >= f"{year}-03-22") & (df.index <= f"{year}-03-23"))
    elif period == 'Summer':
        mask = ((df.index >= f"{year}-06-21") & (df.index <= f"{year}-06-22"))
    elif period == 'Autumn':
        mask = ((df.index >= f"{year}-09-24") & (df.index <= f"{year}-09-25"))

    df = df.loc[mask]
    return df


def to_df(en_perf_evolution, timeline):
    clean_dict = {}
    for k, v in en_perf_evolution.items():
        if isinstance(v, pd.Series):
            clean_dict[k] = v.values
        elif isinstance(v, (list, np.ndarray)):
            clean_dict[k] = np.array(v)
        else:
            raise TypeError(f"Unsupported type for {k}: {type(v)}")
    df = pd.DataFrame(clean_dict, index=timeline)
    return df


def _plot_cashflow(ec_perf, component_id, output_dir, time_horizon):
    """
    Generate annual cashflow bar chart (revenues vs costs) and cumulative cashflow line.

    :param ec_perf: dict with economic performance arrays (from Economics.compute_cashflow)
    :param component_id: str --> prosumer or REC id for title and filename
    :param output_dir: str --> directory where plots are saved
    :param time_horizon: int --> number of years
    """
    years = np.arange(0, time_horizon + 1)

    # --- Plot 1: Annual revenues and costs stacked bar chart ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Revenues (positive, stacked)
    rev_sale = ec_perf['rev_from_sale']
    rev_savings = ec_perf['rev_savings']
    rev_incentives = ec_perf['rev_incentives']
    rev_others = ec_perf['rev_others']

    ax.bar(years, rev_sale, width=0.6, label='Sale revenue', color='tab:blue')
    ax.bar(years, rev_savings, width=0.6, bottom=rev_sale,
           label='Self-cons. savings', color='tab:cyan')
    ax.bar(years, rev_incentives, width=0.6, bottom=rev_sale + rev_savings,
           label='Incentives', color='tab:green')
    ax.bar(years, rev_others, width=0.6, bottom=rev_sale + rev_savings + rev_incentives,
           label='Other revenues', color='tab:olive')

    # Costs (negative, stacked)
    cost_resources = -ec_perf['cost_resources']
    cost_opex = -ec_perf['cost_opex']
    cost_taxes = -ec_perf['cost_taxes']
    cost_taxes_sale = -ec_perf['cost_taxes_on_sale']
    cost_others = -ec_perf['cost_others']
    cost_replacement = -ec_perf['cost_bess_replacement']

    ax.bar(years, cost_resources, width=0.6, label='Grid purchase', color='tab:red')
    ax.bar(years, cost_opex, width=0.6, bottom=cost_resources,
           label='OPEX', color='tab:orange')
    ax.bar(years, cost_taxes, width=0.6, bottom=cost_resources + cost_opex,
           label='Taxes', color='tab:brown')
    ax.bar(years, cost_taxes_sale, width=0.6,
           bottom=cost_resources + cost_opex + cost_taxes,
           label='Tax on sales', color='tab:pink')
    ax.bar(years, cost_others, width=0.6,
           bottom=cost_resources + cost_opex + cost_taxes + cost_taxes_sale,
           label='Other costs', color='tab:gray')
    ax.bar(years, cost_replacement, width=0.6,
           bottom=cost_resources + cost_opex + cost_taxes + cost_taxes_sale + cost_others,
           label='BESS replacement', color='tab:purple')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title(f'Annual Revenues & Costs - {component_id}', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Amount [\u20ac]', fontsize=14)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{component_id}_cashflow_annual.png', dpi=150)
    plt.show()
    plt.close()

    # --- Plot 2: Cumulative cashflow with NPV and PBP ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Annual net cashflow
    inflow = rev_sale + rev_savings + rev_incentives + rev_others
    # Note: cost arrays are already negative from above, negate back
    outflow = (ec_perf['cost_resources'] + ec_perf['cost_opex'] + ec_perf['cost_taxes']
               + ec_perf['cost_taxes_on_sale'] + ec_perf['cost_others']
               + ec_perf['cost_bess_replacement'])
    # Year 0: CAPEX as outflow
    net_cashflow = inflow.copy()
    net_cashflow[0] = -ec_perf['capex']
    net_cashflow[1:] = inflow[1:] - outflow[1:]

    cumulative = np.cumsum(net_cashflow)

    ax.bar(years, net_cashflow, width=0.6, color=['tab:red' if v < 0 else 'tab:green' for v in net_cashflow],
           alpha=0.6, label='Net cashflow')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')

    # Cumulative cashflow on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(years, cumulative, 'o-', color='tab:blue', linewidth=2, markersize=5, label='Cumulative cashflow')
    ax2.set_ylabel('Cumulative [\u20ac]', fontsize=14, color='tab:blue')
    ax2.tick_params(axis='y', labelsize=12, labelcolor='tab:blue')

    # NPV and PBP annotations
    npv = ec_perf['NPV']
    pbp = ec_perf['pbp']
    ax.annotate(f'NPV = {npv:,.0f} \u20ac', xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=14, fontweight='bold',
                color='green' if npv > 0 else 'red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    if 0 < pbp < time_horizon:
        ax.annotate(f'PBP = {pbp:.1f} years', xy=(0.02, 0.87), xycoords='axes fraction',
                    fontsize=14, fontweight='bold', color='tab:blue',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
        # Vertical line at payback
        ax.axvline(x=pbp, color='tab:blue', linewidth=1, linestyle=':', alpha=0.7)

    ax.set_title(f'Cumulative Cashflow - {component_id}', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Net cashflow [\u20ac]', fontsize=14)
    # Combined legend from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{component_id}_cashflow_cumulative.png', dpi=150)
    plt.show()
    plt.close()

    # --- Plot 3: Revenue/cost breakdown pie chart (total over horizon) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Revenues pie
    rev_totals = {
        'Sale revenue': np.sum(rev_sale[1:]),
        'Self-cons. savings': np.sum(rev_savings[1:]),
        'Incentives': np.sum(rev_incentives[1:]),
        'Other revenues': np.sum(rev_others[1:]),
    }
    rev_totals = {k: v for k, v in rev_totals.items() if v > 0}
    if rev_totals:
        colors_rev = ['tab:blue', 'tab:cyan', 'tab:green', 'tab:olive'][:len(rev_totals)]
        wedges1, texts1, autotexts1 = ax1.pie(
            rev_totals.values(), labels=rev_totals.keys(), autopct='%1.1f%%',
            colors=colors_rev, startangle=90, textprops={'fontsize': 11})
        ax1.set_title(f'Revenue Breakdown - {component_id}\nTotal: {sum(rev_totals.values()):,.0f} \u20ac',
                      fontsize=14)

    # Costs pie
    cost_totals = {
        'Grid purchase': np.sum(ec_perf['cost_resources'][1:]),
        'OPEX': np.sum(ec_perf['cost_opex'][1:]),
        'Taxes': np.sum(ec_perf['cost_taxes'][1:]),
        'Tax on sales': np.sum(ec_perf['cost_taxes_on_sale'][1:]),
        'Other costs': np.sum(ec_perf['cost_others'][1:]),
        'BESS replacement': np.sum(ec_perf['cost_bess_replacement'][1:]),
    }
    cost_totals = {k: v for k, v in cost_totals.items() if v > 0}
    if cost_totals:
        colors_cost = ['tab:red', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:purple'][:len(cost_totals)]
        wedges2, texts2, autotexts2 = ax2.pie(
            cost_totals.values(), labels=cost_totals.keys(), autopct='%1.1f%%',
            colors=colors_cost, startangle=90, textprops={'fontsize': 11})
        ax2.set_title(f'Cost Breakdown - {component_id}\nTotal: {sum(cost_totals.values()):,.0f} \u20ac (+ {ec_perf["capex"]:,.0f} \u20ac CAPEX)',
                      fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{component_id}_economic_breakdown.png', dpi=150)
    plt.show()
    plt.close()


def plot(simulation, all_components, output_dir='Output'):
    """
    Generate seasonal and monthly bar plots for prosumers and RECs,
    plus economic performance plots (cashflow, cumulative, breakdown).

    :param simulation: dict with 'timeline', 'start_date', 'time_step'
    :param all_components: dict with 'prosumers', 'recs', etc.
    :param output_dir: str --> directory where plots are saved (default: 'Output')
    """
    timeline=simulation['timeline']
    dt = datetime.strptime(simulation['start_date'], '%d-%m-%Y')
    time_step=simulation['time_step']
    year = dt.year
    time_horizon = len(list(all_components['prosumers'].values())[0].ec_perf.get('rev_from_sale', [])) - 1
    period_list=['Winter', 'Spring', 'Summer', 'Autumn']
    list_prosumer=list(all_components['prosumers'].values())
    list_rec = list(all_components['recs'].values())

    # --- Energy performance plots for prosumers ---
    for prosumer in list_prosumer:
        for carrier in prosumer.carriers:
            df0=to_df(prosumer.en_perf_evolution[carrier], timeline)
            for start, period in enumerate(period_list):
                df=filter_by_period(df0,period,year)
                plt.figure()
                plt.title('{0} {1}'.format(period, prosumer.id))
                plt.xlabel('time [h]')
                plt.ylabel('Power [kW]')
                plt.grid()
                produz = df['prod']
                demand = df['dem']
                asse = df.index
                plt.plot(asse, produz, label='Production',
                         color='blue', linewidth=1)
                plt.plot(asse, demand, label='Demand', color='red', linewidth=1)
                plt.fill_between(asse, produz, demand)
                plt.fill_between(asse, produz, demand, where=(produz > demand), color='orange',
                                     interpolate=True, label='Surplus')
                plt.fill_between(asse, produz, demand, where=(produz <= demand), color='lightblue',
                                 interpolate=True, label='Unmet')
                plt.fill_between(asse, demand, 0, where=(produz >= demand), color='lightyellow',
                                 interpolate=True)
                plt.fill_between(asse, produz, 0, where=(produz <= demand), color='lightyellow',
                                 interpolate=True, label='Self-consumption')
                plt.legend(loc='best')
                plt.grid(True)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))  # max 10 tick
                plt.xticks(rotation=45, ha="right")
                plt.savefig('{0}/{1}_{2}_{3}.png'.format(output_dir, prosumer.id, carrier, period))
                plt.show()
                plt.close()


                df_monthly = df0.resample('ME').sum()*time_step/1000
                plt.figure(figsize=(18, 10))
                ax = plt.subplot(1, 1, 1)
                x_pos = np.arange(1, 13, 1)
                plt.title(f"{carrier} {prosumer.id}", fontsize=20)
                ax.bar(x_pos - 0.24, df_monthly['prod'].values, width=0.10, label='Production', color='tab:blue')
                ax.bar(x_pos - 0.12, df_monthly['dem'].values, width=0.10, label='Demand', color='tab:orange')
                ax.bar(x_pos, df_monthly['self_cons'].values, width=0.10, label='Self-consumption', color='tab:green')
                ax.bar(x_pos + 0.12, df_monthly['surplus'].values, width=0.10, label='Surplus', color='tab:red')
                ax.bar(x_pos + 0.24, df_monthly['unmet'].values, width=0.10, label='Unmet', color='tab:purple')
                plt.title('{0} {1}'.format(carrier, prosumer.id), fontsize=20)
                plt.legend(fontsize=20, framealpha=1, facecolor='white')
                plt.yticks(fontsize=20)
                plt.xticks(x_pos, fontsize=20)
                plt.xlabel('Month', fontsize=20)
                plt.ylabel('Energy [MWh]', fontsize=20)
                plt.savefig('{0}/{1}_{2}_bar.png'.format(output_dir, prosumer.id, carrier))
                plt.show()
                plt.close()

        # --- Economic performance plots for prosumer ---
        if prosumer.ec_perf:
            _plot_cashflow(prosumer.ec_perf, prosumer.id, output_dir, time_horizon)

    # --- Energy performance plots for RECs ---
    for rec in list_rec:
        for carrier in rec.carriers:
            df1=to_df(rec.en_perf_evolution[carrier], timeline)
            for start, period in enumerate(period_list):
                df2=filter_by_period(df1,period,year)
                plt.figure()
                plt.title('{0} {1}'.format(period, rec.id))
                plt.xlabel('time [h]')
                plt.ylabel('Power [kW]')
                plt.grid()
                produz = df2['prod']
                demand = df2['dem']
                asse = df2.index
                plt.plot(asse, produz, label='Production',
                         color='blue', linewidth=1)
                plt.plot(asse, demand, label='Demand', color='red', linewidth=1)
                plt.fill_between(asse, produz, demand)
                plt.fill_between(asse, produz, demand, where=(produz > demand), color='orange',
                                     interpolate=True, label='Surplus')
                plt.fill_between(asse, produz, demand, where=(produz <= demand), color='lightblue',
                                 interpolate=True, label='Unmet')
                plt.fill_between(asse, demand, 0, where=(produz >= demand), color='lightyellow',
                                 interpolate=True)
                plt.fill_between(asse, produz, 0, where=(produz <= demand), color='lightyellow',
                                 interpolate=True, label='Shared')
                plt.legend(loc='best')
                plt.grid(True)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))  # max 10 tick
                plt.xticks(rotation=45, ha="right")
                plt.savefig('{0}/{1}_{2}_{3}.png'.format(output_dir, rec.id, carrier, period))
                plt.show()
                plt.close()

                df_monthly = df1.resample('ME').sum()*time_step/1000
                plt.figure(figsize=(18, 10))
                ax = plt.subplot(1, 1, 1)
                x_pos = np.arange(1, 13, 1)
                plt.title(f"{carrier} {rec.id}", fontsize=20)
                ax.bar(x_pos - 0.24, df_monthly['prod'].values, width=0.10, label='Production', color='tab:blue')
                ax.bar(x_pos - 0.12, df_monthly['dem'].values, width=0.10, label='Demand', color='tab:orange')
                ax.bar(x_pos, df_monthly['shared'].values, width=0.10, label='Shared', color='tab:green')
                ax.bar(x_pos + 0.12, df_monthly['surplus'].values, width=0.10, label='Surplus', color='tab:red')
                ax.bar(x_pos + 0.24, df_monthly['unmet'].values, width=0.10, label='Unmet', color='tab:purple')
                plt.title('{0} {1}'.format(carrier, rec.id), fontsize=20)
                plt.legend(fontsize=20, framealpha=1, facecolor='white')
                plt.yticks(fontsize=20)
                plt.ylabel('Energy [MWh]', fontsize=20)
                plt.xticks(x_pos, fontsize=20)
                plt.xlabel('Month', fontsize=20)
                plt.savefig('{0}/{1}_{2}_bar.png'.format(output_dir, rec.id, carrier))
                plt.show()
                plt.close()

        # --- Economic performance plots for REC ---
        if rec.ec_perf:
            _plot_cashflow(rec.ec_perf, rec.id, output_dir, time_horizon)
