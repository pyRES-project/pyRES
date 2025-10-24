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



def plot(simulation,all_components):
    timeline=simulation['timeline']
    dt = datetime.strptime(simulation['start_date'], '%d-%m-%Y')
    time_step=simulation['time_step']
    year = dt.year
    period_list=['Winter', 'Spring', 'Summer', 'Autumn']
    list_prosumer=list(all_components['prosumers'].values())
    list_rec = list(all_components['recs'].values())
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
                plt.savefig('Output/{0}_{1}_{2}.png'.format(prosumer.id, carrier, period))
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
                plt.savefig('Output/{0}_{1}_bar.png'.format(prosumer.id, carrier))
                plt.show()
                plt.close()

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
                plt.savefig('Output/{0}_{1}_{2}.png'.format(rec.id, carrier, period))
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
                plt.savefig('Output/{0}_{1}_bar.png'.format(rec.id, carrier))
                plt.show()
                plt.close()


