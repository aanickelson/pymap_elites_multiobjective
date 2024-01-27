import pandas as pd
import pingouin as pg
import csv
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt


def pd_from_data(metafile, dataf):
    metadf = pd.read_csv(metafile)
    items = []
    with open(dataf, newline='') as csvfile:
        csvread = csv.reader(csvfile)

        for i, row in enumerate(csvread):
            if i == 0:
                c_names = metadf.columns.tolist()
                row.extend(c_names)
                col_names = row
                continue

            row[-4:] = [float(v) for v in row[-4:]]
            [dom_nm, date_num, run_num, bh_name, hypervol, pol_par, pol_kept, bh_pct_full] = row
            if bh_name == 'auto mo':
                bh_name = 'auto mo st'
            # elif bh_name == 'auto so':
            #     bh_name = 'auto so st'
            if 'auto so' in bh_name:
                continue

            metadata = metadf[(metadf['behavior'] == bh_name) & (metadf['domain'] == dom_nm)].values.flatten().tolist()
            row.extend(metadata)

            items.append(row)

    # print(items)
    df = pd.DataFrame(items, columns=col_names)
    # print(df.groupby(['domain', 'behavior']).size().reset_index(name='counts'))
    merge_df = pd.DataFrame(round(df.groupby(by=['domain', 'behavior'])['hypervol'].agg([len]), 4).reset_index())
    merge_df = merge_df.merge(metadf, on=['domain', 'behavior'])
    stat_cols = ['hypervol', 'n pareto', 'n pol', 'bh pct full']
    for s in stat_cols:
        stat_df = pd.DataFrame(round(df.groupby(by=['domain', 'behavior'])[s].agg([np.mean, sem]), 4)).reset_index()
        stat_df.rename(columns={'mean':f"{s} mean", 'sem':f"{s} sem"}, inplace=True)
        merge_df = merge_df.merge(stat_df, on=['domain', 'behavior'])
    # print(merge_df)

    save_name = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/sum data/NOTES_summary_vals_h_018_019_020_021_m_024_r_585_.csv'
    all_save_name = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/sum data/NOTES_all_vals_h_018_019_020_021_m_024_r_585_.csv'
    merge_df.to_csv(save_name)
    df.to_csv(all_save_name)
    return merge_df, df


def plot_scatter(df):
    groups = df.groupby('domain')
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group['bh size'], group['hypervol'], marker='o', linestyle='', ms=12, label=name)
    ax.legend()

    plt.show()



if __name__ == '__main__':
    metafn = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/Behavior definitions - Sheet1.csv'
    datafn = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/sum data/NOTES_fin_vals_h_018_019_020_021_022_m_024_025_r_579_580_581_585_586_.csv'
    stats_df, raw_df = pd_from_data(metafn, datafn)
    print(stats_df.loc[:, ['domain', 'behavior', 'len', 'hypervol mean']], '\n')
    print(stats_df['hypervol mean'][stats_df['domain'] == 'hopper'])
    print(stats_df['hypervol mean'].corr(stats_df['bh pct full mean']))
    print(stats_df['hypervol mean'].corr(stats_df['bh size']))
    # plot_scatter(raw_df)
    for dom in ['hopper', 'mountain', 'rover']:
        print(f"############    {dom}   ############")
        for comp_against in ['bh pct full mean', 'bh size']:
            print(stats_df['hypervol mean'][stats_df['domain'] == dom].corr(stats_df[comp_against][stats_df['domain'] == dom]))
        print('')
        chars = ['behavior', 'bh size', 'raw', 'policy output','auto']
        for c in chars:
            model1 = pg.anova(dv='hypervol', between=[c], data=raw_df[raw_df['domain'] == dom], detailed=True)
            round(model1, 4)
            try:
                print(model1.loc[:, ['Source', 'F', 'p-unc', 'np2']], '\n')
            except KeyError:
                print(model1)
            print('')