import pandas as pd
import pingouin as pg
import csv
import matplotlib.pyplot as plt
from scipy.stats import sem
import seaborn as sns
import numpy as np


def pd_from_data(metafile, datafiles):
    metadf = pd.read_csv(metafile)
    items = []
    for dataf in datafiles:
        file1 = open(dataf, "r")
        dat = file1.readlines()
        for d in dat:
            vals = d.split(',')
            vals[0] = vals[0].strip()
            vals[1] = vals[1].strip()
            vals[-1] = float(vals[-1].strip())
            [dom, bh_nm, fin_val] = vals
            if 'auto so' in bh_nm:
                continue
            metadata = metadf[(metadf['behavior'] == bh_nm) & (metadf['domain'] == dom)].values.flatten().tolist()
            metadata.append(fin_val)
            if type(metadata[0]) == float:
                x = 0
            items.append(metadata)
        file1.close()

    col_names = metadf.columns.tolist()
    col_names.append('value')
    df = pd.DataFrame(items, columns=col_names)
    # print(df.loc[:, ['behavior', 'value']])
    return df

def test_anova(df, plot=False):
    # https://ethanweed.github.io/pythonbook/05.05-anova2.html#factorial-anova-versus-one-way-anovas
    chars = ['behavior', 'bh size', 'raw', 'policy output','auto']
    for c in chars:
        model1 = pg.anova(dv='value', between=[c], data=df, detailed=True)
        round(model1, 4)
        print(model1.loc[:, ['Source', 'F', 'p-unc', 'np2']], '\n')
    # model2 = pg.pairwise_tests(dv='value',
    #                   between=['raw', 'policy output'],
    #                   padjust='holm',
    #                   data=df)
    # print(model2.loc[:, ['A', 'B', 'p-corr']])
    if plot:
        for c in chars:
            plt.clf()
            sns.pointplot(data=df, x=c, y='value',  hue='domain')  #
            sns.despine()
            if c == 'behavior':
                plt.xticks(rotation=-45)
            plt.ylim([.25, .85])
            plt.show()


if __name__ == '__main__':
    metafn = '/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/Behavior definitions - Sheet1.csv'
    rovfile = ['/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/graphs/graphs_579_580_581_585/NOTES_fin_values.txt']
    hopfile = ['/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data_gym/hopper/graphs/graphs_018_019/NOTES_fin_values.txt']
    mtfile = ['/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data_gym/mountain/graphs/graphs_024/NOTES_fin_values.txt']
    df_rov = pd_from_data(metafn, rovfile)
    df_hop = pd_from_data(metafn, hopfile)
    df_mt = pd_from_data(metafn, mtfile)
    df_all = pd.concat([df_rov, df_hop, df_mt], ignore_index=True)
    # print("############    ROVER   ############")
    # test_anova(df_rov)
    # print("############   HOPPER   ############")
    # test_anova(df_hop)
    # print("############   MOUNTAIN   ############")
    # test_anova(df_mt)
    # print("############   ALL   ############")
    # test_anova(df_all, plot=True)
    # for df in [df_rov, df_hop, df_all]:
    #     df.plot(x='bh size', y='value', kind="scatter")
    #     plt.ylim([0.2, 0.7])
    #     plt.xlim([0, 31])
    #     plt.show()
    #     plt.clf()
    chs = ['behavior', 'bh size', 'raw', 'policy output','auto']
    for ch in chs:
        print('')
        print(ch)
        print(pd.DataFrame(round(df_all.groupby(by=[ch])['value'].agg([np.mean, sem]),4)).reset_index())
        # print(pd.DataFrame(round(df_all.groupby(by=[ch])['value'].sem(),4)).reset_index())

