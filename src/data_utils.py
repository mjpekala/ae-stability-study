import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

def process_file(filepath, max_distance=None):
    '''Reads in the pickle file for a run output and replaces distances
    outputs a DataFrame
    max distance to use, if not specified, will be set to the maximum occurring distance * 2'''
    df = pd.read_pickle(filepath)
    print(df.columns)
    datasets = pd.unique(df['Dataset'])
    examples = pd.unique(df['Example#'])
    print('frame has %d datasets and %d unique examples' % (len(datasets), len(examples)))
    print(datasets)
    d_max = round(df.boundary_distance[df.boundary_distance != np.inf].max())
    print('Maximum boundary distance found was {}'.format(d_max))
    if max_distance == None:
        df.replace(np.inf, 2 * d_max, inplace=True)
        print('Replacing with {}'.format(2 * d_max))
    else:
        df.loc[df.boundary_distance > max_distance, 'boundary_distance'] = max_distance
        print('Replacing with {}'.format(max_distance))
    df.describe()
    return df

def calculate_stability(df, fudge_factor):
    '''Calculates local stability epsilon and maximum value of k that has gaas directions that change class within epsilon
    This is slow, but hard to speed up dramatically'''

    example_stability = df[df.direction_type == 'gaussian'].groupby(['Example#', 'Dataset'])[
        'boundary_distance', 'k', 'delta_l2'].min()
    example_stability.boundary_distance = example_stability.boundary_distance / fudge_factor
    # note a difference that this time I'm replacing the boundary distance with our fudged estimate

    for dataset in df.Dataset.unique():
        for examp in df[df.Dataset == dataset]['Example#'].unique():
            eps = example_stability.loc[examp, dataset].boundary_distance  # look up closest boundary
            x = df.loc[((df['Example#'] == examp) & (df.Dataset == dataset) & (df.direction_type == 'gaas') & (
                        df.boundary_distance < eps)), 'k'].max()
            if np.isnan(x):
                example_stability.loc[(examp, dataset), 'k'] = 0
            else:
                example_stability.loc[(examp, dataset), 'k'] = x

    return example_stability

def get_pairwise_difs(example_stability):
    '''Returns a dictionary of dataframes, with dataset as key
    Dataframe gives differences in epsilon, k, and epsilon vs ae distance
     '''
    idx = pd.IndexSlice
    delta_epk = {}
    for atk in example_stability.index.get_level_values(1).unique():
        delta_epk[atk] = {examp: (
        (example_stability.loc[examp, 'cifar10'].boundary_distance - example_stability.loc[examp, atk].delta_l2), (
                    example_stability.loc[examp, 'cifar10'].boundary_distance - example_stability.loc[
                examp, atk].boundary_distance),
        (example_stability.loc[examp, 'cifar10'].k - example_stability.loc[examp, atk].k)) \
                          for examp in example_stability.loc[idx[:, atk], :].index.get_level_values('Example#')}

    difs = {}
    for atk, val in delta_epk.items():
        difs[atk] = pd.DataFrame.from_dict(val, orient='index')
        difs[atk].columns = ['eps-l2', 'cifar-ae', 'delta_k']
    return difs
