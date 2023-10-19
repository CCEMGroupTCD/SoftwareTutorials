"""
This is a pandas tutorial for the CCEM group meeting.
"""
import ast
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random


def get_ligands_grouped_by_denticity(df):
    ligands_grouped_by_denticity = {}
    for _, row in df.iterrows():
        dent = row['denticity']
        name = row['unique_name']

        # If denticity is not yet in dictionary, add it as empty list
        if not dent in ligands_grouped_by_denticity:
            ligands_grouped_by_denticity[dent] = []

        # Append the name of the ligand to the denticity list of names
        ligands_grouped_by_denticity[dent].append(name)

    return ligands_grouped_by_denticity


if __name__ == '__main__':

    ligand_db_path = '/Users/timosommer/PhD/teaching/Python tutorials/pandas/unique_ligand_db_v1.6_for_tutorial.csv'



    #%% Read in the ligand db .csv file as pandas DataFrame
    df = pd.read_csv(ligand_db_path)


    #%% Read in the ligand db .csv file as pandas DataFrame and read in lists as lists, dicts as dicts etc.
    df = pd.read_csv(ligand_db_path,
                     converters={
                                    'coordinating_elements': ast.literal_eval,
                                    'same_graph_denticities': ast.literal_eval,
                                    'count_metals': ast.literal_eval,
                                    'all_ligand_names': ast.literal_eval
                                }
                     )



    #%% Look at the data
    columns = df.columns.tolist()
    print(f'Columns of DataFrame: {columns}')
    shape = df.shape
    print(f'Shape of DataFrame: {shape}')
    print('DataFrame Information:')
    df.info()


    #%% Change column names
    df = df.rename(columns={'pred_charge': 'charge'})


    #%% Get single row (i.e. one unique ligand)
    df = df.set_index('unique_name')
    ulig = df.loc['unq_CSD-PIQHOM-0-1-a',:]
    df = df.reset_index(names='unique_name')


    #%% Get single column
    stoichs = df['stoichiometry']
    monodentates = df['denticity'] == 1

    # or get the most common ligand
    common_lig = df.sort_values(by='occurrences', ascending=False).iloc[0,:]
    # or
    common_lig = df[df['occurrences'] == df['occurrences'].max()].squeeze()


    #%% Filter entries

    # Get all monodentate ligands
    df_mono = df[monodentates]
    # or
    df_mono = df[df['denticity'] == 1]
    # or
    df_mono = df.query('denticity == 1')

    # get all neutral ligands with denticity more than 2
    df_neutral = df[(df['charge'] == 0) & (df['denticity'] > 2)]

    # and if we have NaN values
    df_charged = df[~df['charge'].isna()]

    # get all ligands with denticity of 1, 2, 3
    df_small_denticities = df[df['denticity'].isin([1, 2, 3])]


    #%% Do some math with ligands
    df['dent_plus_charge'] = df['denticity'] + df['charge']

    df_small = df[['dent_plus_charge', 'charge']]




    #%% Pandas is not only good with numbers, also with strings
    # Get fullerenes
    df['fullerene'] = df['stoichiometry'] == 'C60'
    n_fullerenes = df['fullerene'].sum()

    df['has_C60H60'] = df['stoichiometry'].str.contains('C60H60')
    df['replaced_stoi'] = df['stoichiometry'].str.replace('C10', 'AAA')
    df['split_stoi'] = df['stoichiometry'].str.split('1')
    df = df.drop(columns=['has_C60H60', 'replaced_stoi', 'split_stoi'])

    #%% Pandas is even great with python objects. Lists, dictionaries, classes, everything!

    # Get all ligands with experimental precedence to Cu
    def get_metal_list_from_dict(dic):
        metals = list(dic.keys())
        return metals

    df['metals_with_precedence'] = df['count_metals'].apply(get_metal_list_from_dict)
    # or in short
    df['metals_with_precedence'] = df['count_metals'].apply(lambda dic: list(dic.keys()))

    df['ligand_has_Cu_precedence'] = df['metals_with_precedence'].apply(lambda metals: 'Cu' in metals)
    print(f'N ligands with Cu precedence: {sum(df.ligand_has_Cu_precedence)}')

    #%% Some useful functions
    df['dupl_stoichiometry'] = df['stoichiometry'].duplicated(keep='first')
    unique_stoichs = df.drop_duplicates(subset=['stoichiometry'], keep='first')


    #%% Get some statistics

    # What's the mean number of atoms?
    mean_n_atoms = df['n_atoms'].mean()

    stats = df.describe()
    corr = df.corr(method='spearman')

    # How many ligands with which denticity do we have?
    unique_denticities = df['denticity'].unique()
    denticity_counts = {dent: sum(df['denticity'] ==  dent) for dent in unique_denticities}
    # or in short
    denticity_counts = df['denticity'].value_counts().to_dict()

    # Can we get a collection of all unique ligand names by their denticity?
    ligands_grouped_by_denticity = df.groupby(by='denticity')['unique_name'].agg(list)


    #%% Pandas is much faster than pure python! It uses numpy for vectorization and is heavily optimized.

    # In pandas
    start = datetime.now()
    ligands_grouped_by_denticity = df.groupby(by='denticity')['unique_name'].agg(list)
    pandas_duration = datetime.now() - start
    print(f'Pandas execution time: {pandas_duration}')

    # or in pure python
    start = datetime.now()
    ligands_grouped_by_denticity = get_ligands_grouped_by_denticity(df)
    python_duration = datetime.now() - start
    print(f'Pure python execution time: {python_duration}')
    print(f'In this example, pandas is {python_duration/pandas_duration:.0f} times faster than pure python.')


    #%% Plot some statistics
    df.hist(column='denticity', bins=17)
    plt.show()

    df.hist(column='charge')
    plt.show()

    df.plot(x='n_atoms', y='occurrences', kind='scatter')
    plt.show()


    #%% How to construct a dataframe from dictionaries, lists etc.
    # Using random numbers and putting them in two columns called x_name and y_name

    #%% Construct a dataframe from a list of dictionaries. The usually best and most flexible way.
    data = []
    for i in range(100):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        entry = {'x_name': x, 'y_name': y}
        data.append(entry)
    df_data = pd.DataFrame(data)
    df_new = pd.concat()

    #%% Construct a dataframe from a dictionary of dictionaries, if you want to have an index for each entry.
    data = {}
    for i in range(100):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        entry = {'x_name': x, 'y_name': y}
        data[i] = entry
    df_data = pd.DataFrame.from_dict(data, orient='index')

    #%% Construct a dataframe from a list of lists, if you want to provide the column labels afterwards manually.
    data = []
    for i in range(100):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        entry = [x, y]
        data.append(entry)
    df_data = pd.DataFrame(data, columns=['x_name', 'y_name'])

    #%% Construct a dataframe from a list of lists, if you want to provide the column labels afterwards manually.
    data = []
    for i in range(100):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        entry = [x, y]
        data.append(entry)
    df_data = pd.DataFrame(data, columns=['x_name', 'y_name'])










    #%% Preparation of csv.
    # from src01.utilities import unroll_dict_into_columns
    # df = pd.read_json('/Users/timosommer/PhD/projects/RCA/projects/CreateTMC/data/final_db_versions/unique_ligand_db_v1.6.json', orient='index')
    # # df = df.sample(n=10000)
    # df = unroll_dict_into_columns(df, dict_col='global_props', prefix='gbl_', delete_dict=True)
    # df = unroll_dict_into_columns(df, dict_col='stats', prefix='stats_', delete_dict=True)
    # df = df[['unique_name', 'stoichiometry', 'denticity', 'pred_charge', 'pred_charge_is_confident', 'occurrences', 'local_elements', 'n_protons',
    #    'same_graph_denticities', 'count_metals', 'n_same_graph_denticities', 'n_metals', 'n_same_graphs', 'all_ligand_names', 'gbl_CSD_code', 'gbl_n_atoms', 'stats_min_distance_to_metal']]
    # df = df.rename(columns={'local_elements': 'coordinating_elements', 'gbl_CSD_code': 'CSD_code', 'gbl_n_atoms': 'n_atoms', 'stats_min_distance_to_metal': 'min_distance_to_metal'})
    # df = df.reset_index(drop=True)
    # df.to_csv('/Users/timosommer/PhD/teaching/Python tutorials/pandas/unique_ligand_db_v1.6_for_tutorial.csv', index=False)