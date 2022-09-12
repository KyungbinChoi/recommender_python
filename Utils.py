"""
Evaludataion for recommendation result
"""

import numpy as np
import math, random
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def rmse():

    return

def mae():

    return

def get_cosine_similarity(vector_arr = None, df = None):
    """cosine similarity 

    Args:
        vector_arr (list or array, optional): input for cosine similarity of two vectors. Defaults to None.
        df (dataframe, optional): input for cosine similarity of two matrix (dataframe). Defaults to None.
    """
    if vector_arr:
        sim_result = np.dot(vector_arr[0], vector_arr[1]) / (np.linalg.norm(vector_arr[0],vector_arr[1]))
    if df:
        sim_result = cosine_similarity(df, df)
    return sim_result

def split_train_test(df, grouping = False, time_order =False, seed_fix=False, sample_pct = 0.0):
    """train & test split

    Args:
        df (dataframe): raw dataframe (user rating dataframe)
        grouping (bool or string, optional): grouping key for group by split. Defaults to False.
        time_order (bool or int, optional): time ordering split (not random sampling). test set will be last n event
        seed_fix (bool, optional): setting seed number . Defaults to False.
        sample_pct (float) : sampling ratio in case of random sampling
    """
    
    def sampling_func(data, seed_num):
        np.random.seed(seed_num)
        N = len(data)
        sample_n = int(len(data)*sample_pct)
        sample = data.take(np.random.permutation(N)[:sample_n])
        return sample

    if seed_fix:
        seed_num = 42    
    else:
        seed_num = random.randint(1,100)
    
    if grouping:
        if time_order:
            df['time_order'] = df.groupby(by=['userId'])['timestamp'].transform(lambda x: x.rank(method='first',ascending =False))
            train_df = df.loc[df['time_order']> time_order, :]
            test_df = df.loc[df['time_order']<= time_order, :]
        else:
            train_df = df.groupby(grouping, group_keys= False).apply(sampling_func, seed_num)
            test_df = df.drop(df.index[train_df.index])
    else:
        train_df, test_df = train_test_split(df, test_size = 1-sample_pct)

    return train_df, test_df