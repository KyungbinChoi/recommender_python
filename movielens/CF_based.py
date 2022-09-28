"""
Collaborative filtering 
Coumpute Similarity of neighborhoods
Save recommendation results using item based collaborative filtering & user based collaborative filtering
"""

import os
import pandas as pd
import numpy as np
import commons.Utils as U
from commons.dataloader import movielens_dataloader

def pre_process(rating_df, cf_base = 'item'):
    """rating data 에서 pivoting 된 dataframe 반환

    Args:
        rating_df (dataframe): train set 에 해당하는 rating dataframe
        cf_base (str, optional): item based cf , user based cf 에 따라 pivot 기준 선택. Defaults to 'item'.

    Returns:
        dataframe: similarity 가 계산된 dataframe (item by item | user by user)
    """
    if cf_base == "item":
        rating_pivot = rating_df.pivot_table('rating', index = 'movieId', columns='userId').fillna(0)
    elif cf_base == "user":
        rating_pivot = rating_df.pivot_table('rating', index = 'userId', columns='movieId').fillna(0)
    similarity_arr = U.get_cosine_similarity(df = rating_pivot)
    similarity_df = pd.DataFrame(data=similarity_arr, columns = rating_pivot.index, index= rating_pivot.index)

    return similarity_df

def get_test_data(rating_df, last_n):
    """get train, test set dataframe based on user history

    Args:
        item_df (dataframe): rating dataframe
        last_n : test set size (last n reviews on time line)
    return:
        user rating train set, test set
    """
    train_df, test_df = U.split_train_test(rating_df, grouping = 'userId', time_order =last_n, seed_fix=True)
    return train_df, test_df

def item_based_cf_recommender(user_row, similarity_df, history_num=5, result_num = 3):
    
    return 

def user_based_cf_recommender(user_row, similarity_df, history_num=5, result_num = 3):
    
    return 

def get_item_based_CF_result(train_df, item_df, similarity_df):

    return 

def get_user_based_CF_result(train_df, item_df, similarity_df):
    
    return 



def main(cf_base = "item"):
    """main function for execution
    Args:
        cf_base (str, optional): if item, running based on item based collaborative filtering. Defaults to "item".
    """
    dataloader = movielens_dataloader() 
    item_df = dataloader.get_item_data()
    rating_df = dataloader.get_rating_data()
    train_df, test_df = get_test_data(rating_df, last_n=3)
    similarity_df = pre_process(train_df, cf_base = 'item')
    if cf_base =="item":
        recommendations = get_item_based_CF_result(train_df, item_df, similarity_df)
    elif cf_base == "user":
        recommendations = get_user_based_CF_result()
    test_history_df = test_df.sort_values(by=['userId','timestamp']).groupby('userId')['movieId'].apply(list)
    recommendations_with_test = pd.merge(recommendations, test_history_df, on='userId', how='left')
    result_path = os.getcwd()+'/results/movielens_'+'{}_CF'.format(cf_base)+'.pkl'
    print(result_path)
    recommendations_with_test.to_pickle(result_path)
    
    return