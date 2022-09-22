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

def pre_process(item_df, tag_df):
    return


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


def CF_based_recommender(user_row, item_df, item_cf_df, history_num=5, result_num = 3):
    """
    """
    
    def get_item_based_cf(movieId, df, result_num = 3):
        items =  item_cf_df[movieId].sort_values(ascending=False).index.values
        results = items[items != movieId][:result_num]
        return results

    recommendation_list = np.array([])
    last_view_history = user_row[:history_num]
    for h in last_view_history:
        recommendation_list = np.concatenate((recommendation_list, get_item_based_cf(h)), axis=None)
    
    # 추천 목록 중 과거에 봤던 목록은 제거
    recommendation_result = np.setdiff1d(recommendation_list, last_view_history)[:result_num]
    return recommendation_result

