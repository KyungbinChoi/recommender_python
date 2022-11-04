"""
Collaborative filtering 
Coumpute Similarity of neighborhoods
Save recommendation results using item based collaborative filtering & user based collaborative filtering
"""

import os,sys
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

def item_based_cf_recommender(user_id, movie_id, neighbor_size = 0):
    """Item based 

    Args:
        user_id (int): User id
        movie_id (int): Item id (아직 평가하지 않은 item 에 해당)
        neighbor_size (int, optional): 유사한 item 개수의 설정 값. movie_id 의 평점 예측 시 활용. Defaults to 0. 

    Returns:
        int: user_id 에 해당하는 user 가 특정 item_id 에 해당하는 item의 예측 평점 값
    """
    if movie_id in item_similarity:
        sim_scores = item_similarity[movie_id] # movie_id 와 다른 item 간 rating similarity
        user_rating = item_user_matrix[user_id] # user_id 가 남긴 평점
        non_rating_idx = user_rating[user_rating.isnull()].index # user_id 가 아직 평가하지 않은 item 의 id list
        user_rating = user_rating.dropna() # user_id 의 평가 점수
        sim_scores = sim_scores.drop(non_rating_idx) # user_id 에 해당하는 user가 평가한 item 들과 movie_id 간 유사도 점수 
        
        if neighbor_size == 0:
            mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum() # movie_id 에 해당하는 예측 평점
        else:
            if len(sim_scores) >1: # 영화를 평가한 사람이 최소 2명 이상인 경우
                neighbor_size = min(neighbor_size, len(sim_scores))
                sim_scores = np.array(sim_scores)
                user_rating = np.array(user_rating)
                item_idx = np.argsort(sim_scores) # ascending sort, 유사도가 가장 작은 순서
                sim_scores = sim_scores[item_idx][-neighbor_size:] # 위에서 가장 작은 순서로 정렬했으므로 가장 뒤에서 부터 indexing
                user_rating = user_rating[item_idx][-neighbor_size:]
                mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
            else:
                mean_rating = 3.0
    else: #Cold start item
        mean_rating = 3.0
    
    return mean_rating

def get_item_based_CF_result(user_id, n_items=5, neighbor_size = 30):
    """item base CF의 추천 item 목록 반환 함수

    Args:
        user_id (int): User id
        n_items (int, optional): 추천 결과 개수 설정값  Defaults to 5.
        neighbor_size (int, optional): CF 기반 예측 시 고려할 유사 item 수. Defaults to 30.

    Returns:
        array : 추천 item 의 id list
    """
    user_movie = user_item_matrix.loc[user_id].copy()
    for movie in user_item_matrix.columns:
        if pd.notnull(user_movie.loc[movie]): # 해당 영화가 시청한 영화인 경우
            user_movie.loc[movie] = 0
        else:
            user_movie.loc[movie] = item_based_cf_recommender(user_id, movie, neighbor_size)
    movie_sort = user_movie.sort_values(ascending=False)[:n_items]
    recom_movies = item_df.set_index('movieId').loc[movie_sort.index]
    recommendations_title = recom_movies['title']
    print(recommendations_title)
    return np.array(recom_movies.index)

def user_based_cf_recommender(user_id, movie_id, neighbor_size = 0):
    if movie_id in rating_bias:
        sim_scores = user_similarity[user_id].copy()
        movie_ratings = rating_bias[movie_id].copy()
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)

        if neighbor_size ==0:
            prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            prediction += rating_mean[user_id]
        else:
            if len(sim_scores)> 1:
                neighbor_size = min(neighbor_size, len(sim_scores))
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                user_idx = np.argsort(sim_scores)
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                prediction += rating_mean[user_id]
            else:
                prediction = rating_mean[user_id]
    else:
        prediction = rating_mean[user_id]
    return prediction

def get_user_based_CF_result(user_id, n_items=5, neighbor_size = 30):
    user_movie = user_item_matrix.loc[user_id].copy()
    for movie in user_item_matrix:
        if pd.notnull(user_movie.loc[movie]): # 해당 영화가 시청한 영화인 경우
            user_movie.loc[movie] = 0
        else:
            user_movie.loc[movie] = user_based_cf_recommender(user_id, movie, neighbor_size)
    movie_sort = user_movie.sort_values(ascending=False)[:n_items]
    recom_movies = item_df.set_index('movieId').loc[movie_sort.index]
    recommendations_title = recom_movies['title']
    print(recommendations_title)
    return np.array(recom_movies.index)


if __name__ == "__main__":
    cf_base = sys.argv[1]
    dataloader = movielens_dataloader() 
    item_df = dataloader.get_item_data()
    rating_df = dataloader.get_rating_data()
    item_ft = list(rating_df.movieId.value_counts()[rating_df.movieId.value_counts()>10].index)

    train_df, test_df = get_test_data(rating_df, last_n=3)
    train_df = train_df.loc[train_df['movieId'].isin(item_ft)]
    user_item_matrix = train_df.pivot_table(values='rating', index = 'userId', columns = 'movieId') 
    item_user_matrix = user_item_matrix.T
    item_similarity = pre_process(train_df, cf_base = 'item')
    user_similarity = pre_process(train_df, cf_base = 'user')    
    
    rating_mean = user_item_matrix.mean(axis=1) # average score by user
    rating_bias = (user_item_matrix.T - rating_mean).T # mean centering

    if cf_base == "item": # item base cf
        print('Start recommendation for all users =============')
        pred_df = pd.DataFrame(train_df['userId'].unique(), columns=['userId'])
        pred_df['rec_item_list'] = pred_df['userId'].apply(get_item_based_CF_result)
        print('Finish recommendation for all users =============')
        result_path = os.getcwd()+'/results/movielens_'+'{}_CF'.format(cf_base)+'.pkl'
        print('Save recommendation results =============')
        pred_df.to_pickle(result_path)
    elif cf_base == "user": # user based cf
        print('Start recommendation for all users =============')
        pred_df = pd.DataFrame(train_df['userId'].unique(), columns=['userId'])
        pred_df['rec_item_list'] = pred_df['userId'].apply(get_user_based_CF_result)
        print('Finish recommendation for all users =============')
        result_path = os.getcwd()+'/results/movielens_'+'{}_CF'.format(cf_base)+'.pkl'
        print('Save recommendation results =============')
        pred_df.to_pickle(result_path)
    else:
        print('Choose item base or user base Collaborative filtering')

    
