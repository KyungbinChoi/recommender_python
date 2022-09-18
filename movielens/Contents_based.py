"""
Contents based recommender

Using main function
    - running contents based recommendation
    - save recommendation results for test set
    - extract recommendation score (rmse)
"""
import os
import pandas as pd
import numpy as np
import commons.Utils as U
from commons.dataloader import movielens_dataloader
from sklearn.feature_extraction.text import CountVectorizer

def pre_process(item_df, tag_df):
    """pre processing movie item dataframe (movie title, genre)

    Args:
        item_df (DataFrame) : movielens item data
        item_tag (DataFrame) : movielens tag data
    retrun : 
        ordered item matrix by similarity (item X item)
    """
    
    item_df['year'] = item_df['title'].str.extract(r'([0-9]{4})')
    item_df['genres'] = item_df['genres'].str.replace('|', ' ')
    item_tags = tag_df.groupby('movieId')['tag'].apply(lambda x:"|".join(x)).reset_index()
    item_tags.columns = ['movieId', 'tag']
    item_df = pd.merge(item_df, item_tags, on = 'movieId', how='left')
    item_df = item_df.fillna("")
    item_df['item_desc'] = item_df['genres'] + "|"  + item_df['tag']

    corpus = item_df['item_desc'].values
    vectorizer = CountVectorizer(max_df=.8, min_df = 2)
    X = vectorizer.fit_transform(corpus)
    item_attr_df = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())
    similarilty_df = U.get_cosine_similarity(df = item_attr_df).argsort()[:, ::-1]

    return similarilty_df

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

def contents_based_recommender(user_row, item_df, gerne_item_sim, history_num=5, result_num = 3):
    """applied function for contents based recommendation

    Args:
        user_row (array): user history (last N item)
        history_num (int, optional) : number of considered histories. Defaults to 3.  
        result_num (int, optional): number of results.
    """
    
    def get_recommend_movie_list(movie_id, top=3):
        # movieId 의 item dataframe 에서의 index를 추출
        item_idx = item_df.loc[item_df['movieId']==movie_id, :].index[0]
        # 해당 index 순서의 similar items list 를 산출
        similar_items = np.delete(gerne_item_sim[item_idx, :], item_idx)

        # 해당 list 에서 처음 뽑은 index 에 해당한 번호를 제외하고 나머지 앞쪽부터 N개 추출
        filtered_idx = similar_items[:top]
        # filter된 list 에 해당하는 movie id 추출 
        result = item_df.iloc[filtered_idx, 0].values
        return result

    recommendation_list = np.array([])
    last_view_history = user_row[:history_num]
    for h in last_view_history:
        recommendation_list = np.concatenate((recommendation_list,get_recommend_movie_list(h)), axis=None)

    recommendation_result = np.setdiff1d(recommendation_list, last_view_history)[:result_num]
    return recommendation_result

def get_prediction_result(train_df, item_df, gerne_item_sim):
    """contents based recommendation
       extract next item based last 5 items that haven't seen yet

    Args:
        train_df (dataframe): User rating dataframe (Train set)

    return:
        recommendation result dataframe (for comparing test set)
    """
    train_user_df = train_df.sort_values(by=['userId','timestamp'],ascending=False).groupby('userId')['movieId'].apply(list)
    recommendation_result = train_user_df.apply(contents_based_recommender, item_df=item_df, gerne_item_sim=gerne_item_sim).reset_index()
    user_smp = np.random.randint(1,50)
    print(user_smp)
    print("Sample user's last history")
    user_history_item = train_user_df[user_smp][:5]
    smp_result = recommendation_result.loc[recommendation_result['userId']==user_smp, 'movieId'].values[0]
    print(item_df.loc[item_df['movieId'].isin(user_history_item), ['title','genres']])
    print(smp_result)
    print('=======recommendation result==========')
    print(item_df.loc[item_df['movieId'].isin(smp_result), ['title','genres']])

    return recommendation_result

def main():
    dataloader = movielens_dataloader() 
    item_df = dataloader.get_item_data()
    rating_df = dataloader.get_rating_data()
    tag_df = dataloader.get_tags_data()
    gerne_item_sim = pre_process(item_df, tag_df)
    train_df, test_df = get_test_data(rating_df, last_n=3)
    recommendations = get_prediction_result(train_df, item_df, gerne_item_sim)
    result_path = os.getcwd()+'/results/movielens_contents_based_results.pkl'
    print(result_path)
    recommendations.to_pickle(result_path)

    return None