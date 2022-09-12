"""
Contents based recommender

Using main function
    - running contents based recommendation
    - save recommendation results for test set
    - extract recommendation score (rmse)
"""

import pandas as pd
import Utils as U
from dataloader import movielens_dataloader
from sklearn.feature_extraction.text import CountVectorizer

def pre_process(item_df, tag_df):
    """pre processing movie item dataframe (movie title, genre)

    Args:
        item_df (DataFrame) : movielens item data
        item_tag (DataFrame) : movielens tag data
    retrun : 
        conut vector matrix dataframe from genre
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

    return item_attr_df

def get_test_data(rating_df, last_n):
    """get train, test set dataframe based on user history

    Args:
        item_df (dataframe): rating dataframe
        last_n : test set size (last n reviews on time line)
    """
    train_df, test_df = U.split_train_test(rating_df, grouping = 'userId', time_order =last_n, seed_fix=True)
    return train_df, test_df

def contents_based_recommender(similarilty_df, item_df):
    """_summary_

    Args:
        similarilty_df (dataframe): _description_
        item_df (dataframe): _description_
    """

    return

def get_prediction_result():
    
    return


def main():
    dataloader = movielens_dataloader() 
    item_df = dataloader.get_item_data()
    
    return

if __name__ == "__main__":
    main()