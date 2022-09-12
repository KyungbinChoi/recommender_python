"""
dataloader module for rating and item data
"""

import os
import pandas as pd
import numpy as np

class movielens_dataloader:
    def __init__(self) -> None:
        """
        setting constant terms & data path
        """
        os.chdir('./movielens/data')
        self.rating_path = "./ratings.csv"
        self.item_path = "./movies.csv"
        self.tags_path = "./tags.csv"
        
        self.rating_cols = ["userId", 'movieId', 'rating', 'timestamp']
        self.items_cols = ["movieId", 'title', 'genres']
        self.tags_cols = ["userId", 'movieId', 'tag', 'timestamp']
        pass

    def get_rating_data(self):
        """
            load rating dataframe
        """
        ratings = pd.read_csv(self.rating_path)
        for col in self.rating_cols:
            if col in ratings.columns:
                pass
            else:
                print("Not contatined column or different name. Column name : {}".format(col))
        print("shape of data : {}".format(ratings.shape))
        return ratings
    
    def get_item_data(self):
        """
            load item (movies) dataframe
        """
        items = pd.read_csv(self.item_path)
        for col in self.items_cols:
            if col in items.columns:
                pass
            else:
                print("Not contatined column or different name. Column name : {}".format(col))
        print("shape of data : {}".format(items.shape))
        return items

    def get_tags_data(self):
        """
            load user-item tagging dataframe
        """
        tag = pd.read_csv(self.tags_path)
        for col in self.tags_cols:
            if col in tag.columns:
                pass
            else:
                print("Not contatined column or different name. Column name : {}".format(col))
        print("shape of data : {}".format(tag.shape))
        return tag

    def get_pivoting_dataframe(self):
        """
            construct user-item matrix 
        """
        return