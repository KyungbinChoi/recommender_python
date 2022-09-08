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
        
        self.rating_cols = []
        self.items_cols = []
        self.tags_cols = []
        pass

    def get_rating_data(self):
        """
            load rating dataframe
        """
        return
    
    def get_item_data(self):
        """
            load item (movies) dataframe
        """
        return

    def get_tags_data(self):
        """
            load user-item tagging dataframe
        """
        return

    def get_pivoting_dataframe(self):
        """
            construct user-item matrix 
        """
        return