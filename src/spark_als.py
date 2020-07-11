from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import pandas as pd
import numpy as np
import itertools
import os

# Starting a spark session locally
spark = SparkSession \
    .builder \
    .appName("als-reco") \
    .getOrCreate()

class ALSReco():
    def __init__(self):

        self.data_files = os.path.join(os.path.dirname(__file__), "../data")
        self.movies = self.get_movie_df()
        self.links = self.get_id_links()
        self.ratings = self.get_ratings()

    def get_movie_df(self):
        movies_file_path = "movies_metadata.csv"
        df_movies = spark.read.csv(os.path.join(self.data_files, movies_file_path), header=True).select('id', 'title') \
                    .withColumn('id', col('id').cast('int'))
        # Removing null rows
        df_movies = df_movies.filter(df_movies['id'].isNotNull())
        # Filtering out the corrupted rows
        df_movies = df_movies.filter(~df_movies['title'].rlike("\[*\]"))
        # There are a few duplicate ids and titles
        df_movies = df_movies.dropDuplicates(['id'])

        return df_movies

    def get_id_links(self):
        link_file = "links_small.csv"
        df_links = spark.read.csv(os.path.join(self.data_files, link_file), header=True).select('movieId', 'tmdbId') \
                    .withColumn('movieId', col('movieId').cast('int')) \
                    .withColumn('tmdbId', col('tmdbId').cast('int'))
        # Removing null rows
        df_links = df_links.na.drop()

        return df_links

    def get_ratings(self):
        ratings_file_path = "ratings_small.csv"
        df_ratings = spark.read.csv(os.path.join(self.data_files, ratings_file_path), header=True).select('userId', 'movieId', 'rating') \
                    .withColumn('userId', col('userId').cast('int')) \
                    .withColumn('movieId', col('movieId').cast('int')) \
                    .withColumn('rating', col('rating').cast('int')) \
        # Removing nulls
        df_ratings = df_ratings.na.drop()

        return df_ratings

    def get_optimal_model(self):
        """
        The tunable parameters are explored in the jupyter notebook spark_als.ipynb
        rank = 40
        maxIter = 20
        regParam = 0.1
        """
        als = ALS(rank=40, maxIter=20, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
        model = als.fit(self.ratings)
        return model

    def get_userRecs(self, model):
        # Generate top 10 movie recommendations for each user
        userRecs = model.recommendForAllUsers(10)
        return userRecs

    def movieId_to_title(self, movieId):
        tmdbId = self.links.where(self.links['movieId'] == movieId).collect()[0].tmdbId
        try:
            title = self.movies.where(self.movies['id'] == tmdbId).collect()[0].title
        except IndexError:
            title = ''
        return title

    def recommend_movies(self, userId):
        opt_model = self.get_optimal_model()
        userRecs = self.get_userRecs(opt_model)
        
        reco_mov = [row.movieId for row in userRecs.select('recommendations').where(userRecs['userId'] == userId).collect()[0][0]]

        for i, mov in zip(range(1,11), map(self.movieId_to_title, reco_mov)):
            print(f"{i}. {mov}")
