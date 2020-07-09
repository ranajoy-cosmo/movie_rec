import argparse
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class KnnRecommender():
    def __init__(self):
        self.data_files = os.path.join(os.path.dirname(__file__), "../data")
        self.df_movies = self.get_clean_movie_data()
        self.df_ratings = self.get_clean_rating_data()
        self.df_links = self.get_linking_data()
        self.df_interaction = self.get_interaction_matrix()
        self.enc_user, self.enc_mov, self.dec_user, self.dec_mov = self.encode_indices()

    def get_clean_movie_data(self):
        # Read in the movie id and title
        df_movies = pd.read_csv(os.path.join(self.data_files, 'movies_metadata.csv'), usecols=['id', 'title'])
        # Some bad ids with date. Remove them.
        # Drop rows with bad id and change dtype
        df_movies.drop(df_movies[df_movies.id.str.isnumeric() == False].index, inplace=True)
        df_movies.id = df_movies.id.astype(np.int32)
        # Drop rows with null values
        df_movies.dropna(inplace=True)
        # Remove duplicates in id and title
        df_movies.drop_duplicates(subset=['id'], inplace=True)
        df_movies.drop_duplicates(subset=['title'], inplace=True)
        return df_movies

    def get_clean_rating_data(self):
        # Read in the ratings file
        df_ratings = pd.read_csv(os.path.join(self.data_files, 'ratings_small.csv'), usecols=['userId', 'movieId', 'rating'], dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
        return df_ratings

    def get_linking_data(self):
        # Link between movieId and tmdbId
        df_links = pd.read_csv(os.path.join(self.data_files, "links_small.csv"), usecols=['movieId', 'tmdbId'])
        # Few null values present
        df_links.dropna(inplace=True)
        df_links.movieId = df_links.movieId.astype(np.int32)
        df_links.tmdbId = df_links.tmdbId.astype(np.int32)
        return df_links

    def get_interaction_matrix(self, user_thres=25, movie_thres=25):
        df_interaction = self.df_ratings.pivot(index='userId', columns='movieId', values='rating')
        # Drop users and movies that do not make the threshold of munimum ratings
        # Caveat: After removing users, movies will end up with fewer total ratings. On removing movies, users will in turn end up with fewer ratings.
        # TO DO: Find a way of optimally removing users and movies to keep the threshold.
        # df_interaction.dropna(axis=0, thresh=user_thres, inplace=True)
        # df_interaction.dropna(axis=1, thresh=movie_thres, inplace=True)
        # Now fill the remaining NaNs with 0
        df_interaction.fillna(0, inplace=True)
        return df_interaction

    def encode_indices(self):
        """
        Numbering the rows and columns sequentially from 0.
        Only the df_interaction gets transformed
        """
        orig_ind = self.df_interaction.index
        orig_cols = self.df_interaction.columns
        df_shape = self.df_interaction.shape
        # Encoder
        enc_user = dict(zip(orig_ind, np.arange(df_shape[0])))
        enc_mov = dict(zip(orig_cols, np.arange(df_shape[1])))
        # Decoder
        dec_user = dict(zip(np.arange(df_shape[0]), orig_ind))
        dec_mov = dict(zip(np.arange(df_shape[1]), orig_cols))
        # Reindexing rows and column names
        self.df_interaction.rename(index=enc_user, columns=enc_mov, inplace=True)
        return enc_user, enc_mov, dec_user, dec_mov

    def movieId_to_title(self, movieId):
        """
        Get the title by linking the tmdbId in ratings to id in movies
        """
        tmdbId = self.df_links[self.df_links.movieId == movieId].tmdbId
        movie_title = self.df_movies.set_index('id').loc[tmdbId]
        return movie_title.iloc[0]['title']

    def title_to_movieId(self, movie_title):
        """
        Get the tmdbID by linking the tmdbId in ratings to id in movies
        """
        tmdbId = self.df_movies[self.df_movies.title == movie_title].id
        movieId = self.df_links[self.df_links.tmdbId == tmdbId].movieId
        return movie_idx.index

    def get_user_top_movie_ratings(self, userId, weight=1, num_top_movies=5):
        """
        Get the top movies of an user with the ratings weighted
        Returns the movieId of the movie
        """
        df_user_top_movies = self.df_ratings[self.df_ratings.userId == userId].sort_values('rating', ascending=False)[:num_top_movies]
        df_user_top_movies.drop(['userId'], axis=1, inplace=True)
        df_user_top_movies.rating *= weight
        return df_user_top_movies

    def get_nearest_ids(self, idx, interaction_matrix, n_neighbors=5):
        """
        Get the nearest rows by KNN method
        Return the distance and the indices
        """
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        model_knn.fit(interaction_matrix)
        distances, indices = model_knn.kneighbors(interaction_matrix[idx], n_neighbors=n_neighbors+1)
        return distances[0,1:], indices[0,1:]

    def recommend_movie_user_user(self, userId, num_movies=10):
        """
        Recommend movies to user by User-User filtering
        """
        num_user_neighbors = 5
        enc_userId = self.enc_user[userId]
        interaction_matrix = csr_matrix(self.df_interaction)
        # Getting the closest users
        dist_list, userId_list = self.get_nearest_ids(enc_userId, interaction_matrix, num_user_neighbors)
        # Compile the weighted rating list of top movies from closest users
        df_rec_movies = pd.DataFrame(columns=['movieId', 'rating'])
        for dist, userId in zip(dist_list, userId_list):
            # The movieId returned is already decoded
            df_rec_movies = df_rec_movies.append(self.get_user_top_movie_ratings(self.dec_user[userId], weight=1.0 - dist), ignore_index=True)
        top_recommended_movieIds = df_rec_movies.sort_values('rating', ascending=False).movieId.unique()[:num_movies]
        return top_recommended_movieIds

    def recommend_movie_item_item(self, userId, num_movies=10):
        """
        Recommend movies to user by Item-Item filtering
        """
        num_movie_neighbors = 5
        interaction_matrix_sparse = csr_matrix(self.df_interaction).transpose()
        # Encoded list of user's top movies
        user_top_movies = self.get_user_top_movie_ratings(userId, num_top_movies=5)
        df_rec_movies = pd.DataFrame(columns=['movieId', 'rating'])
        for i in range(5):
            movieId, rating = user_top_movies.iloc[i]['movieId'], user_top_movies.iloc[i]['rating']
            dist_list, movieId_list = self.get_nearest_ids(self.enc_mov[movieId], interaction_matrix_sparse, num_movie_neighbors)
            dec_movieId_list = [self.dec_mov[mov] for mov in movieId_list]
            df_rec_movies = df_rec_movies.append(pd.DataFrame.from_dict({'movieId': dec_movieId_list, 'rating': dist_list*rating}), ignore_index=True)
        top_recommended_movieIds = df_rec_movies.sort_values('rating', ascending=False).movieId.unique()[:num_movies]
        return top_recommended_movieIds

    def recommend_movies(self, userId, method, num_movies=10):
        method_list = {'user_user': self.recommend_movie_user_user, 'item_item': self.recommend_movie_item_item}
        top_movieIds = method_list[method](userId, num_movies)
        for i, movieId in zip(range(num_movies), top_movieIds):
            print(f"{i+1}: {self.movieId_to_title(movieId)}")

