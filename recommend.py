import argparse
import os
from movie_rec.src.knn_reco import KnnRecommender
from movie_rec.src.spark_als import ALSReco

if __name__=="__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('userId', type=int, help='ID of user to make recommendations for')
    parser.add_argument('method', type=str, help='Type of KNN collaborative filter method', choices=['item_item', 'user_user', 'als'])
    parser.add_argument('--verbosity', '-v', type=int, choices=[0,1], default=0, help='Verbosity of the code')
    in_args = parser.parse_args()

    print(f"Recommended movies for user {in_args.userId}:")
    if in_args.method in ['item_item', 'user_user']:
        kr = KnnRecommender()
        kr.recommend_movies(in_args.userId, in_args.method)
    elif in_args.method == 'als':
        als_rec = ALSReco()
        als_rec.recommend_movies(in_args.userId)

