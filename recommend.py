import argparse
import os
from movie_rec.src.knn_reco import KnnRecommender

if __name__=="__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('userId', type=int, help='ID of user to make recommendations for')
    parser.add_argument('method', type=str, help='Type of KNN collaborative filter method', choices=['item_item', 'user_user'])
    parser.add_argument('--verbosity', '-v', type=int, choices=[0,1], default=0, help='Verbosity of the code')
    in_args = parser.parse_args()

    kr = KnnRecommender()
    print(f"Recommended movies for user {in_args.userId}:")
    kr.recommend_movies(in_args.userId, in_args.method)

