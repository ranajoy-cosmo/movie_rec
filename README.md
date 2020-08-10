<h1 align="center">movie_rec</h3>
<p align="center">
  A movie recommendation engine using Collaborative Filtering techniques
</p>

## Table of Contents
* [About the Project](#about-the-project)
  * [Features](#features)
  * [Dataset](#dataset)
  * [Built with](#built-with)
* [What is Recommendation System?](#what-is-rec-sys)
  * [Collaborative Filtering](#col-filt)
  * [k-Nearest Neighbour classification](#knn-class)
  * [Matrix factorisation](#mat-fac)
* [API](#api)
  * [Setting up](#setup)
  * [Running the script](#run-script)
* [References](#ref)

## About the Project
The goal of this project is to demonstrate a recommender systems, in particular the class of methods called Collaborative Filtering. I 

### Features
* Collaborative filtering
  * User-User recommender with KNN
  * Item-Item recommender with KNN
  * Matrix factorisation using ALS
* Demographic filtering

### Dataset
The dataset is taken from the TMDB dataset on kaggle at [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7).
  
### Built with
* pandas
* numpy
* scipy
* pyspark
* scikit-learn

## What is a Recommendation System?

A recommendation system is what its name suggests. It is a tool that intelligently recommends an object, be it movies or clothes, to an entity, say, an user. A recommendation system is particularly useful for organisations that need to suggest items to its customers. It may do this in several ways: based on features inherent to the item and the user, based on the user's past activity, or based on similarities deduced from the interaction between group of users and items.

### Collaborative Filtering

Collaborative filtering methods are those that rely solely on the past interaction between users and items. In the case of movies, this interaction might be ratings given to movies by users, or a "like" or "dislike". The model takes these user-item interactions and determines the closeness between the given population of users and items, and recommends items that appear the closest to a particular user.

#### k-nearest Neighbours

ABC

## API

To facilitate the use of the recommendation engine, an API is available for execution from the command line.

### Setting up


To setup the dataset run `bash get_data.sh` from the `movie_rec` directory

## Running the scripts

From the `movie_rec` directory simply type the following command

```
python recommend.py <userId> <method>
```

Where you pass the user's Id and the method you'd like to use. Options for the method are `user_user`, `item_item`, `als`

## References
