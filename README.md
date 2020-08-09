<h1 align="center">movie_rec</h3>
<p align="center">
  A movie recommendation engine using Collaborative Filtering techniques
</p>

## Table of Contents
* [About the Project](#about-the-project)
  * [Features](#features)
  * [Built with](#built-with)
* [What is Collaborative Filtering](#what-is-col_filt)

## About the Project
The goal of this project is to demonstrate a recommender systems, in particular the class of methods called Collaborative Filtering. I 

### Features
* Collaborative filtering
  * User-User recommender with KNN
  * Item-Item recommender with KNN
  * Matrix factorisation using ALS
* Demographic filtering
  
### Built with
* pandas
* numpy
* scipy
* pyspark
* scikit-learn

## What is Collaborative Filtering
A recommender system is particularly useful for organisations that need to suggest items to its customers. It may do this in several intelligent ways that based on features inherent to the item and the customer, based on the user's past activity, or based on similarities deduced from the interaction between group of users and items.

Collaborative filtering methods are those that rely solely on the past interaction between users and items 

## Getting the dataset

The dataset is taken from the TMDB dataset on kaggle
To setup the dataset run `bash get_data.sh` from the `movie_rec` directory

## Running the scripts

From the `movie_rec` directory simply type the following command

```
python recommend.py <userId> <method>
```

Where you pass the user's Id and the method you'd like to use. Options for the method are `user_user`, `item_item`, `als`
