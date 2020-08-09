<h1 align="center">movie_rec</h3>
<p align="center">
  A movie recommendation system demonstrating Collaborative Filtering techniques
</p>

## Table of Contents
* [About the Project](#about-the-project)
  * [Features](#features)
  * [Built with](#built-with)
* [What is Collaborative Filtering](#what-is-col_filt)

## About the Project
The goal of this project is to demonstrate some popular collborative filtering methods for making movie recommendations

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

## Getting the dataset

The dataset is taken from the TMDB dataset on kaggle
To setup the dataset run `bash get_data.sh` from the `movie_rec` directory

## Running the scripts

From the `movie_rec` directory simply type the following command

```
python recommend.py <userId> <method>
```

Where you pass the user's Id and the method you'd like to use. Options for the method are `user_user`, `item_item`, `als`
