<h1 align="center">movie_rec</h3>
<p align="center">
  A movie recommendation system demonstrating some popular recommendation systems
</p>

## Features
* Collaborative filtering
  * User-User recommender with KNN
  * Item-Item recommender with KNN
  * SVD decomposition (coming soon)
  
## Powered by
* pandas
* numpy
* scipy
* pyspark
* scikit-learn

## Getting the dataset

The dataset is taken from the TMDB dataset on kaggle
To setup the dataset run the following script

```
mkdir data
cd data
kaggle datasets download -d rounakbanik/the-movies-dataset
unzip the-movies-dataset.zip
```

## Running the scripts

From the `movie_rec` directory simply type the following command

```
python recommend.py <userId> <method>
```

Where you pass the user's Id and the method you'd like to use. Options for the method are `user_user` and `item_item`