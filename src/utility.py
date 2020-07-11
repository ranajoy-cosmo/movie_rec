from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pandas as pd
import os

# Starting a spark session locally
spark = SparkSession \
    .builder \
    .appName("movie-rec-explore") \
    .getOrCreate()

def to_dtype_remove_null(df, col_name, dtype_out):
    df = df.withColumn(col_name, col(col_name).cast(dtype_out))
    df = df.filter(df[col_name].isNotNull())
    return df

def load_and_clean_movie_metadata():
    # Let's read in the movie metadata
    file_path = os.path.join(os.path.dirname(__file__), "../data/movies_metadata.csv")
    movie_met = spark.read.csv(file_path, header=True)

    # Dropping the columns not needed
    movie_met = movie_met.drop(*['homepage', 'imdb_id', 'original_title', 'poster_path', 'video', 'belongs_to_collection', 'revenue', 'tagline', 'overview', 'spoken_languages', 'production_countries'])
    
    # Cleaning the id, popularity, runtime, vote_average, vote_count columns
    # Changing to type int and removing Nan/null rows
    movie_met = to_dtype_remove_null(movie_met, 'id', 'int')
    movie_met = to_dtype_remove_null(movie_met, 'popularity', 'float')
    movie_met = to_dtype_remove_null(movie_met, 'runtime', 'int')
    movie_met = to_dtype_remove_null(movie_met, 'vote_average', 'float')
    movie_met = to_dtype_remove_null(movie_met, 'vote_count', 'int')
    movie_met = to_dtype_remove_null(movie_met, 'budget', 'int')

    # Keeping only the release year
    # Cleaning NaN/Null rows
    movie_met = movie_met.withColumn('year', year(movie_met['release_date'])).drop('release_date')
    movie_met = movie_met.filter(movie_met['year'].isNotNull())

    # Let's just keep the adult=False rows and drop this column.
    movie_met = movie_met.filter(movie_met['adult'] == 'False').drop('adult')

    # Let's just keep the movies that are released.
    movie_met = movie_met.filter(movie_met['status'] == 'Released').drop('status')

    # There are a few duplicate ids and titles
    movie_met = movie_met.dropDuplicates(['id'])
    movie_met = movie_met.fillna("unknown", subset=["original_language"])

    # Transfering the spark dataframe to pandas for some transformations that are better in pandas
    df_mmd = movie_met.toPandas()
    del movie_met

    # Extract genres
    df_mmd.genres = df_mmd.genres.map(lambda col_str: [col_dict['name'] for col_dict in eval(col_str)])

    # Extract production companies
    df_mmd.at[9164, 'production_companies'] = "[]"
    df_mmd.production_companies = df_mmd.production_companies.map(lambda col_str: [col_dict['name'] for col_dict in eval(col_str)])

    # Categories the budgets into bins
    df_mmd['budget_cat'] = pd.cut(df_mmd.budget, bins=[0,1000,100000,10000000,1000000000], labels=["<1k", "1k-100k", "100k-10M", ">10M"])
    df_mmd.budget_cat = df_mmd.budget_cat.cat.add_categories('uncat')
    df_mmd.budget_cat.fillna('uncat', inplace=True)

    # Categorising run times
    df_mmd['runtime_cat'] = pd.cut(df_mmd.runtime, bins=[0,30,120,480,1256], labels=["<30m", "30m-2h", "2h-4h", ">4h"], right=True)
    df_mmd.runtime_cat = df_mmd.runtime_cat.cat.add_categories('uncat')
    df_mmd.runtime_cat.fillna('uncat', inplace=True)

    return df_mmd
