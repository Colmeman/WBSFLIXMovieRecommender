import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic, accuracy, SVD, NMF
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from collections import defaultdict
import numpy as np
import pickle

csv_file_path_movies = r'C:\Users\gusta\Documents\Data Science Bootcamp\Data Science\Chapter 8\Data\ml-latest-small\movies.csv'
csv_file_path_ratings = r'C:\Users\gusta\Documents\Data Science Bootcamp\Data Science\Chapter 8\Data\ml-latest-small\ratings.csv'
csv_file_path_tags = r'C:\Users\gusta\Documents\Data Science Bootcamp\Data Science\Chapter 8\Data\ml-latest-small\tags.csv'
csv_file_path_links = r'C:\Users\gusta\Documents\Data Science Bootcamp\Data Science\Chapter 8\Data\ml-latest-small\links.csv'
movies_df = pd.read_csv(csv_file_path_movies)
ratings_df = pd.read_csv(csv_file_path_ratings)
tags_df = pd.read_csv(csv_file_path_tags)
links_df = pd.read_csv(csv_file_path_links)
rating_count_df = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
model = pickle.load(open(r'C:\Users\gusta\Documents\Data Science Bootcamp\Data Science\Chapter 8\trained_pipe_knn.sav', 'rb'))

def popularity_recommender():
    n = input('Enter number of movies do you want to display: ')
    movies_info_columns = ['movieId', 'title', 'genres']
    filtered_movies = rating_count_df[(rating_count_df['mean'] > 4) & (rating_count_df['count'] > 100)]
    filtered_movies_sorted = filtered_movies.sort_values(by='mean', ascending=False)
    filtered_moviesId = filtered_movies_sorted['movieId'].values
    filtered_moviesId_mask = ratings_df['movieId'].isin(filtered_moviesId)
    filtered_movies_info = ratings_df.loc[filtered_moviesId_mask, 'movieId'].drop_duplicates()
    mean_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
    filtered_movies_info = pd.merge(filtered_movies_info, movies_df, on='movieId', how='left')
    filtered_movies_info = pd.merge(filtered_movies_info, mean_ratings, on='movieId', how='left')
    filtered_movies_info.rename(columns={'rating': 'mean_rating'}, inplace=True)
    filtered_movies_info = filtered_movies_info.set_index('movieId').loc[filtered_moviesId].reset_index()
    return(filtered_movies_info.head(int(n)))

def Top_N_recomendations():
  movie_titles = movies_df['title'].tolist()
  name = input('Enter the name of the movie: ')
  matching_titles = [title for title in movie_titles if name.lower() in title.lower()]
  if matching_titles:
    selected_movie_title = matching_titles[0]
    result = movies_df[movies_df['title'] == selected_movie_title]['movieId']
    if not result.empty:
        movie_id = result.iloc[0]
    else:
        print(f"No movie found with the title '{selected_movie_title}'.")
  else:
    print(f"No movie found containing '{name}'.")
  n = input('Enter the amount of similar movies you want to find: ')
  user_movie_matrix = pd.pivot_table(data=ratings_df,
                                  values='rating',
                                  index='userId',
                                  columns='movieId',
                                  fill_value=0)
  movies_cosines_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T),
                                    columns=user_movie_matrix.columns,
                                    index=user_movie_matrix.columns)
  movies_cosines_df = pd.DataFrame(movies_cosines_matrix[movie_id])
  movies_cosines_df = movies_cosines_df.rename(columns={movie_id: name})
  movies_cosines_df = movies_cosines_df[movies_cosines_df.index != movie_id]
  movies_cosines_df = movies_cosines_df.sort_values(by=name, ascending=False)
  no_of_users_rated_both_movies = [sum((user_movie_matrix[movie_id] > 0) & (user_movie_matrix[i] > 0)) for i in movies_cosines_df.index]
  movies_cosines_df['users_who_rated_both_movies'] = no_of_users_rated_both_movies
  movies_cosines_df = movies_cosines_df[movies_cosines_df["users_who_rated_both_movies"] > 10]
  movies_cosines_df = movies_cosines_df.rename(columns={name: f'{name}_cosine'})
  movies_info_columns = ['movieId', 'title', 'genres']
  top_n_recomendations = (movies_cosines_df
                              .head(int(n))
                              .reset_index()
                              .merge(movies_df.drop_duplicates(subset='movieId'),
                                     on='movieId',
                                     how='left')
                              [movies_info_columns + [f'{name}_cosine',	'users_who_rated_both_movies']] 
                              )
  return(top_n_recomendations)

def get_top_n_2():
  user_recommendations = []
  user_id_str = input('Enter the user ID: ')
  user_id = int(user_id_str)
  n = input('Enter de number of movies you want to display: ')
  unrated_movies=ratings_df[~ratings_df.movieId.isin(
                                      ratings_df[ratings_df.userId==user_id].movieId.unique()
                                      )
                  ].movieId.unique()
  for iid in unrated_movies:
    _, _, _, est, _ = model.predict(user_id,iid)
    user_recommendations.append((iid, est))
  ordered_recommendations = sorted(user_recommendations, key=lambda x: x[1], reverse=True)
  ordered_recommendations_top_n = ordered_recommendations[:int(n)]  
  top_n = ordered_recommendations_top_n
  tuples_df = pd.DataFrame(top_n, columns=["movieId", "estimated_rating"])
  reduced_df = movies_df.drop_duplicates(subset='movieId').copy()
  tuples_df_expanded = tuples_df.merge(reduced_df, on="movieId", how='left')
  tuples_df_expanded = tuples_df_expanded.sort_values(by="title")
  return(tuples_df_expanded) 
get_top_n_2()

st.title("WBSFLIX MOVIE RECOMMENDER")
st.write("...")

st.header('Most Popular Movies')

st.header('Since you watched ...')

st.header('We though you might like these movies')

