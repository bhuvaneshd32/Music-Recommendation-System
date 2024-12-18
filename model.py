import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def start_model_cluster():
    global data
    data = pd.read_csv('data.csv')

    # **Clustering Songs with K-Means**
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    global song_cluster_pipeline
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                    ('kmeans', KMeans(n_clusters=20, random_state=31))],
                                    verbose=True)

    X = data.select_dtypes(np.number)

    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    data['cluster_label'] = song_cluster_labels

    # **Build Recommender System**
    '''
    !pip install spotipy
    '''
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    import protected_key

    global sp
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=protected_key.client_id,
                                                            client_secret=protected_key.client_secret))

def find_song(name, year):
    from collections import defaultdict

    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)



def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)

        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values

        if len(song_vector) == 1:
            song_vector = song_vector[0]

        song_vectors.append(song_vector)  

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    from collections import defaultdict

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs(song_list, n_songs=10):
    start_model_cluster()

    from scipy.spatial.distance import cdist
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    
    song_center = get_mean_vector(song_list, data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    return rec_songs[metadata_cols].to_dict(orient='records')

'''
USAGE:

result = recommend_songs([{'name': 'Come As You Are', 'year': 1991},
                          {'name': 'Smells Like Teen Spirit', 'year': 1991},
                          {'name': 'Lithium', 'year': 1992},
                          {'name': 'All Apologies', 'year': 1993},
                          {'name': 'Stay Away', 'year': 1993}], n_songs=10)
'''
