# Song Popularity Prediction using Spotify API

## Project Overview

This project aims to predict whether a song is likely to become popular using audio features retrieved from the Spotify API. We use historical data on the most streamed songs on Spotify in 2024, extract audio features for these songs and additional test playlists, and build a machine learning model to classify the songs as potentially popular or not.

The model uses features such as danceability, energy, loudness, and tempo, among others, to determine whether a song has a high chance of becoming a hit.

## Project Structure

song_recommendation_project/
│
├── data/
│   ├── raw/                           # Raw datasets like 'Most Streamed Spotify Songs 2024.csv'
│   │   └── Most_Streamed_Spotify_Songs_2024.csv
│   ├── processed/                     # Cleaned and processed audio feature data
│   │   └── audio_features.csv         # CSV with audio features used for training and prediction
│   └── external/                      # Any external data, like playlists used for testing
│
├── notebooks/
│   ├── audio_features_creation.ipynb  # Notebook for fetching audio features via Spotify API
│   └── most_streamed_songs.ipynb      # EDA, model training, evaluation, and saving the model
│
├── src/                               # Source code for the project
│   ├── __init__.py                    # Marks this folder as a Python package
│   ├── data_loader.py                 # Script to load and preprocess data
│   ├── feature_engineering.py         # Scripts for feature extraction, transformations, etc.
│   ├── model.py                       # Training, evaluation, and prediction functions
│   └── spotify_api.py                 # Code that interacts with the Spotify API
│
├── models/
│   └── model.pkl                      # Trained model saved for predictions
│
├── results/
│   ├── eda_visuals/                   # PNG files of graphs generated during EDA
│   │   └── feature_distribution.png
│   └── model_performance.png          # PNG files of model performance graphs
│
├── requirements.txt                   # Dependencies and libraries used in the project
├── README.md                          # Overview and documentation of the project
└── .gitignore                         # To ignore unnecessary files (e.g., large CSVs, model.pkl, etc.)



## Data

1. **Most Streamed Spotify Songs 2024.csv**: This dataset contains a list of the most streamed songs on Spotify in 2024. It includes information on song name, artist, and streaming metrics.
2. **Audio Features CSVs**: The audio features were obtained using the Spotify API for both the most streamed songs and additional playlists for testing the model.

## Notebooks

1. **`audio_features_creation.ipynb`**: This notebook connects to the Spotify API to retrieve audio features such as danceability, energy, and tempo for each song. These features serve as input data for the model.
   
   **Credit**: The method for extracting audio features was inspired by Juan de Dios Santos in his blog post ["Is My Spotify Music Boring?"](https://towardsdatascience.com/is-my-spotify-music-boring-an-analysis-involving-music-data-and-machine-learning-47550ae931de).

2. **`most_streamed_songs.ipynb`**: This notebook contains the full analysis pipeline, starting with Exploratory Data Analysis (EDA) of the dataset, feature engineering, model training, evaluation, and saving the model.

## Model

The model is a classification model that predicts whether a song will become popular based on its audio features. We used a range of machine learning algorithms and selected the one with the best performance for our final model.

- **Model Type**: Logistic Regression
- **Metrics**: Accuracy, Precision, Recall

The final model is saved as `model.pkl` and can be used for future predictions.

## Results

The results of the analysis include visualizations of the feature distributions and the performance of the model in terms of accuracy and recall.

Key findings:
- Danceability and energy are strong indicators of song popularity.
- Popular songs tend to have a slightly faster tempo compared to less popular songs.

## How to Run the Project

### Step 1: Clone the repository
bash
git clone https://github.com/your_username/song-recommendation-project.git

### Step 2: Install the required dependencies
Make sure you have Python 3.7+ installed, then install the dependencies listed in requirements.txt:

bash
pip install -r requirements.txt

### Step 3: Run the notebooks

bash
jupyter notebook notebooks/audio_features_creation.ipynb
jupyter notebook notebooks/most_streamed_songs.ipynb

### Step 4: Use the trained model
To use the trained model to make predictions on new songs, load model.pkl and use it with new audio features from the Spotify API.

### Dependencies
The project depends on the following Python libraries:

spotipy for accessing the Spotify API
pandas for data manipulation
scikit-learn for model training and evaluation
matplotlib and seaborn for visualizations
jupyter for running notebooks
To install all dependencies, run:

bash
pip install -r requirements.txt

## Acknowledgments
Special thanks to Juan de Dios Santos for his insightful blog post, "Is My Spotify Music Boring?", which inspired the audio feature extraction process used in this project. The code used for this part is his authorship.

Thanks to the Spotify API for providing the audio features used for analysis.