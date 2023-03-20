# from fileinput import filename
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib
import joblib


def genre_predictor():
    global music_data
    global md_model
    path1 = Path()

    # creating an instance of the class; a model object
    md_model = DecisionTreeClassifier()

    for filename in path1.glob('*.csv'):
        # importing the data / loading .csv file /
        music_data = pd.read_csv(filename)

        # md_model.fit(input_data_of_md, output_data_of_md)
        # prediction1 = md_model.predict([[21, 1], [22, 0]])

        # defining the input and output data of the music_data
        input_data_of_md = music_data.drop(columns='genre')
        output_data_of_md = music_data['genre']

        # splitting the data set into train and test
        traininput, testinput, trainout, testout = train_test_split(input_data_of_md, output_data_of_md, test_size=0.25)

        # training the model using the training input/output data sets
        md_model.fit(traininput, trainout)

        # saving the trained model
        joblib.dump(md_model, 'music-genre-model.joblib')

        # once we've saved the trained model we can directly load it and go directly to predicting
        md_model = joblib.load('music-genre-model.joblib')

        # making predictions on the input test data sets based on the training data
        prediction1 = md_model.predict(testinput)

        # to check accuracy; of prediction compared to the output test data set
        accuracy1 = accuracy_score(testout, prediction1)
        print(f'this is the prediction, {prediction1} ----- and this is the accuracy, {accuracy1}')

        print(music_data)

    
genre_predictor()
