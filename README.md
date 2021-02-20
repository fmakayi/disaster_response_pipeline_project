# Disaster Response Pipeline Project

## Project Components
1. ETL Pipeline
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
3. Flask Web App
Display visualization in a flask web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements

Must give credit to FigureEight for the data set.

The link to the repository can be found [here](https://github.com/fmakayi/disaster_response_pipeline_project.git)
