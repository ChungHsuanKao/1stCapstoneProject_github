# Problem statment
I’m interested in figuring out whether those stereotypical thinkings are right or not based on a
statistician or a data scientist angles. Can famous actors, director, or both bring the highest
revenue? How social media and networking service company ex: Facebook influence the movie
industry? Is a high IMDB score a good sign for movie companies that they are gonna make
money for sure? All in all, I want to know what are the most important features for a successful
movie? Can we actually create a model to predict the profit for a movie? Therefore, the movie
company can use the model to estimate their strategy in producing a movie.

# Strategy
As the prediction is movie’s revenue (gross - budget), I would like to use and compare both
classification (Classify revenues in to different groups) and regression (Use the number from the
dataset directly) approaches on this supervised data. In terms of predictors, I want to use
director’s names, actors names, IMDB score, facebook’s likes (actors, directors, and movies),
number of critical reviews on imdb , number of users reviews, Number of people who voted for
the movie, and content rating of the movie.
For the training data, I am planning to split the whole dataset into two parts: ⅓ of dataset is
testing data and ⅔ of dataset is training data.

# Datasets
* movie_metadata.csv: Original dataset from https://data.world/data-society/imdb-5000-movie-dataset
* final_wrangle.csv: csv file after 1st clean-up 
* final_pre.csv: csv file after removing outliers and features having high correlation with other features
* final_eda.csv: csv file after EDA analysis and 2nd clean-up (ready for modeling)

# Web scrapped datasets
* imdb_month.json: json file includes movie's month and date from IMDB website
* imdb_budget.json: json file includes movie's budget from IMDB website
* imdb_content_rating.json: json file includes movie's content rating from IMDB website
* imdb_gross.json: json file includes movie's gross from IMDB website
* imdb_titleyear.json: json file includes movie's title year from IMDB website

# module.py
The file includes functions for EDA analysis (permutation, bootstrapping, simulation, statistical analysis for p-value).

# iPython notebooks
* **MovieRevenuePrediction_Data collection and data wrangling.ipynb**: The file shows how I transform the raw data into a data that I can perform prediction
* **MovieRevenuePrediction_Exploratory data analysis.ipynb**: The file shows the variables I found that are particularly significant in terms of explaining the answer to my project question.
* **MovieRevenuePrediction_machine learning.ipynb**: The file shows how I perform the classification prediciton.
