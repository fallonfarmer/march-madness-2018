# march-madness-2018

This project uses the data provided by the Kaggle Machine Learning Mania challenge
and generates predictions for March Madness brackets.
More about the competition:
https://www.kaggle.com/c/mens-machine-learning-competition-2018

## Feature Engineering
- recent performance
- `power_5`
- `preseason_rank`
- `champ`
validated with `ConferenceChamps_2018.csv` using data from [cbssports.com](https://www.cbssports.com/college-basketball/news/selection-sunday-show-2018-ncaa-tournament-conference-champions-and-automatic-bids/)

## Ideas for future work

##### Feature Engineering
- regular season champion
- regular season wins
- regular season win/loss ratio
- regular season average points per game
- regular season average points allowed
- tenure of coach
- prob of win without X player, based on historical play by play of games won without that player, or played very little
- distance of traveled for game
- days since last game

##### Feature selection
- correlations / sploms of variables
- stepwise model
- feature importance (RandomForestClassifier)
- feature impact (LogisticRegression)

##### Model training
- 10,000 simulations
- XGBoost
- KNN
- Keras neural net
- Ensembles

## References:

##### Feature generation and selection:
- [Matt Harvey of CoastAI 2016 model](https://blog.coast.ai/this-is-how-i-used-machine-learning-to-accurately-predict-villanova-to-win-the-2016-march-madness-ba5c074f1583)
- [Adepsh Pande 2017 model](https://adeshpande3.github.io/Applying-Machine-Learning-to-March-Madness)
- [Ideas for data](https://www.techrepublic.com/article/march-madness-5-data-sources-that-could-predict-the-2017-ncaa-championship/)

##### Elo scores:
- [fivethirtyeight methodology](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/)
- [Kaggle elo in python example](https://www.kaggle.com/kplauritzen/elo-ratings-in-python/notebook)
- [Kaggle elo example 2](https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings)
