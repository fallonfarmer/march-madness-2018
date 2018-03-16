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
- regular season champion, conf coach of the year, conf player of the year
https://en.wikipedia.org/wiki/2017%E2%80%9318_NCAA_Division_I_men%27s_basketball_season#Conference_winners_and_tournaments
- regular season wins
- regular season win/loss ratio, in conf and overall
https://en.wikipedia.org/wiki/2017%E2%80%9318_NCAA_Division_I_men%27s_basketball_season#Conference_standings
- regular season average points per game
- regular season average points allowed
- tenure of coach
- prob of win without X player, based on historical play by play of games won without that player, or played very little
- distance of traveled for game
- days since last game
- [regular season upsets](https://en.wikipedia.org/wiki/2017%E2%80%9318_NCAA_Division_I_men%27s_basketball_season#Upsets)
- historical wiki data https://en.wikipedia.org/wiki/Category:NCAA_Division_I_men%27s_basketball_seasons
- nearest distance from historical tournament finalists (small school upsets)
- binary for if the team was in the championship game or final four in the past 4 years
- average number of rounds they advanced in the NCAA tournament over the past 4-6 years

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
