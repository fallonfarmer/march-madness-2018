# march-madness-2018

This project uses the data provided by the Kaggle Machine Learning Mania challenge
and generates predictions for March Madness brackets.
More about the competition:
https://www.kaggle.com/c/mens-machine-learning-competition-2018

## Feature Engineering
- `power_5`: used indicator directly from `TeamConferences.csv`
- `preseason_rank`: found the first ranking using minimum day of ranking from `MasseyOrdinals.csv`
- `champ`: used `ConferenceTourneyGames.csv` to find the winning team for the last day of conference play for given season, then converted to binary for if the team was the champ that season
  - validated with `ConferenceChamps_2018.csv` using data from [cbssports.com](https://www.cbssports.com/college-basketball/news/selection-sunday-show-2018-ncaa-tournament-conference-champions-and-automatic-bids/)

## References:

##### Feature generation and selection:
- [Matt Harvey of CoastAI 2017 model](https://blog.coast.ai/this-is-how-i-used-machine-learning-to-accurately-predict-villanova-to-win-the-2016-march-madness-ba5c074f1583)
- [Adepsh Pande 2017 model](https://adeshpande3.github.io/Applying-Machine-Learning-to-March-Madness)
- [Ideas for data](https://www.techrepublic.com/article/march-madness-5-data-sources-that-could-predict-the-2017-ncaa-championship/)

##### Elo scores:
- [fivethirtyeight methodology](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/)
- [Kaggle elo in python example](https://www.kaggle.com/kplauritzen/elo-ratings-in-python/notebook)
- [Kaggle elo example 2](https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings)
