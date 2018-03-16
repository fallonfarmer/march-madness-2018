## Ideas for future work

##### Feature Engineering
- [historical wiki data](https://en.wikipedia.org/wiki/Category:NCAA_Division_I_men%27s_basketball_seasons)
- [regular season champion, conf coach of the year, conf player of the year](https://en.wikipedia.org/wiki/2017%E2%80%9318_NCAA_Division_I_men%27s_basketball_season#Conference_winners_and_tournaments)
- regular season wins
- [regular season win/loss ratio, in conf and overall](https://en.wikipedia.org/wiki/2017%E2%80%9318_NCAA_Division_I_men%27s_basketball_season#Conference_standings)
- regular season average points per game
- regular season average points allowed
- recent performance
- tenure of coach
- prob of win without X player, based on historical play by play of games won without that player, or played very little
- distance of traveled for game
- days since last game
- [regular season upsets](https://en.wikipedia.org/wiki/2017%E2%80%9318_NCAA_Division_I_men%27s_basketball_season#Upsets)
- nearest distance from historical tournament finalists (small school upsets)
- binary for if the team was in the championship game or final four in the past 4 years
- average number of rounds they advanced in the NCAA tournament over the past 4-6 years

##### Feature selection
- correlations / sploms of variables
- stepwise model
- feature importance (RandomForestClassifier)
- feature impact (LogisticRegression coefficients)
- PCA

##### Model training
- 10,000 simulations
- XGBoost
- KNN
- Keras neural net
- Ensembles
