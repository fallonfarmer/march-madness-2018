
# coding: utf-8
import math
import csv
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, linear_model

base_elo = 1600
team_elos = {}  # Reset each year.
team_stats = {}
X = []
y = []
submission_data = []
folder = 'data'
results_folder = 'results'
prediction_year = 2018

def calc_elo(win_team, lose_team, season):
    winner_rank = get_elo(season, win_team)
    loser_rank = get_elo(season, lose_team)

    """
    This is originally from from:
    http://zurb.com/forrst/posts/An_Elo_Rating_function_in_Python_written_for_foo-hQl
    """
    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank


def initialize_data():
    for i in range(1985, prediction_year+1):
        team_elos[i] = {}
        team_stats[i] = {}


def get_elo(season, team):
    try:
        return team_elos[season][team]
    except:
        try:
            # Get the previous season's ending value.
            team_elos[season][team] = team_elos[season-1][team]
            return team_elos[season][team]
        except:
            # Get the starter elo.
            team_elos[season][team] = base_elo
            return team_elos[season][team]


def predict_winner(team_1, team_2, model, season, stat_fields):
    features = []

    # Team 1
    features.append(get_elo(season, team_1))
    for stat in stat_fields:
        features.append(get_stat(season, team_1, stat))

    # Team 2
    features.append(get_elo(season, team_2))
    for stat in stat_fields:
        features.append(get_stat(season, team_2, stat))

    return model.predict_proba([features])


def update_stats(season, team, fields):
    """
    This accepts some stats for a team and udpates the averages.
    First, we check if the team is in the dict yet. If it's not, we add it.
    Then, we try to check if the key has more than 5 values in it.
        If it does, we remove the first one
        Either way, we append the new one.
    If we can't check, then it doesn't exist, so we just add this.
    Later, we'll get the average of these items.
    """
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        # Make sure we have the field.
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []

        if len(team_stats[season][team][key]) >= 9:
            team_stats[season][team][key].pop()
        team_stats[season][team][key].append(value)


def get_stat(season, team, field):
    if field == 'high_rank' or field == 'power_5':
        try:
            return team_stats[season][team][field][0]
        except:
            return 0
    else:
        try:
            l = team_stats[season][team][field]
            return sum(l) / float(len(l))
        except:
            return 0


def build_team_dict():
    team_ids = pd.read_csv(folder + '/Teams.csv')
    team_id_map = {}
    for index, row in team_ids.iterrows():
        team_id_map[row['TeamID']] = row['TeamName']
    return team_id_map


def build_season_data(all_data):
    # Calculate the elo for every game for every team, each season.
    # Store the elo per season so we can retrieve their end elo
    # later in order to predict the tournaments without having to
    # inject the prediction into this loop.
    print("Building season data.")
    for index, row in all_data.iterrows():
        # Used to skip matchups where we don't have usable stats yet.
        skip = 0

        # Get starter or previous elos.
        team_1_elo = get_elo(row['Season'], row['WTeamID'])
        team_2_elo = get_elo(row['Season'], row['LTeamID'])

        # Add 100 to the home team (# taken from Nate Silver analysis.)
        if row['WLoc'] == 'H':
            team_1_elo += 100
        elif row['WLoc'] == 'A':
            team_2_elo += 100

        # We'll create some arrays to use later.
        team_1_features = [team_1_elo]
        team_2_features = [team_2_elo]

        # Build arrays out of the stats we're tracking..
        for field in stat_fields:
            team_1_stat = get_stat(row['Season'], row['WTeamID'], field)
            team_2_stat = get_stat(row['Season'], row['LTeamID'], field)
            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1

        if skip == 0:  # Make sure we have stats.
            # Randomly select left and right and 0 or 1 so we can train
            # for multiple classes.
            if random.random() > 0.5:
                X.append([row['Season']] + [row['WTeamID']] + team_1_features + [row['LTeamID']] + team_2_features + [0])
            else:
                X.append([row['Season']] + [row['LTeamID']] + team_2_features + [row['WTeamID']] + team_1_features + [1])

        # AFTER we add the current stuff to the prediction, update for
        # next time. Order here is key so we don't fit on data from the
        # same game we're trying to predict.
        if row['WFTA'] != 0 and row['LFTA'] != 0:
            stat_1_fields = {
                'score': row['WScore'],
                'fgp': row['WFGM'] / row['WFGA'] * 100,
                'fga': row['WFGA'],
                'fga3': row['WFGA3'],
                '3pp': row['WFGM3'] / row['WFGA3'] * 100,
                'ftp': row['WFTM'] / row['WFTA'] * 100,
                'or': row['WOR'],
                'dr': row['WDR'],
                'ast': row['WAst'],
                'to': row['WTO'],
                'stl': row['WStl'],
                'blk': row['WBlk'],
                'pf': row['WPF'],
                'form':row['Wform'],
                'power_5':row['WTeam_p5'],
                'high_rank':row['WTeam_rank']
            }
            stat_2_fields = {
                'score': row['LScore'],
                'fgp': row['LFGM'] / row['LFGA'] * 100,
                'fga': row['LFGA'],
                'fga3': row['LFGA3'],
                '3pp': row['LFGM3'] / row['LFGA3'] * 100,
                'ftp': row['LFTM'] / row['LFTA'] * 100,
                'or': row['LOR'],
                'dr': row['LDR'],
                'ast': row['LAst'],
                'to': row['LTO'],
                'stl': row['LStl'],
                'blk': row['LBlk'],
                'pf': row['LPF'],
                'form':row['Lform'],
                'power_5':row['LTeam_p5'],
                'high_rank':row['LTeam_rank']
            }
            update_stats(row['Season'], row['WTeamID'], stat_1_fields)
            update_stats(row['Season'], row['LTeamID'], stat_2_fields)

        # Now that we've added them, calc the new elo.
        new_winner_rank, new_loser_rank = calc_elo(
            row['WTeamID'], row['LTeamID'], row['Season'])
        team_elos[row['Season']][row['WTeamID']] = new_winner_rank
        team_elos[row['Season']][row['LTeamID']] = new_loser_rank

    return X


stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr',
                   'ast', 'to', 'stl', 'blk', 'pf', 'form', 'power_5', 'high_rank']

labels = ['Season', 't1','t1elo', 't1score', 't1fga', 't1fgp', 't1fga3', 't13pp', 't1ftp', 't1or', 't1dr',
                   't1ast', 't1to', 't1stl', 't1blk', 't1pf', 't1form', 't1p5', 't1rank',
                  't2', 't2elo', 't2score', 't2fga',
                  't2fgp', 't2fga3', 't23pp', 't2ftp', 't2or', 't2dr',
                   't2ast', 't2to', 't2stl', 't2blk', 't2pf', 't2form', 't2p5', 't2rank', 't2_win']

initialize_data()



# read data
season_data = pd.read_csv(folder + '/RegularSeasonDetailedResults.csv')
season_data.columns
season_data.shape



tourney_data = pd.read_csv(folder + '/NCAATourneyDetailedResults_2003_2017.csv')
tourney_data.columns
tourney_data.shape



conferences = pd.read_csv('Data/TeamConferences.csv')
conferences.drop('ConfAbbrev', axis = 1, inplace = True)



massey = pd.read_csv('Data/MasseyOrdinals.csv')
preseason_rank = massey[['Season', 'RankingDayNum', 'TeamID', 'OrdinalRank']].groupby(['Season', 'TeamID'], as_index = False).agg(min)



# combine data
frames = [season_data, tourney_data]
all_data = pd.concat(frames)
print(all_data.shape)
all_data.head()
all_data['Wform'] = 1
all_data['Lform'] = 0

all_data = all_data.merge(conferences, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'], how = 'left')
all_data.rename(columns = {'power_5':'WTeam_p5'}, inplace = True)
all_data = all_data.merge(conferences, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'], how = 'left')
all_data.rename(columns = {'power_5':'LTeam_p5'}, inplace = True)
all_data['p5_diff'] = all_data['WTeam_p5'] - all_data['LTeam_p5']

all_data = all_data.merge(preseason_rank, left_on = ['Season', 'WTeamID'], right_on = ['Season', 'TeamID'], how = 'left')
all_data.rename(columns = {'OrdinalRank':'WTeam_rank'}, inplace = True)
all_data = all_data.merge(preseason_rank, left_on = ['Season', 'LTeamID'], right_on = ['Season', 'TeamID'], how = 'left')
all_data.rename(columns = {'OrdinalRank':'LTeam_rank'}, inplace = True)
all_data['rank_diff'] = all_data['WTeam_rank'] - all_data['LTeam_rank']
print(all_data.shape)

# Build the working data.
df = build_season_data(all_data)

preds = pd.DataFrame(df, columns = labels)

print(preds.shape)

print(preds.columns)
drop_cols = ['Season', 't1', 't2', 't2_win']
X = preds.drop(drop_cols, axis = 1)
y = preds['t2_win']

preds[['t2rank', 't1rank']].head()


print("Fitting on %d samples." % len(X))

model = sklearn.linear_model.LogisticRegression()
# model = RandomForestClassifier(max_depth = 2)

# Check accuracy.
print("Doing cross-validation.")
print(cross_validation.cross_val_score(model, np.array(X), np.array(y), cv=10, scoring='accuracy').mean())

model.fit(X, y)

# Now predict tournament matchups.
print("Getting teams.")
seeds = pd.read_csv(folder + '/NCAATourneySeeds.csv')
# for i in range(2016, 2017):
tourney_teams = []
for index, row in seeds.iterrows():
    if row['Season'] == prediction_year:
        tourney_teams.append(row['TeamID'])

# Build our prediction of every matchup.
print("Predicting matchups.")
tourney_teams.sort()
for team_1 in tourney_teams:
    for team_2 in tourney_teams:
        if team_1 < team_2:
            prediction = predict_winner(
                team_1, team_2, model, prediction_year, stat_fields)
            label = str(prediction_year) + '_' + str(team_1) + '_' + str(team_2)
            submission_data.append([label, prediction[0][0]])

# Write the results.
print("Writing %d results." % len(submission_data))
with open(results_folder + '/submission-2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'pred'])
    writer.writerows(submission_data)

# Now so that we can use this to fill out a bracket, create a readable
# version.
print("Outputting readable results.")
team_id_map = build_team_dict()
readable = []
less_readable = []  # A version that's easy to look up.
for pred in submission_data:
    parts = pred[0].split('_')
    less_readable.append(
        [team_id_map[int(parts[1])], team_id_map[int(parts[2])], pred[1]])
    # Order them properly.
    if pred[1] > 0.5:
        winning = int(parts[1])
        losing = int(parts[2])
        proba = pred[1]
    else:
        winning = int(parts[2])
        losing = int(parts[1])
        proba = 1 - pred[1]
    readable.append(
        [
            '%s beats %s: %f' %
            (team_id_map[winning], team_id_map[losing], proba)
        ]
    )
with open(results_folder + '/readable-predictions-2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(readable)
with open(results_folder + '/less-readable-predictions-2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(less_readable)
