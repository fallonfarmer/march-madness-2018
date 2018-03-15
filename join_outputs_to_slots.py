import pandas as pd

folder = 'data'
results_folder = 'results'
prediction_year = 2018

def format_pred_outputs( pred ):
    '''
    Take outputs from model.py and format them for joining to slot data
    Return pred dataframe with added columns Year, TeamAID, TeamBID, PredIDConcat, ProbAOverB
    '''
    print("format prediction outputs")
    formatted_data = []
    headers = [ 'Year', 'TeamAID', 'TeamBID', 'PredIDConcat', 'ProbAOverB' ]
    # for each row in pred data
    for index, row in pred.iterrows():
        # split id column, format is YYYY_TeamAID_TeamBID
        id_split = row.id.split( '_' )

        # get year
        year = id_split[0]
        # get team A (first one alphabetically)
        teamA = id_split[1]
        # get team B (second one alphabetically)
        teamB = id_split[2]

        # get win probability of team A beating team B
        prob = row.pred

        # save to new dataframe
        row_data = [year, teamA, teamB, teamA + "_" + teamB, prob]
        formatted_data.append(row_data)

    # return dataframe
    return pd.DataFrame(formatted_data, columns=headers)

def get_slot_winner( pred_df, slot_row ):
    '''
    For the given round, get the teams in the given slots
    and then join to the team's probability of beating that other one
    Return list of StrongTeamID, StrongTeamName
    where StrongTeam = the team with the higher win probability
    '''
    print("get slot winner")
    # create lookup of strong team ID by strong team name, also same for weak team
    name_id_dict = {}
    name_id_dict[slot_row.StrongTeamName] = slot_row.StrongTeamID
    name_id_dict[slot_row.WeakTeamName] = slot_row.WeakTeamID

    # concatenate them in the alphabetical order of StrongTeamName vs WeakTeamName
    # to match the unique pred id, which is alpha sorted
    alpha_teams = sorted([slot_row.StrongTeamName, slot_row.WeakTeamName])
    teamA = alpha_teams[0]
    teamB = alpha_teams[1]
    SlotIDConcat = name_id_dict[ teamA ] + "_" + name_id_dict[ teamB ]

    # get the ProbAOverB from pred_data
    # where SlotIDConcat equals PredIDConcat
    probA = pred_df[pred_df['PredIDConcat'] == SlotIDConcat]['ProbAOverB'].iloc[0]

    # save the highest probability for the next round
    winner_data = []
    if probA >= 0.5:
        # if ProbAOverB is bigger, then Team A advances
        winner_data = [name_id_dict[ teamA ], teamA]
    else:
        # else, Team B advances
        winner_data = [name_id_dict[ teamB ], teamB]

    return winner_data


if __name__ == "__main__":
    # slots data
    slot_dtypes = {
        'StrongTeamID': str,
        'WeakTeamID': str
    }
    slots =  pd.read_csv(folder + '/NCAATourneySlots_Detailed_2018.csv', dtype=slot_dtypes)
    slots.head()
    # model prediction outputs
    predictions =  pd.read_csv(results_folder + '/submission.csv')
    predictions.head()

    # format pred data to join with slots
    pred_formatted = format_pred_outputs( predictions )
    pred_formatted.head()

    # for each slot
    for row in slots.itertuples(index=False):
        print("Joining slots to predictions")
        # join slots and pred to get higher prob team
        # returns [StrongTeamID, StrongTeamName]
        slot_winner = get_slot_winner(pred_formatted, row)

        # save higher team in appropriate slot for next round
        next_slot = row.NextSlot
        seed_type = row.NextSeed

        # assign the updated winner data in the next slot
        slots.loc[slots['Slot'] == next_slot, [seed_type + 'TeamID',seed_type + 'TeamName']] = slot_winner[0], slot_winner[1]

    slots.tail()
