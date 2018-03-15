import pandas as pd
import csv

folder = 'data'
results_folder = 'results'
predictions_file = '/submission-2.csv'
slots_output_file = '/less-readable-predictions-by-slots.csv'
slots_readable_file = '/readable-predictions-by-slots.csv'
# list to collect all print statements to go in a results file
readable_outputs = []

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

def get_readable_slot( slot_id ):
    '''
    Splits slot id into components Round, Region to get readable names
    Returns readable string of the slot
    '''
    print("get readable slot name")
    # rounds lookup
    rounds = {
        'R0': 'First Four',
        'R1': 'First Round',
        'R2': 'Second Round',
        'R3': 'Sweet 16',
        'R4': 'Elite Eight',
        'R5': 'Final Four',
        'R6': 'National Championship',
    }
    # region lookup
    regions = {
        'W': 'East',
        'X': 'Midwest',
        'Y': 'South',
        'Z': 'West',
        'WX': 'East/Midwest',
        'YZ': 'South/West',
        'CH': 'Final'
    }

    # get round from slot id split, first two letters
    slot_round = slot_id[:2]
    # get region from slot id split
    slot_region = ''
    if int(slot_id[1]) < 5:
        # first 4 rounds, it's the third letter
        slot_region = slot_id[2]
    else:
        # rounds 5 and 6, third and fourth letters
        slot_region = slot_id[2:4]
    # return combined region - round string
    return regions[ slot_region ] + ' - ' + rounds[ slot_round ]

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
    readable_outputs.append(
        [
            'Chance that %s beats %s: %f' %
            (teamA,teamB, probA)
        ]
    )
    readable_outputs.append(
        [
            'Chance that %s beats %s: %f' %
            (teamB,teamA, 1 - probA)
        ]
    )
    if 0.39 <= probA <= 0.61:
        readable_outputs.append(
            ['***Close call!']
        )

    # save the highest probability for the next round
    winner_data = []
    # get readable slots from slot ID
    slot_name = get_readable_slot( row.Slot )
    if probA >= 0.5:
        # if ProbAOverB is bigger, then Team A advances
        readable_outputs.append(
            [
                '%s winner: %s (%f)' %
                (slot_name, teamA, probA)
            ]
        )
        winner_data = [name_id_dict[ teamA ], teamA]
    else:
        # else, Team B advances
        readable_outputs.append(
            [
                '%s winner: %s (%f)' %
                (slot_name, teamB, 1 - probA)
            ]
        )
        winner_data = [name_id_dict[ teamB ], teamB]

    # alert if weak team beats strong team
    if slot_row.WeakTeamID == winner_data[0]:
        readable_outputs.append(
            ['*****Upset alert!']
        )
    return winner_data


# slots data
slot_dtypes = {
    'StrongTeamID': str,
    'WeakTeamID': str
}
slots =  pd.read_csv(folder + '/NCAATourneySlots_Detailed_2018.csv', dtype=slot_dtypes)
slots.head()
# model prediction outputs
predictions =  pd.read_csv(results_folder + predictions_file)
predictions.head()

# format pred data to join with slots
pred_formatted = format_pred_outputs( predictions )
pred_formatted.head()

# for each slot
for row in slots.itertuples(index=False):
    print("Joining slots to predictions")
    readable_outputs.append(
        ['------------------------------------------']
    )
    if row.Slot == 'R7WIN':
        readable_outputs.append(
            [
                'Overall 2018 champion: %s' %
                (row.StrongTeamName)
            ]
        )
        break
    # join slots and pred to get higher prob team
    # returns [StrongTeamID, StrongTeamName]
    slot_winner = get_slot_winner(pred_formatted, row)

    # save higher team in appropriate slot for next round
    next_slot = row.NextSlot
    seed_type = row.NextSeed

    # assign the updated winner data in the next slot
    slots.loc[slots['Slot'] == next_slot, [seed_type + 'TeamID',seed_type + 'TeamName']] = slot_winner[0], slot_winner[1]

slots.tail()
# output updated slot data to csv in results folder
print("Write updated slots data to file")
slots.to_csv(results_folder + slots_output_file, index=False)
# create readable results csv
print("Writing %d readable bracket results." % len(slots))
with open(results_folder + slots_readable_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(readable_outputs)
