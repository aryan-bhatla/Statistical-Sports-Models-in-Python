#----------------------------------------------------------------------------------------------------- # 
# Modules
#----------------------------------------------------------------------------------------------------- # 
import numpy as np 
import pandas as pd

from scipy.optimize import minimize


#----------------------------------------------------------------------------------------------------- # 
# Maximum Likelihood Estimation
#----------------------------------------------------------------------------------------------------- # 
def log_likelihood_maximization(params: list[float], data_df: pd.DataFrame, team_indices: dict[str, int]) -> float:
    '''

        Parameters: 
            params (list[float]): List containing the Home Field Advantage (HFA) and team ratings
            data_df (pd.DataFrame): Pandas DataFrame containing match data
            team_indices (dict[str, int]): Dictionary mapping team names to their respective indices in the ratings list
        
        Returns: 
            - log_likelihood (float): The negative log likelihood value

    '''
    # Unpack Home Field Advantage (HFA) and team ratings from params
    HFA, *ratings = params

    # Initialize log likelihood
    log_likelihood = 0

    # Iterate through each row in the DataFrame
    for _, row in data_df.iterrows():

        # Get indices for home and away teams
        home_team_index = team_indices[row['Home Team']]
        away_team_index = team_indices[row['Away Team']]

        # Get ratings for home and away teams
        home_rating = ratings[home_team_index]
        away_rating = ratings[away_team_index]

        # Calculate the logistic function for the current game
        logistic_function = 1 / (1 + np.exp(-(HFA + home_rating - away_rating)))

        # Update log likelihood based on the game result
        if row['Home Pts'] > row['Away Pts']:
            log_likelihood += np.log(logistic_function)
        else:
            log_likelihood += np.log(1 - logistic_function)

    # Return negative likelihood as this will be minimized
    return -log_likelihood


#----------------------------------------------------------------------------------------------------- # 
# Bradley Terry Model
#----------------------------------------------------------------------------------------------------- # 
def bradley_terry(date: list[str], away_teams: list[str], away_points: np.ndarray, home_teams: list[str], home_points: np.ndarray):
    '''

        Parameters: 
            date (list(str)): List of dates for matches that were played 
            away_teams (list(str)): List of names of away teams 
            away_points np.ndarray: Numpy array of number of points scored by away teams 
            home_teams (list(str)): List of names of home teams 
            home_points np.ndarray: Numpy array of number of points scord by home teams 
        
        Returns: 
            data_df (pd.DataFrame): Pandas DataFrame containing final results 

    '''
    # Create dataframe to store data 
    data = {'Date': date, 'Away Team': away_teams, 'Away Pts': away_points, 'Home Team': home_teams, 'Home Pts': home_points}
    data_df = pd.DataFrame(data = data)

    # Create a mapping of team names to indices
    team_names = list(set(away_teams) | set(home_teams))
    team_indices = {team_name: index for index, team_name in enumerate(team_names)}

    # Define initial ratings and HFA
    initial_ratings = [1 for _ in team_names]
    initial_HFA = 0.3
    initial_params = [initial_HFA] + initial_ratings

    # Perform optimization using maximum likelihood estimation
    result = minimize(log_likelihood_maximization, initial_params, args=(data_df, team_indices))

    # Extract optimized parameters
    optimized_params = result.x
    optimized_HFA = optimized_params[0]
    optimized_team_ratings = optimized_params[1:]

    # Update the DataFrame using optimized parameters
    for index, row in data_df.iterrows():

        # Ratings for each team 
        home_team_rating = optimized_team_ratings[team_indices[row['Home Team']]]
        away_team_rating = optimized_team_ratings[team_indices[row['Away Team']]]

        # Calculate total points
        total_points = row['Home Pts'] + row['Away Pts']

        # Calculate margin of victory
        margin_of_victory = row['Home Pts'] - row['Away Pts']

        # Logistic function = 1 / (1 + np.exp(-(HFA + HT - AT)))
        logistic_function = 1 / (1 + np.exp(-(optimized_HFA + home_team_rating - away_team_rating)))

        # If home team wins -> game_result = 1 and result_function = logistic_function 
        if margin_of_victory > 0: 
            game_result = 1
            result_function = logistic_function 

        # If home team loses -> game_result = 0 and result_function = 1 - logistic_function
        else: 
            game_result = 0
            result_function = 1 - logistic_function
    
        # Calculate log results
        log_result = np.log(result_function)

        # Append columns to dataframe
        data_df.at[index, 'Game Total'] = total_points
        data_df.at[index, 'Home MOV'] = margin_of_victory
        data_df.at[index, 'Logistic Function'] = logistic_function
        data_df.at[index, 'Game Result'] = game_result
        data_df.at[index, 'Result Function'] = result_function
        data_df.at[index, 'LN of Result %'] = log_result

    return data_df 


#----------------------------------------------------------------------------------------------------- # 
# Example usage
#----------------------------------------------------------------------------------------------------- # 
read_data = pd.read_excel('Excel Models/Bradley Terry Model.xlsx', usecols=['Date', 'Away Team', 'Away Pts', 'Home Team', 'Home Pts'], skiprows=3)
dataframe = bradley_terry(read_data['Date'].tolist(), 
                          read_data['Away Team'].tolist(), 
                          np.array(read_data['Away Pts']), 
                          read_data['Home Team'].tolist(), 
                          np.array(read_data['Home Pts']))
print(dataframe)
