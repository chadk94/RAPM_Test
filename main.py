import sqlite3
import streamlit as st
import app
# Set page config
from tqdm import tqdm
import nba_api.live.nba.endpoints
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playbyplayv2, leaguegamefinder, commonteamroster
from nba_api.stats.static import teams
from sklearn.linear_model import Ridge
import nba_on_court as noc
from collections import defaultdict
import time

# Configuration
SEASON = '2024'  # WNBA season
LEAGUE_ID = '10'  # WNBA league ID
LAMBDA = 2000  # Ridge regression regularization parameter (tune this)


def get_wnba_teams():
    """Fetch all WNBA teams for the season."""
    return teams.get_wnba_teams()


def get_game_ids(season, league_id):
    """Fetch game IDs for the WNBA season."""
    game_finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable=league_id,
        season_type_nullable='Regular Season'
    )
    games = game_finder.get_data_frames()[0]
    game_ids = games['GAME_ID'].unique()
    return game_ids


def get_team_roster(team_id, season):
    """Fetch team roster for a given team and season."""
    roster = commonteamroster.CommonTeamRoster(
        team_id=team_id,
        season=season,
        league_id_nullable=LEAGUE_ID
    )
    return roster.get_data_frames()[0]


'''def get_play_by_play(game_id):
    """Fetch play-by-play data for a game."""
    time.sleep(5)
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id).play_by_play.get_data_frame()
    pbp_with_players=noc.players_on_court(pbp)
    return pbp_with_players
'''


def get_play_by_play(game_id, db_path='pbp_data.db'):
    """
    Fetch play-by-play data for a game using pandas built-in SQL methods.
    Only fetches new data if the game_id doesn't exist in the database.

    Args:
        game_id: The game ID to fetch data for
        db_path: Path to the SQLite database

    Returns:
        DataFrame with play-by-play data including players on court
    """
    conn = sqlite3.connect(db_path)

    # Always check if data exists in database first
    try:
        # Try to load existing data
        query = "SELECT * FROM play_by_play WHERE game_id = ?"
        existing_data = pd.read_sql_query(query, conn, params=(game_id,))

        if not existing_data.empty:
            print(f"Loading play-by-play data for game {game_id} from database...")
            conn.close()
            # Remove the game_id column since it was added for storage
            return existing_data.drop('GAME_ID', axis=1)

    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        # Table doesn't exist yet, will be created when we store data
        pass

    # Only fetch if data doesn't exist
    print(f"Game {game_id} not found in database. Fetching from API...")
    time.sleep(5)
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id).play_by_play.get_data_frame()
    pbp_with_players = noc.players_on_court(pbp)

    # Prepare data for storage by adding game_id column
    pbp_storage = pbp_with_players.copy()

    # Store in database using pandas to_sql
    pbp_storage.to_sql('play_by_play', conn, if_exists='append', index=False)
    conn.close()

    print(f"Stored play-by-play data for game {game_id} in database")
    return pbp_with_players


def list_stored_games(db_path='pbp_data.db'):
    """List all game IDs stored in the database."""
    try:
        conn = sqlite3.connect(db_path)
        stored_games = pd.read_sql_query("SELECT DISTINCT game_id FROM play_by_play", conn)
        conn.close()
        return stored_games['game_id'].tolist()
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return []


def delete_game_data(game_id, db_path='pbp_data.db'):
    """Delete stored data for a specific game."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM play_by_play WHERE game_id = ?", (game_id,))
    rows_deleted = cursor.rowcount
    conn.commit()
    conn.close()

    if rows_deleted > 0:
        print(f"Deleted data for game {game_id}")
    else:
        print(f"No data found for game {game_id}")

    return rows_deleted > 0

def estimate_possessions(pbp_df):
    """Estimate possessions in a stint based on play-by-play events."""
    # Simplified possession estimation (FGA, TOV, FTA/2, OREB)
    fga = len(pbp_df[pbp_df['EVENTMSGTYPE'].isin([1, 2])])  # Field goals made/missed
    tov = len(pbp_df[pbp_df['EVENTMSGTYPE'] == 5])  # Turnovers
    fta = len(pbp_df[pbp_df['EVENTMSGTYPE'] == 3])  # Free throws THIS IS AN APPROXIMATION RN ITS NOT GREAT
    oreb = len(pbp_df[pbp_df['EVENTMSGTYPE'] == 4])  # Rebounds (approximate offensive)
    possessions = fga + tov + 0.4 * fta - oreb
    return max(possessions, 1)  # Avoid zero possessions
def normalize_time(pctimestring, period):
    """Normalize PCTIMESTRING to actual game time by removing 120-second increments."""
    INCREMENT_SECONDS = 120
    return float(pctimestring) - (period * INCREMENT_SECONDS)


def process_stints(pbp_df, home_roster, away_roster):
    """Process play-by-play to extract stints, handling first stint and simultaneous substitutions."""
    stints = []
    current_lineup = {'home': set(), 'away': set()}
    stint_start_time = 0
    stint_start_period = 1
    stint_start_score = {'home': 0, 'away': 0}
    game_started = False

    def get_score_at_time(pbp_df, period, target_time):
        """Find the most recent score at or before the normalized target time."""
        events = pbp_df[
            (pbp_df['PERIOD'] == period) &
            (pbp_df['PCTIMESTRING'].notna()) &
            (pbp_df['SCORE'].notna())
        ]
        if events.empty:
            return {'home': 0, 'away': 0}
        events['normalized_time'] = events.apply(
            lambda x: normalize_time(x['PCTIMESTRING'], x['PERIOD']), axis=1
        )
        valid_events = events[
            (events['normalized_time'].notna()) &
            (events['normalized_time'] <= target_time)
        ]
        if valid_events.empty:
            return {'home': 0, 'away': 0}
        latest_event = valid_events.sort_values('normalized_time', ascending=False).iloc[0]
        score_str = latest_event['SCORE']
        try:
            home_score, away_score = map(int, score_str.split('-'))
            return {'home': home_score, 'away': away_score}
        except (ValueError, AttributeError):
            print(f"Invalid SCORE format: {score_str}")
            return {'home': 0, 'away': 0}

    # Get initial score for game start
    stint_start_score = get_score_at_time(pbp_df, period=1, target_time=0)

    # Compute normalized time
    pbp_df['normalized_time'] = pbp_df.apply(
        lambda x: normalize_time(x['PCTIMESTRING'], x['PERIOD']), axis=1
    )

    # Group events by PERIOD and PCTIMESTRING
    grouped = pbp_df.groupby(['PERIOD', 'PCTIMESTRING'])

    for (period, raw_time), group in grouped:
        if pd.isna(raw_time):
            continue
        current_time = normalize_time(raw_time, period)
        if pd.isna(current_time):
            continue

        # Use the last event's lineup
        last_event = group.iloc[-1]
        new_lineup = {
            'home': set([
                last_event[f'HOME_PLAYER{i}'] for i in range(1, 6)
                if not pd.isna(last_event[f'HOME_PLAYER{i}']) and
                last_event[f'HOME_PLAYER{i}'] in home_roster['PLAYER_ID'].values
            ]),
            'away': set([
                last_event[f'AWAY_PLAYER{i}'] for i in range(1, 6)
                if not pd.isna(last_event[f'AWAY_PLAYER{i}']) and
                last_event[f'AWAY_PLAYER{i}'] in away_roster['PLAYER_ID'].values
            ])
        }

        # Set first stint if lineup is valid and not yet set
        if not game_started:
            current_lineup = new_lineup
            stint_start_time = 0  # Assume game starts at normalized time 0
            stint_start_period = 1
            stint_start_score = get_score_at_time(pbp_df, period=1, target_time=0)
            game_started = True
            print(f"Game started with initial lineup at period {period}, normalized_time {current_time}")
            # Don't record stint yet; wait for change or end
            continue

        # Detect lineup change
        lineup_changed = (
            new_lineup['home'] != current_lineup['home'] or
            new_lineup['away'] != current_lineup['away']
        )
        if lineup_changed and len(current_lineup['home']) == 5 and len(current_lineup['away']) == 5:
            # End current stint
            stint_end_score = get_score_at_time(pbp_df, period, current_time)
            point_diff = (
                (stint_end_score['home'] - stint_end_score['away']) -
                (stint_start_score['home'] - stint_start_score['away'])
            )

            # Estimate possessions
            stint_pbp = pbp_df[
                (pbp_df['PERIOD'] == stint_start_period) &
                (pbp_df['normalized_time'].notna()) &
                (pbp_df['normalized_time'] >= stint_start_time) &
                (pbp_df['normalized_time'] <= current_time)
            ]
            possessions = estimate_possessions(stint_pbp)
            point_diff_per_80 = point_diff / possessions * 80 if possessions > 0 else 0

            stints.append({
                'home_players': current_lineup['home'].copy(),
                'away_players': current_lineup['away'].copy(),
                'point_diff': point_diff_per_80,
                'possessions': possessions
            })
            stint_start_time = current_time+1
            stint_start_period = period
            stint_start_score = get_score_at_time(pbp_df, period, current_time)

        # Update lineup and stint start if valid
        if len(new_lineup['home']) == 5 and len(new_lineup['away']) == 5:
            current_lineup = new_lineup

    # Handle final stint
    if len(current_lineup['home']) == 5 and len(current_lineup['away']) == 5:
        stint_end_score = get_score_at_time(pbp_df, period, current_time)
        point_diff = (
            (stint_end_score['home'] - stint_end_score['away']) -
            (stint_start_score['home'] - stint_start_score['away'])
        )
        stint_pbp = pbp_df[
            (pbp_df['PERIOD'] == stint_start_period) &
            (pbp_df['normalized_time'].notna()) &
            (pbp_df['normalized_time'] >= stint_start_time) &
            (pbp_df['normalized_time'] <= current_time)
        ]
        possessions = estimate_possessions(stint_pbp)
        point_diff_per_80 = point_diff / possessions * 80 if possessions > 0 else 0
        stints.append({
            'home_players': current_lineup['home'].copy(),
            'away_players': current_lineup['away'].copy(),
            'point_diff': point_diff_per_80,
            'possessions': possessions
        })

    return stints

def build_design_matrix(stints, all_players):
    """Build design matrix for RAPM regression."""
    X = np.zeros((len(stints), len(all_players)))
    y = np.zeros(len(stints))
    weights = np.zeros(len(stints))

    player_id_to_idx = {pid: idx for idx, pid in enumerate(all_players)}

    for i, stint in enumerate(stints):
        for player in stint['home_players']:
            X[i, player_id_to_idx[player]] = 1
        for player in stint['away_players']:
            X[i, player_id_to_idx[player]] = -1
        y[i] = stint['point_diff']
        weights[i] = stint['possessions']

    return X, y, weights


def calculate_rapm(X, y, weights, all_players, lambda_reg):
    """Calculate RAPM using ridge regression."""
    model = Ridge(alpha=lambda_reg, fit_intercept=False)
    model.fit(X, y, sample_weight=weights)
    rapm_values = model.coef_

    rapm_df = pd.DataFrame({
        'PLAYER_ID': all_players,
        'RAPM': rapm_values
    })
    return rapm_df


def main():
    print("Fetching WNBA teams...")
    wnba_teams = get_wnba_teams()
    team_ids = [team['id'] for team in wnba_teams]

    print("Fetching game IDs...")
    stints = []
    all_players = set()
    for SEASON in tqdm(('2023','2024','2025'),desc='Processessing Seasons'):
        game_ids = get_game_ids(SEASON, LEAGUE_ID)
        print("Fetching rosters...",SEASON)
        rosters = {}
        all_players=set(all_players)
        for team_id in team_ids:
            roster = get_team_roster(team_id, SEASON)
            rosters[team_id] = roster
            all_players.update(roster['PLAYER_ID'].values)

        all_players = list(all_players)

        print("Processing play-by-play data...")
        for game_id in tqdm(game_ids, desc="Processing games"):  # Limit to 10 games for testing
            try:
                pbp_df = get_play_by_play(game_id)
                home_team_id = pbp_df['PLAYER1_TEAM_ID'].dropna().iloc[0]
                away_team_id = pbp_df['PLAYER2_TEAM_ID'].dropna().iloc[0]
                home_roster = rosters.get(home_team_id, pd.DataFrame())
                away_roster = rosters.get(away_team_id, pd.DataFrame())

                if not home_roster.empty and not away_roster.empty:
                    game_stints = process_stints(pbp_df, home_roster, away_roster)
                    stints.extend(game_stints)
            except Exception as e:
                print(f"Error processing game {game_id}: {e}")

    print("Building design matrix...")
    X, y, weights = build_design_matrix(stints, all_players)

    print("Calculating RAPM...")
    rapm_df = calculate_rapm(X, y, weights, all_players, LAMBDA)

    # Merge player names
    player_names = pd.concat([roster[['PLAYER_ID', 'PLAYER','TeamID']] for roster in rosters.values()])
    print (player_names)
    print (wnba_teams)
    wnba_teams_df=pd.DataFrame(wnba_teams)
    player_names = player_names.merge(
        wnba_teams_df[['id', 'full_name']],
        left_on='TeamID',
        right_on='id',
        how='left'
    )
    player_names=player_names.rename(columns={'full_name':'Team'})
    rapm_df = rapm_df.merge(player_names, on='PLAYER_ID', how='left')
    rapm_df['RAPM']=rapm_df['RAPM']*-1
    print("RAPM Results:")
    print(rapm_df[['PLAYER', 'RAPM']].sort_values(by='RAPM', ascending=False))

    # Save to CSV
    rapm_df.to_csv('wnba_rapm_2024.csv', index=False)

st.set_page_config(
    page_title="RAPM Team Comparison",
    page_icon="🏀",
    layout="wide"
)
if __name__ == "__main__":
   # main()
    app.run_app()
