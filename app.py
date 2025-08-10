
import streamlit as st
st.set_page_config(
    page_title="RAPM Team Comparison",
    page_icon="🏀",
    layout="wide"
)
import pandas as pd
import numpy as np


def load_sample_data():
    data=pd.read_csv('wnba_rapm_2024.csv').dropna()
    return data
def run_app():
    st.title("🏀 RAPM Team Comparison Tool")
    st.markdown("Select 5 players for each team to compare their combined RAPM values")

    # Load data
    # Replace this line with: df = your_actual_dataframe
    df = load_sample_data()

    # Verify required columns exist
    if 'PLAYER' not in df.columns or 'RAPM' not in df.columns:
        st.error("DataFrame must contain 'PLAYER' and 'RAPM' columns")
        return

    # Sort players by RAPM for better UX
    if 'Team' in df.columns:
        st.subheader("🏀 Team Filter")
        team_a_filtered_df = df
        team_b_filtered_df = df
        filter_col1, filter_col2 = st.columns(2)

        teams = sorted(df['Team'].unique())

        with filter_col1:
            st.write("**🔵 Team A - Available Teams:**")
            team_a_teams = st.multiselect(
                "Select teams for Team A players (leave empty for all teams):",
                options=teams,
                default=[],
                key="team_a_filter"
            )

            if team_a_teams:
                team_a_filtered_df = df[df['Team'].isin(team_a_teams)]

        with filter_col2:
            st.write("**🔴 Team B - Available Teams:**")
            team_b_teams = st.multiselect(
                "Select teams for Team B players (leave empty for all teams):",
                options=teams,
                default=[],
                key="team_b_filter"
            )

            if team_b_teams:
                team_b_filtered_df = df[df['Team'].isin(team_b_teams)]

        st.markdown("---")

    # Sort players by RAPM for better UX
    team_a_sorted = team_a_filtered_df.sort_values('RAPM', ascending=False).reset_index(drop=True)
    team_b_sorted = team_b_filtered_df.sort_values('RAPM', ascending=False).reset_index(drop=True)

    team_a_player_names = team_a_sorted['PLAYER'].tolist()
    team_b_player_names = team_b_sorted['PLAYER'].tolist()

    # Create two columns for team selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 Team A")
        team_a_players = []
        for i in range(5):
            # Filter out already selected players to prevent duplicates
            available_players = [p for p in team_a_player_names if p not in team_a_players]
            selected = st.selectbox(
                f"Player {i + 1}:",
                options=[""] + available_players,
                key=f"team_a_player_{i}"
            )
            if selected:
                team_a_players.append(selected)

    with col2:
        st.subheader("🔴 Team B")
        team_b_players = []
        for i in range(5):
            # Filter out already selected players to prevent duplicates
            available_players = [p for p in team_b_player_names
                                 if p not in team_b_players and p not in team_a_players]
            selected = st.selectbox(
                f"Player {i + 1}:",
                options=[""] + available_players,
                key=f"team_b_player_{i}"
            )
            if selected:
                team_b_players.append(selected)

    # Calculate and display results
    st.markdown("---")

    if len(team_a_players) > 0 or len(team_b_players) > 0:
        # Calculate RAPM totals
        team_a_rapm = 0
        team_b_rapm = 0

        if team_a_players:
            team_a_rapm = df[df['PLAYER'].isin(team_a_players)]['RAPM'].sum()

        if team_b_players:
            team_b_rapm = df[df['PLAYER'].isin(team_b_players)]['RAPM'].sum()

        # Display results in columns
        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                label="🔵 Team A Total RAPM",
                value=f"{team_a_rapm:.2f}",
                delta=f"{len(team_a_players)}/5 players selected"
            )

            if team_a_players:
                st.write("**Team A Players:**")
                for player in team_a_players:
                    rapm_val = df[df['PLAYER'] == player]['RAPM'].iloc[0]
                    st.write(f"• {player}: {rapm_val:.2f}")

        with result_col2:
            st.metric(
                label="🔴 Team B Total RAPM",
                value=f"{team_b_rapm:.2f}",
                delta=f"{len(team_b_players)}/5 players selected"
            )

            if team_b_players:
                st.write("**Team B Players:**")
                for player in team_b_players:
                    rapm_val = df[df['PLAYER'] == player]['RAPM'].iloc[0]
                    st.write(f"• {player}: {rapm_val:.2f}")

        with result_col3:
            difference = team_a_rapm - team_b_rapm
            st.metric(
                label="📊 RAPM Difference (A - B)",
                value=f"{difference:.2f}",
                delta="Team A advantage" if difference > 0 else "Team B advantage" if difference < 0 else "Tied"
            )

            # Show which team is better
            if difference > 0:
                st.success(f"🔵 Team A has a {difference:.2f} RAPM advantage")
            elif difference < 0:
                st.error(f"🔴 Team B has a {abs(difference):.2f} RAPM advantage")
            else:
                st.info("🤝 Teams are tied in RAPM")

    # Display available players table
    with st.expander("📋 View Available Players by Team Filter"):
        tab1, tab2 = st.tabs(["🔵 Team A Available Players", "🔴 Team B Available Players"])

        with tab1:
            display_cols = ['PLAYER', 'RAPM']
            if 'Team' in team_a_sorted.columns:
                display_cols.append('Team')

            st.dataframe(
                team_a_sorted[display_cols].style.format({'RAPM': '{:.2f}'}),
                use_container_width=True
            )

        with tab2:
            display_cols = ['PLAYER', 'RAPM']
            if 'Team' in team_b_sorted.columns:
                display_cols.append('Team')

            st.dataframe(
                team_b_sorted[display_cols].style.format({'RAPM': '{:.2f}'}),
                use_container_width=True
            )
    st.markdown("---")
    st.header("⏱️ Minutes-Weighted RAPM Calculator")

    # League selection for game length
    league_col1, league_col2 = st.columns([1, 3])
    with league_col1:
        league = st.selectbox("League:", ["NBA", "WNBA"], key="league_select")

    game_minutes = 48 if league == "NBA" else 40
    total_team_minutes = 240 if league == "NBA" else 200

    with league_col2:
        st.markdown(f"**{league}**: {game_minutes} minutes per game, {total_team_minutes} total team minutes")

    # Standard rotation minutes based on league
    if league == "NBA":
        rotation_presets = {
            "Starter 1": 36, "Starter 2": 34, "Starter 3": 32, "Starter 4": 30, "Starter 5": 28,
            "6th Man": 24, "Rotation 7": 20, "Rotation 8": 18, "Rotation 9": 12, "Bench": 6
        }
    else:  # WNBA
        rotation_presets = {
            "Starter 1": 32, "Starter 2": 30, "Starter 3": 28, "Starter 4": 26, "Starter 5": 24,
            "6th Player": 20, "Rotation 7": 16, "Rotation 8": 14, "Rotation 9": 10, "Bench": 0
        }

    st.markdown(
        f"Assign minutes to players and calculate weighted RAPM based on playing time ({game_minutes} minutes per player max, {total_team_minutes} total team minutes)")

    # Create columns for minutes assignment
    minutes_col1, minutes_col2 = st.columns(2)

    with minutes_col1:
        st.subheader("🔵 Team A Minutes")
        team_a_minutes = {}
        team_a_total_minutes = 0

        # Show all available players with rotation presets
        st.write("**Player Minutes Assignment:**")

        # Auto-rotation button
        if st.button("🎯 Auto-Set Rotation", key="auto_rotation_a"):
            # Set suggested minutes for top players
            for idx, player in enumerate(team_a_player_names[:10]):  # Only set for top 10 players
                if idx < len(list(rotation_presets.values())):
                    suggested_minutes = list(rotation_presets.values())[idx]
                    st.session_state[f"minutes_a_all_{player}"] = suggested_minutes

        # Reset button
        if st.button("🔄 Reset Team A", key="reset_a"):
            for player in team_a_player_names:
                st.session_state[f"minutes_a_all_{player}"] = 0

        # Show all available players
        for idx, player in enumerate(team_a_player_names):
            # Default to 0, but suggest rotation minutes for top players on first load
            if f"minutes_a_all_{player}" not in st.session_state:
                if idx < len(list(rotation_presets.values())):
                    st.session_state[f"minutes_a_all_{player}"] = list(rotation_presets.values())[idx]
                else:
                    st.session_state[f"minutes_a_all_{player}"] = 0

            minutes = st.number_input(
                f"{player}:",
                min_value=0,
                max_value=game_minutes,
                value=st.session_state[f"minutes_a_all_{player}"],
                step=1,
                key=f"minutes_a_all_{player}",
                help=f"RAPM: {df[df['PLAYER'] == player]['RAPM'].iloc[0]:.2f}"
            )
            if minutes > 0:
                team_a_minutes[player] = minutes
                team_a_total_minutes += minutes

        # Show total minutes with color coding
        minutes_color_a = "🔴" if team_a_total_minutes > total_team_minutes else "🟢" if team_a_total_minutes == total_team_minutes else "🟡"
        st.metric(f"{minutes_color_a} Team A Total Minutes", f"{team_a_total_minutes}/{total_team_minutes}")

    with minutes_col2:
        st.subheader("🔴 Team B Minutes")
        team_b_minutes = {}
        team_b_total_minutes = 0

        # Show all available players with rotation presets
        st.write("**Player Minutes Assignment:**")

        # Auto-rotation button
        if st.button("🎯 Auto-Set Rotation", key="auto_rotation_b"):
            # Set suggested minutes for top players
            for idx, player in enumerate(team_b_player_names[:10]):  # Only set for top 10 players
                if idx < len(list(rotation_presets.values())):
                    suggested_minutes = list(rotation_presets.values())[idx]
                    st.session_state[f"minutes_b_all_{player}"] = suggested_minutes

        # Reset button
        if st.button("🔄 Reset Team B", key="reset_b"):
            for player in team_b_player_names:
                st.session_state[f"minutes_b_all_{player}"] = 0

        # Show all available players
        for idx, player in enumerate(team_b_player_names):
            # Default to 0, but suggest rotation minutes for top players on first load
            if f"minutes_b_all_{player}" not in st.session_state:
                if idx < len(list(rotation_presets.values())):
                    st.session_state[f"minutes_b_all_{player}"] = list(rotation_presets.values())[idx]
                else:
                    st.session_state[f"minutes_b_all_{player}"] = 0

            minutes = st.number_input(
                f"{player}:",
                min_value=0,
                max_value=game_minutes,
                value=st.session_state[f"minutes_b_all_{player}"],
                step=1,
                key=f"minutes_b_all_{player}",
                help=f"RAPM: {df[df['PLAYER'] == player]['RAPM'].iloc[0]:.2f}"
            )
            if minutes > 0:
                team_b_minutes[player] = minutes
                team_b_total_minutes += minutes

        # Show total minutes with color coding
        minutes_color_b = "🔴" if team_b_total_minutes > total_team_minutes else "🟢" if team_b_total_minutes == total_team_minutes else "🟡"
        st.metric(f"{minutes_color_b} Team B Total Minutes", f"{team_b_total_minutes}/{total_team_minutes}")

    # Calculate and display weighted RAPM
    if team_a_minutes or team_b_minutes:
        st.markdown("---")
        st.subheader("📊 Minutes-Weighted RAPM Results")

        # Calculate weighted RAPM for each team
        team_a_weighted_rapm = 0
        team_b_weighted_rapm = 0

        if team_a_minutes and team_a_total_minutes > 0:
            for player, minutes in team_a_minutes.items():
                player_rapm = df[df['PLAYER'] == player]['RAPM'].iloc[0]
                weight = minutes / game_minutes  # Proportion of game played
                team_a_weighted_rapm += player_rapm * weight

        if team_b_minutes and team_b_total_minutes > 0:
            for player, minutes in team_b_minutes.items():
                player_rapm = df[df['PLAYER'] == player]['RAPM'].iloc[0]
                weight = minutes / game_minutes  # Proportion of game played
                team_b_weighted_rapm += player_rapm * weight

        # Display results
        weighted_col1, weighted_col2, weighted_col3 = st.columns(3)

        with weighted_col1:
            st.metric(
                label="🔵 Team A Weighted RAPM",
                value=f"{team_a_weighted_rapm:.3f}",
                delta=f"{team_a_total_minutes} total minutes"
            )

            if team_a_minutes:
                st.write("**Team A Breakdown:**")
                for player, minutes in team_a_minutes.items():
                    if minutes > 0:
                        player_rapm = df[df['PLAYER'] == player]['RAPM'].iloc[0]
                        weight = minutes / game_minutes
                        contribution = player_rapm * weight
                        st.write(f"• {player}: {minutes}min ({weight:.1%}) = {contribution:.3f}")

        with weighted_col2:
            st.metric(
                label="🔴 Team B Weighted RAPM",
                value=f"{team_b_weighted_rapm:.3f}",
                delta=f"{team_b_total_minutes} total minutes"
            )

            if team_b_minutes:
                st.write("**Team B Breakdown:**")
                for player, minutes in team_b_minutes.items():
                    if minutes > 0:
                        player_rapm = df[df['PLAYER'] == player]['RAPM'].iloc[0]
                        weight = minutes / game_minutes
                        contribution = player_rapm * weight
                        st.write(f"• {player}: {minutes}min ({weight:.1%}) = {contribution:.3f}")

        with weighted_col3:
            weighted_difference = team_a_weighted_rapm - team_b_weighted_rapm
            st.metric(
                label="📊 Weighted RAPM Difference",
                value=f"{weighted_difference:.3f}",
                delta="Team A advantage" if weighted_difference > 0 else "Team B advantage" if weighted_difference < 0 else "Tied"
            )

            # Show which team is better
            if weighted_difference > 0:
                st.success(f"🔵 Team A has a {weighted_difference:.3f} weighted advantage")
            elif weighted_difference < 0:
                st.error(f"🔴 Team B has a {abs(weighted_difference):.3f} weighted advantage")
            else:
                st.info("🤝 Teams are tied in weighted RAPM")

        # Warning for minute allocation
        if team_a_total_minutes != total_team_minutes or team_b_total_minutes != total_team_minutes:
            st.warning(
                f"⚠️ Note: Teams should have exactly {total_team_minutes} minutes allocated for accurate game-level comparison")

    else:
        st.info("👆 Assign minutes to players above to see weighted RAPM calculations")

    # Display available players table
    with st.expander("📋 View Available Players by Team Filter"):
        tab1, tab2 = st.tabs(["🔵 Team A Available Players", "🔴 Team B Available Players"])

        with tab1:
            display_cols = ['PLAYER', 'RAPM']
            if 'Team' in team_a_sorted.columns:
                display_cols.append('Team')

            st.dataframe(
                team_a_sorted[display_cols].style.format({'RAPM': '{:.2f}'}),
                use_container_width=True
            )

        with tab2:
            display_cols = ['PLAYER', 'RAPM']
            if 'Team' in team_b_sorted.columns:
                display_cols.append('Team')

            st.dataframe(
                team_b_sorted[display_cols].style.format({'RAPM': '{:.2f}'}),
                use_container_width=True
            )
def main():
    run_app()

if __name__ == "__main__":
    main()

