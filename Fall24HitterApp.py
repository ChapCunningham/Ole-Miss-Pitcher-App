import streamlit as st
import pandas as pd

# Load the dataset
file_path = 'Fall_Trackman_Master.csv'  # Replace with the correct path in your Streamlit setup
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data(file_path)

# Coerce numeric values where necessary
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Balls'] = pd.to_numeric(df['Balls'], errors='coerce')
df['Strikes'] = pd.to_numeric(df['Strikes'], errors='coerce')

# Streamlit app layout
st.title("Hitter Stats - Fall 2024")

# Filters for Balls and Strikes
balls = st.selectbox("Select Balls:", options=['All', 0, 1, 2, 3])
strikes = st.selectbox("Select Strikes:", options=['All', 0, 1, 2])

# Filter for Pitcher Handedness
pitcher_throws = st.selectbox(
    "Pitcher Handedness:",
    options=['Both', 'Right', 'Left'], 
    index=0  # Default to 'Both'
)

# Function to filter data based on user selections
def filter_data(balls, strikes, pitcher_throws):
    filtered_data = df.copy()

    # Filter for Balls if 'All' is not selected
    if balls != 'All':
        filtered_data = filtered_data[filtered_data['Balls'] == balls]
    
    # Filter for Strikes if 'All' is not selected
    if strikes != 'All':
        filtered_data = filtered_data[filtered_data['Strikes'] == strikes]
    
    # Filter for Pitcher Handedness
    if pitcher_throws != 'Both':
        filtered_data = filtered_data[filtered_data['PitcherThrows'] == pitcher_throws]
    
    return filtered_data

# Filter data based on user selections
filtered_data = filter_data(balls, strikes, pitcher_throws)

# Function to calculate hitter stats
def calculate_hitter_stats(df):
    # Plate Appearances (PA)
    df['PA'] = df.groupby('Batter')['PitchofPA'].transform(lambda x: (x == 1).sum())

    # At Bats (AB) - Exclude BB, HBP, Sacrifice Fly/Bunt
    ab_df = df[~df['PitchCall'].isin(['BallCalled', 'HitByPitch', 'Sacrifice', 'SacrificeFly'])]
    df['AB'] = ab_df.groupby('Batter')['PitchofPA'].transform(lambda x: (x == 1).sum())

    # Hits and Walks
    hits_df = df[df['PlayResult'].isin(['Single', 'Double', 'Triple', 'HomeRun'])]
    walks_df = df[df['KorBB'] == 'Walk']

    # Total Hits
    df['Hits'] = hits_df.groupby('Batter')['PlayResult'].transform('count')

    # Home Runs (HR)
    df['HR'] = df[df['PlayResult'] == 'HomeRun'].groupby('Batter')['PlayResult'].transform('count')

    # Strikeouts (SO)
    df['SO'] = df[df['KorBB'] == 'Strikeout'].groupby('Batter')['KorBB'].transform('count')

    # Walks (BB)
    df['BB'] = walks_df.groupby('Batter')['KorBB'].transform('count')

    # Check if the 'HBP' column exists, otherwise use 0 for Hit by Pitch
    if 'HBP' not in df.columns:
        df['HBP'] = 0

    # Check if the 'SacrificeFly' column exists, otherwise set to 0
    if 'SacrificeFly' not in df.columns:
        df['SacrificeFly'] = 0

    # Batting Average (AVG) - Hits / At Bats
    df['AVG'] = df['Hits'] / df['AB']

    # On Base Percentage (OBP) - (Hits + Walks + HBP) / (AB + BB + HBP + Sacrifice Flies)
    df['OBP'] = (df['Hits'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SacrificeFly'])

    # Slugging Percentage (SLG)
    df['1B'] = hits_df[hits_df['PlayResult'] == 'Single'].groupby('Batter')['PlayResult'].transform('count')
    df['2B'] = hits_df[hits_df['PlayResult'] == 'Double'].groupby('Batter')['PlayResult'].transform('count')
    df['3B'] = hits_df[hits_df['PlayResult'] == 'Triple'].groupby('Batter')['PlayResult'].transform('count')

    df['SLG'] = (df['1B'] + df['2B']*2 + df['3B']*3 + df['HR']*4) / df['AB']

    # On Base Plus Slugging (OPS) = OBP + SLG
    df['OPS'] = df['OBP'] + df['SLG']

    return df




# Calculate the stats for the filtered data
hitter_stats = calculate_hitter_stats(filtered_data)

# Select columns to display
columns_to_display = ['Batter', 'PA', 'AB', 'AVG', 'OBP', 'SLG', 'OPS', 'SO', 'BB', 'HR']

# Remove duplicates and display the final stats
hitter_stats = hitter_stats[columns_to_display].drop_duplicates()

# Display the data in Streamlit
st.dataframe(hitter_stats)

