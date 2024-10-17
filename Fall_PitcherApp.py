import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Load the real dataset
file_path = 'AllTrackman_fall_2024_df AS OF 10_16 - AllTrackman_fall_2024_df.csv'  # Replace with the correct path in your Streamlit setup
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

test_df = load_data(file_path)

# Ensure numeric conversion for the columns where aggregation will be done
numeric_columns = ['RelSpeed', 'SpinRate', 'Tilt', 'RelHeight', 'RelSide', 
                   'Extension', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle', 'ExitSpeed']

# Coerce non-numeric values to NaN
for col in numeric_columns:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# Streamlit app layout
st.title("Ole Miss Pitcher Heat Maps (Fall 2024)")

# Dropdown widget to select the pitcher
pitcher_name = st.selectbox(
    "Select Pitcher:",
    options=test_df['Pitcher'].unique()
)

# Dropdown widget to select the batter side (Right, Left, or Both)
batter_side = st.selectbox(
    "Select Batter Side:",
    options=['Right', 'Left', 'Both']
)

# Dropdown widget for the number of strikes, with an "All" option
strikes = st.selectbox(
    "Select Strikes:",
    options=['All', 0, 1, 2]
)

# Dropdown widget for the number of balls, with an "All" option
balls = st.selectbox(
    "Select Balls:",
    options=['All', 0, 1, 2, 3]
)

# Function to filter data based on the dropdown selections
def filter_data(pitcher_name, batter_side, strikes, balls):
    pitcher_data = test_df[test_df['Pitcher'] == pitcher_name]

    if batter_side == 'Both':
        pitcher_data = pitcher_data[pitcher_data['BatterSide'].isin(['Right', 'Left'])]
    else:
        pitcher_data = pitcher_data[pitcher_data['BatterSide'] == batter_side]
    
    if strikes != 'All':
        pitcher_data = pitcher_data[pitcher_data['Strikes'] == strikes]
    
    if balls != 'All':
        pitcher_data = pitcher_data[pitcher_data['Balls'] == balls]
    
    return pitcher_data

# Functions to generate Pitch Traits and Plate Discipline tables
def format_dataframe(df):
    df = df.copy()
    percent_columns = ['InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%']
    
    for col in df.columns:
        if col in percent_columns:
            df[col] = df[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) and isinstance(x, (int, float)) else 'N/A')
        elif df[col].dtype.kind in 'f':  # if it's a float type column
            df[col] = df[col].apply(lambda x: round(x, 2) if pd.notna(x) else 'N/A')
        else:
            df[col] = df[col].fillna('N/A')
    return df

def generate_pitch_traits_table(pitcher_name, batter_side, strikes, balls):
    pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls)
    
    grouped_data = pitcher_data.groupby('TaggedPitchType').agg(
        Count=('TaggedPitchType', 'size'),
        RelSpeed=('RelSpeed', 'mean'),
        InducedVertBreak=('InducedVertBreak', 'mean'),
        HorizontalBreak=('HorzBreak', 'mean'), 
        SpinRate=('SpinRate', 'mean'),
        RelHeight=('RelHeight', 'mean'),
        RelSide=('RelSide', 'mean'),
        Extension=('Extension', 'mean'),
        VertApprAngle=('VertApprAngle', 'mean'),
        Tilt=('Tilt', 'mean'),
        ExitSpeed=('ExitSpeed', lambda x: x.mean() if x.notna().sum() > 0 else 'N/A')
    ).reset_index()

    grouped_data = grouped_data.sort_values(by='Count', ascending=False)
    formatted_data = format_dataframe(grouped_data)

    st.subheader("Pitch Traits:")
    st.dataframe(formatted_data)

def calculate_in_zone(df):
    return df[(df['PlateLocHeight'] >= 1.5) & (df['PlateLocHeight'] <= 3.3775) &
              (df['PlateLocSide'] >= -0.708) & (df['PlateLocSide'] <= 0.708)]

def generate_plate_discipline_table(pitcher_name, batter_side, strikes, balls):
    pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls)
    total_pitches = len(pitcher_data)
    
    def calculate_metrics(df):
        in_zone_pitches = calculate_in_zone(df)
        total_in_zone = len(in_zone_pitches)
        total_swings = df[df['PitchCall'].isin(['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay'])].shape[0]
        total_whiffs = df[df['PitchCall'] == 'StrikeSwinging'].shape[0]
        total_chase = df[~df.index.isin(in_zone_pitches.index) & df['PitchCall'].isin(['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay'])].shape[0]
        in_zone_whiffs = in_zone_pitches[in_zone_pitches['PitchCall'] == 'StrikeSwinging'].shape[0]
        
        metrics = {
            'InZone%': (total_in_zone / len(df)) * 100 if len(df) > 0 else 'N/A',
            'Swing%': (total_swings / len(df)) * 100 if len(df) > 0 else 'N/A',
            'Whiff%': (total_whiffs / total_swings) * 100 if total_swings > 0 else 'N/A',
            'Chase%': (total_chase / total_swings) * 100 if total_swings > 0 else 'N/A',
            'InZoneWhiff%': (in_zone_whiffs / total_in_zone) * 100 if total_in_zone > 0 else 'N/A'
        }
        return metrics

    plate_discipline_data = pitcher_data.groupby('TaggedPitchType').apply(calculate_metrics).apply(pd.Series).reset_index()
    plate_discipline_data['Count'] = pitcher_data.groupby('TaggedPitchType')['TaggedPitchType'].count().values
    plate_discipline_data['Pitch%'] = (plate_discipline_data['Count'] / total_pitches) * 100
    plate_discipline_data = plate_discipline_data.sort_values(by='Count', ascending=False)

    plate_discipline_data = plate_discipline_data[['TaggedPitchType', 'Count', 'Pitch%', 'InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%']]
    formatted_data = format_dataframe(plate_discipline_data)

    st.subheader("Plate Discipline:")
    st.dataframe(formatted_data)

# Generate tables for Pitch Traits and Plate Discipline
generate_pitch_traits_table(pitcher_name, batter_side, strikes, balls)
generate_plate_discipline_table(pitcher_name, batter_side, strikes, balls)

# Function to generate the Pitch Usage (By Count) table
def generate_pitch_usage_table(pitcher_name):
    pitcher_df = test_df[test_df['Pitcher'] == pitcher_name]

    counts_order = [(0, 0), (1, 0), (2, 0), (3, 0),
                    (0, 1), (0, 2), (1, 1), (2, 1),
                    (3, 1), (1, 2), (2, 2), (3, 2)]

    grouped = pitcher_df.groupby(['TaggedPitchType', 'Balls', 'Strikes']).size().reset_index(name='Count')

    pivot_table = pd.DataFrame(index=grouped['TaggedPitchType'].unique())

    for count in counts_order:
        balls, strikes = count
        count_data = grouped[(grouped['Balls'] == balls) & (grouped['Strikes'] == strikes)]
        total_pitches_for_count = count_data['Count'].sum()

        if total_pitches_for_count > 0:
            count_data['Pitch%'] = (count_data['Count'] / total_pitches_for_count) * 100
        else:
            count_data['Pitch%'] = 0

        count_data['Pitch%'] = count_data['Pitch%'].round(2)
        count_data = count_data[['TaggedPitchType', 'Pitch%']].set_index('TaggedPitchType')
        pivot_table[f'({balls},{strikes}) Pitch%'] = count_data['Pitch%']

    def add_in_zone_percentage(count_df, balls, strikes):
        count_in_zone_df = calculate_in_zone(count_df)
        in_zone_grouped = count_in_zone_df.groupby('TaggedPitchType').size().reset_index(name='InZoneCount')
        total_pitches = count_df.groupby('TaggedPitchType').size().reset_index(name='TotalCount')

        in_zone_percentage = pd.merge(in_zone_grouped, total_pitches, on='TaggedPitchType', how='right')
        in_zone_percentage['InZone%'] = (in_zone_percentage['InZoneCount'] / in_zone_percentage['TotalCount']) * 100
        in_zone_percentage['InZone%'] = in_zone_percentage['InZone%'].round(2).fillna(0)

        pivot_table[f'({balls},{strikes}) InZone%'] = in_zone_percentage.set_index('TaggedPitchType')['InZone%']

    count_0_0_df = pitcher_df[(pitcher_df['Balls'] == 0) & (pitcher_df['Strikes'] == 0)]
    add_in_zone_percentage(count_0_0_df, 0, 0)

    count_1_1_df = pitcher_df[(pitcher_df['Balls'] == 1) & (pitcher_df['Strikes'] == 1)]
    add_in_zone_percentage(count_1_1_df, 1, 1)

    pivot_table = pivot_table.fillna(0).applymap(lambda x: f'{x:.2f}%' if isinstance(x, (int, float)) else x)
    pivot_table.reset_index(inplace=True)

    st.subheader("Pitch Usage (By Count):")
    st.dataframe(pivot_table)

# Generate the Pitch Usage (By Count) table
generate_pitch_usage_table(pitcher_name)
