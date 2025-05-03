import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import math
from datetime import datetime

# File paths for datasets
fall_file_path = 'FINAL FALL CSV 2024 - filtered_fall_trackman (1).csv'  # Fall dataset
winter_file_path = 'WINTER_ALL_trackman.csv'  # Winter dataset

# File path for Spring Preseason dataset
spring_file_path = "Spring Intrasquads MASTER.csv"
season_25_file_path = "2025_SEASON.csv"

# Load Spring Preseason dataset



@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Coerce invalid dates to NaT
        df = df[df['Date'].notna()]  # Remove rows with invalid dates
    return df


# Load Fall and Winter datasets
fall_df = load_data(fall_file_path)
winter_df = load_data(winter_file_path)
spring_df = load_data(spring_file_path)
season_df = load_data(season_25_file_path)

# Add a column to distinguish datasets
fall_df['Season'] = 'Fall'
winter_df['Season'] = 'Winter'
spring_df['Season'] = 'Spring Preseason'
season_df['Season'] = '2025 Season'

# Combine datasets for "All" option
all_data_df = pd.concat([fall_df, winter_df, spring_df, season_df])

# Default to "Fall" dataset initially
test_df = season_df
test_df = test_df[test_df['PitcherTeam'] == 'OLE_REB']

# File paths for the CLASS+ and OTHER Rolling Stats CSVs
fall_rolling_path = 'fall_CLASS+_by_date.csv'
winter_rolling_path = 'winter_CLASS+_by_date (1).csv'

# Load the new datasets
fall_rolling_df = load_data(fall_rolling_path)
winter_rolling_df = load_data(winter_rolling_path)

# Add Season column to distinguish the datasets
fall_rolling_df['Season'] = 'Fall'
winter_rolling_df['Season'] = 'Winter'



# Ensure numeric conversion for the columns where aggregation will be done
numeric_columns = ['RelSpeed', 'SpinRate', 'Tilt', 'RelHeight', 'RelSide', 
                   'Extension', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle', 'ExitSpeed','VertRelAngle']

# Coerce non-numeric values to NaN
for col in numeric_columns:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# Streamlit app layout
st.title("OMBSB Pitcher Reports")

# Sidebar for filters
st.sidebar.header("Filters")

# Dropdown for dataset selection (Fall, Winter, Spring, or All)
dataset_selection = st.sidebar.selectbox(
    "Select Dataset:",
    options=['2025 Season','Fall', 'Winter', 'Spring Preseason', 'All']
)

# Apply dataset selection
if dataset_selection == 'Fall':
    test_df = fall_df
elif dataset_selection == 'Winter':
    test_df = winter_df
elif dataset_selection == 'Spring Preseason':
    test_df = spring_df

elif dataset_selection == '2025 Season':
    test_df = season_df
else:  # "All"
    test_df = all_data_df


# Filter by team
test_df = test_df[test_df['PitcherTeam'].isin(['OLE_REB', 'OLE_PRAC'])]


# Dropdown widget to select the pitcher
pitcher_name = st.sidebar.selectbox(
    "Select Pitcher:",
    options=test_df['Pitcher'].unique()
)

heatmap_type = st.sidebar.selectbox(
    "Select Heatmap Type:",
    options=["Frequency", "Whiff", "Exit Velocity"]
)


# Dropdown widget to select the batter side (Right, Left, or Both)
batter_side = st.sidebar.selectbox(
    "Select Batter Side:",
    options=['Both', 'Right', 'Left']  # Added 'Both' option
)

# Dropdown widget for the number of strikes, with an "All" option
strikes = st.sidebar.selectbox(
    "Select Strikes:",
    options=['All', 0, 1, 2]
)

# Dropdown widget for the number of balls, with an "All" option
balls = st.sidebar.selectbox(
    "Select Balls:",
    options=['All', 0, 1, 2, 3]
)

# Date Filtering Section
st.sidebar.header("Date Filtering")

# Option to choose the type of date filter
# Date Filtering Section
st.sidebar.header("Date Filtering")

# Option to choose the type of date filter
date_filter_option = st.sidebar.selectbox(
    "Select Date Filter:",
    options=["All", "Single Date", "Date Range"]
)

# Initialize date variables
selected_date = None
start_date = None
end_date = None

# Display appropriate date picker based on the selected option
if date_filter_option == "Single Date":
    selected_date = st.sidebar.date_input(
        "Select a Date",
        value=datetime.today()
    )
elif date_filter_option == "Date Range":
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[datetime.today(), datetime.today()]
    )
    # Check if the user selected a valid date range
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        st.sidebar.warning("Please select a valid date range.")


# File path for Spring Rolling CLASS+ dataset
spring_rolling_path = "spring_CLASS+_by_date.csv"
season_rolling_path = "2025_CLASS+_by_date.csv"

# Load Spring rolling CLASS+ dataset
spring_rolling_df = load_data(spring_rolling_path)
spring_rolling_df['Season'] = 'Spring Preseason'

season_rolling_df = load_data(season_rolling_path)
season_rolling_df['Season'] = '2025 Season'



# Update the rolling dataset selection
if dataset_selection == 'Fall':
    rolling_df = fall_rolling_df
elif dataset_selection == 'Winter':
    rolling_df = winter_rolling_df
elif dataset_selection == 'Spring Preseason':
    rolling_df = spring_rolling_df
elif dataset_selection == '2025 Season':
    rolling_df = season_rolling_df
else:  # "All"
    rolling_df = pd.concat([fall_rolling_df, winter_rolling_df, spring_rolling_df, season_rolling_df])


def filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data for the selected pitcher
        pitcher_data = test_df[test_df['Pitcher'] == pitcher_name]

        # Apply filtering for batter side, including 'Both' option
        if batter_side == 'Both':
            pitcher_data = pitcher_data[pitcher_data['BatterSide'].isin(['Right', 'Left'])]
        else:
            pitcher_data = pitcher_data[pitcher_data['BatterSide'] == batter_side]

        # Apply filtering for strikes if 'All' is not selected
        if strikes != 'All':
            pitcher_data = pitcher_data[pitcher_data['Strikes'] == strikes]

        # Apply filtering for balls if 'All' is not selected
        if balls != 'All':
            pitcher_data = pitcher_data[pitcher_data['Balls'] == balls]

        # Apply date filtering
        if date_filter_option == "Single Date" and selected_date:
            selected_datetime = pd.to_datetime(selected_date)
            pitcher_data = pitcher_data[pitcher_data['Date'].dt.date == selected_datetime.date()]
        elif date_filter_option == "Date Range" and start_date and end_date:
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date)
            pitcher_data = pitcher_data[
                (pitcher_data['Date'] >= start_datetime) & 
                (pitcher_data['Date'] <= end_datetime)
            ]

        return pitcher_data
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame to avoid further errors


def plot_heatmaps(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date, map_type):
    try:
        # Filter data with date parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)
        
        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return
        
        # Remove rows where PlateLocSide or PlateLocHeight is NaN, for plotting purposes only
        plot_data = pitcher_data.dropna(subset=['PlateLocSide', 'PlateLocHeight'])
        
        if plot_data.empty:
            st.write("No data available to plot after filtering.")
            return
        
        # Get unique pitch types thrown by the selected pitcher
        unique_pitch_types = plot_data['TaggedPitchType'].unique()
        
        # Limit number of subplots per row (e.g., 3 per row)
        n_pitch_types = len(unique_pitch_types)
        plots_per_row = 3  # Set number of plots per row
        n_rows = math.ceil(n_pitch_types / plots_per_row)  # Calculate the number of rows needed
        
        # Adjust figure size dynamically
        fig_width = 12 * plots_per_row  # Set width based on number of plots per row
        fig_height = 16 * n_rows  # Set height to fit all rows

        # Create subplots with the appropriate number of rows and columns
        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(fig_width, fig_height))
        
        if n_pitch_types == 1:
            axes = [axes]  # Ensure axes is iterable
        else:
            axes = axes.flatten()  # Flatten axes array for easier access

        # Loop over each unique pitch type and create heatmaps
        for i, (ax, pitch_type) in enumerate(zip(axes, unique_pitch_types)):
            pitch_type_data = plot_data[plot_data['TaggedPitchType'] == pitch_type]
            
            if map_type == 'Frequency':
                # All pitches are used for frequency maps
                heatmap_data = pitch_type_data
            elif map_type == 'Whiff':
                # Only use pitches with 'StrikeSwinging' for heatmap
                heatmap_data = pitch_type_data[pitch_type_data['PitchCall'] == 'StrikeSwinging']
            elif map_type == 'Exit Velocity':
                # Use all pitches for Exit Velocity but map ExitSpeed
                heatmap_data = pitch_type_data

            # Scatter plot for all pitches
            ax.scatter(
                pitch_type_data['PlateLocSide'], 
                pitch_type_data['PlateLocHeight'], 
                color='black',  # Color for the dots
                edgecolor='white',  # Add a white border to make dots stand out
                s=300,  # Size of the dots
                alpha=0.7  # Transparency to allow overlap
            )
            
            # Check if enough data points exist for a heatmap
            if len(heatmap_data) >= 5:
                bw_adjust_value = 0.5 if len(heatmap_data) > 50 else 1  # Adjust bandwidth for small datasets
                sns.kdeplot(
                    x=heatmap_data['PlateLocSide'], 
                    y=heatmap_data['PlateLocHeight'], 
                    fill=True, 
                    cmap='Spectral_r' if map_type == 'Frequency' else 'coolwarm', 
                    levels=6, 
                    ax=ax,
                    bw_adjust=bw_adjust_value
                )

            # Add strike zone as a rectangle with black edgecolor
            strike_zone_width = 1.66166  # feet changed for widest raw strike (formerly 17/12)
            strike_zone_params = {
                'x_start': -strike_zone_width / 2,
                'y_start': 1.5,
                'width': strike_zone_width,
                'height': 3.3775 - 1.5
            }
            strike_zone = patches.Rectangle(
                (strike_zone_params['x_start'], strike_zone_params['y_start']),
                strike_zone_params['width'],
                strike_zone_params['height'],
                edgecolor='black',  # Black edge color for the strike zone
                facecolor='none',
                linewidth=2
            )
            ax.add_patch(strike_zone)
            
            # Set axis limits and remove ticks
            ax.set_xlim(-2, 2)
            ax.set_ylim(1, 4)
            ax.set_xticks([])  # Remove x-ticks
            ax.set_yticks([])  # Remove y-ticks
            
            # Remove axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Set pitch type as title
            ax.set_title(f"{pitch_type} ({pitcher_name})", fontsize=24)  # Increased font size

            # Equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
        
        # Remove any unused subplots
        for j in range(len(unique_pitch_types), len(axes)):
            fig.delaxes(axes[j])

        # Add a main title for all the heatmaps
        season = pitcher_data['Season'].iloc[0] if 'Season' in pitcher_data.columns else "Unknown"
        plt.suptitle(f"{pitcher_name} {map_type} Heatmap ({season} Season)", 
                     fontsize=30, fontweight='bold')
        
        # Adjust the layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for suptitle
        
        # Show the updated figure
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error generating {map_type} heatmaps: {e}")











# Function to calculate InZone% and Chase%
def calculate_in_zone(df):
    # Strike zone boundaries
    in_zone = df[
        (df['PlateLocHeight'] >= 1.5) & 
        (df['PlateLocHeight'] <= 3.3775) & 
        (df['PlateLocSide'] >= -0.83083) & 
        (df['PlateLocSide'] <= 0.83083)
    ]
    return in_zone

# Function to manually format the dataframe before displaying
# Function to manually format the dataframe before displaying, with alternating row colors
# Function to manually format the dataframe before displaying, with rounding and alternating row colors
# Function to manually format the dataframe before displaying, with rounding and alternating row colors
# Function to manually format the dataframe before displaying (no alternating row colors)
def format_dataframe(df):
    df = df.copy()  # Create a copy to avoid warnings
    percent_columns = ['InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%']

    # Format percentages and floats
    for col in df.columns:
        if col in percent_columns:
            df[col] = df[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) and isinstance(x, (int, float)) else 'N/A')  # Add % symbol
        elif df[col].dtype.kind in 'f':  # if it's a float type column
            df[col] = df[col].apply(lambda x: round(x, 2) if pd.notna(x) else 'N/A')
        else:
            df[col] = df[col].fillna('N/A')  # Fill NaN with N/A for non-float columns

    return df

# Load CLASS+ CSV into a DataFrame
class_plus_file_path = "player_pitch_summary_with_count (10).csv"  # Replace with the actual path

@st.cache_data
def load_class_plus_data(file_path):
    df = pd.read_csv(file_path)
    
    # Rename Pitch Types in CLASS+ DataFrame to match Streamlit app
    pitch_type_mapping = {
        "4S": "Fastball",
        "SI": "Sinker",
        "FC": "Cutter",
        "SL": "Slider",
        "CU": "Curveball",
        "FS": "Splitter",
        "CH": "ChangeUp"
    }
    df["PitchType"] = df["PitchType"].map(pitch_type_mapping)
    
    return df

class_plus_df = load_class_plus_data(class_plus_file_path)


# File path for Winter CLASS+ dataset
winter_class_plus_file_path = "winter_CLASS+.csv"

# Load Winter CLASS+ CSV
@st.cache_data
def load_winter_class_plus_data(file_path):
    df = pd.read_csv(file_path)
    df['Season'] = 'Winter'  # Add season identifier
    # Rename pitch types to match other datasets
    df["PitchType"] = df["PitchType"].map({
        "4S": "Fastball",
        "SI": "Sinker",
        "FC": "Cutter",
        "SL": "Slider",
        "CU": "Curveball",
        "FS": "Splitter",
        "CH": "ChangeUp"
    })
    return df

# Load the Winter CLASS+ dataset
winter_class_plus_df = load_winter_class_plus_data(winter_class_plus_file_path)


# File path for Spring CLASS+ dataset
spring_class_plus_file_path = "spring_CLASS+.csv"

season_class_plus_file_path = "2025_CLASS+.csv"

# Load Spring CLASS+ CSV
@st.cache_data
def load_spring_class_plus_data(file_path):
    df = pd.read_csv(file_path)
    df['Season'] = 'Spring Preseason'  # Add season identifier
    # Rename pitch types to match other datasets
    df["PitchType"] = df["PitchType"].map({
        "4S": "Fastball",
        "SI": "Sinker",
        "FC": "Cutter",
        "SL": "Slider",
        "CU": "Curveball",
        "FS": "Splitter",
        "CH": "ChangeUp"
    })
    return df

# Load the Spring CLASS+ dataset
spring_class_plus_df = load_spring_class_plus_data(spring_class_plus_file_path)


# Load Spring CLASS+ CSV
@st.cache_data
def load_season_class_plus_data(file_path):
    df = pd.read_csv(file_path)
    df['Season'] = '2025 Season'  # Add season identifier
    # Rename pitch types to match other datasets
    df["PitchType"] = df["PitchType"].map({
        "4S": "Fastball",
        "SI": "Sinker",
        "FC": "Cutter",
        "SL": "Slider",
        "CU": "Curveball",
        "FS": "Splitter",
        "CH": "ChangeUp"
    })
    return df

# Load the Spring CLASS+ dataset
season_class_plus_df = load_season_class_plus_data(season_class_plus_file_path)


# Add Season identifier to Fall CLASS+ dataset
class_plus_df['Season'] = 'Fall'

# Combine Fall and Winter CLASS+ datasets
all_class_plus_df = pd.concat([class_plus_df, winter_class_plus_df, spring_class_plus_df,season_class_plus_df])



def generate_pitch_traits_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on input parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        # Check if filtered data is empty
        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Group by 'TaggedPitchType' and calculate aggregated metrics
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
            VertRelAngle=('VertRelAngle', 'mean')
        ).reset_index()

        # Rename columns for clarity
        rename_columns = {
            'TaggedPitchType': 'Pitch',
            'RelSpeed': 'Velo',
            'InducedVertBreak': 'iVB',
            'HorizontalBreak': 'HB',
            'SpinRate': 'Spin',
            'RelHeight': 'RelH',
            'RelSide': 'RelS',
            'Extension': 'Ext',
            'VertApprAngle': 'VAA',
            'VertRelAngle': 'VRA'
        }
        grouped_data = grouped_data.rename(columns=rename_columns)

        # Convert all numeric columns to numeric format and coerce errors to NaN
        numeric_columns = ['Velo', 'iVB', 'HB', 'Spin', 'RelH', 'RelS', 'Ext', 'VAA','VRA']
        for col in numeric_columns:
            grouped_data[col] = pd.to_numeric(grouped_data[col], errors='coerce')

        # Round numeric columns to 1 decimal place
        grouped_data[numeric_columns] = grouped_data[numeric_columns].apply(lambda x: x.round(1))

        # Select CLASS+ data based on dataset selection
        # Select CLASS+ data based on dataset selection
        if dataset_selection == 'Fall':
            filtered_class_plus = class_plus_df[class_plus_df["playerFullName"] == pitcher_name]
        elif dataset_selection == 'Winter':
            filtered_class_plus = winter_class_plus_df[winter_class_plus_df["playerFullName"] == pitcher_name]
        elif dataset_selection == 'Spring Preseason':
            filtered_class_plus = spring_class_plus_df[spring_class_plus_df["playerFullName"] == pitcher_name]
        elif dataset_selection == '2025 Season':
            filtered_class_plus = season_class_plus_df[season_class_plus_df["playerFullName"] == pitcher_name]
        else:  # "All"
            filtered_class_plus = all_class_plus_df[all_class_plus_df["playerFullName"] == pitcher_name]

            filtered_class_plus = (
                filtered_class_plus.groupby("PitchType")
                .apply(lambda x: pd.Series({
                    "CLASS+": np.average(x["CLASS+"], weights=x["Count"]) if "Count" in x else x["CLASS+"].mean()
                }))
                .reset_index()
            )

        # Merge aggregated data with CLASS+ scores
        grouped_data = pd.merge(
            grouped_data,
            filtered_class_plus[["PitchType", "CLASS+"]],  # Select only relevant columns
            how="left",
            left_on="Pitch",
            right_on="PitchType"
        )

        # Drop redundant 'PitchType' column and fill missing CLASS+ scores with "N/A"
        grouped_data = grouped_data.drop(columns=["PitchType"], errors="ignore")
        grouped_data["CLASS+"] = pd.to_numeric(grouped_data["CLASS+"], errors="coerce").fillna("N/A")

        # Sort by 'Count' (most frequently thrown pitches first)
        grouped_data = grouped_data.sort_values(by='Count', ascending=False)

        # Calculate "All" row
        total_count = grouped_data["Count"].sum()
        weighted_averages = {
            column: np.average(
                grouped_data[column].dropna(), weights=grouped_data["Count"].loc[grouped_data[column].notna()]
            ) if grouped_data[column].notna().any() else "N/A"
            for column in numeric_columns
        }

        # Calculate the weighted average for CLASS+
        valid_class_plus = grouped_data.loc[grouped_data["CLASS+"] != "N/A", "CLASS+"].astype(float)
        valid_class_plus_weights = grouped_data.loc[grouped_data["CLASS+"] != "N/A", "Count"]
        class_plus_weighted_avg = (
            np.average(valid_class_plus, weights=valid_class_plus_weights) if not valid_class_plus.empty else "N/A"
        )

        all_row = {
            'Pitch': 'All',
            'Count': total_count,
            'Velo': round(weighted_averages['Velo'], 1) if pd.notna(weighted_averages['Velo']) else 'N/A',
            'iVB': round(weighted_averages['iVB'], 1) if pd.notna(weighted_averages['iVB']) else 'N/A',
            'HB': round(weighted_averages['HB'], 1) if pd.notna(weighted_averages['HB']) else 'N/A',
            'Spin': round(weighted_averages['Spin'], 1) if pd.notna(weighted_averages['Spin']) else 'N/A',
            'RelH': round(weighted_averages['RelH'], 1) if pd.notna(weighted_averages['RelH']) else 'N/A',
            'RelS': round(weighted_averages['RelS'], 1) if pd.notna(weighted_averages['RelS']) else 'N/A',
            'Ext': round(weighted_averages['Ext'], 1) if pd.notna(weighted_averages['Ext']) else 'N/A',
            'VAA': round(weighted_averages['VAA'], 1) if pd.notna(weighted_averages['VAA']) else 'N/A',
            'CLASS+': round(class_plus_weighted_avg, 1) if pd.notna(class_plus_weighted_avg) else 'N/A'
        }

        # Append "All" row to the DataFrame
        all_row_df = pd.DataFrame([all_row])
        grouped_data = pd.concat([grouped_data, all_row_df], ignore_index=True)

        # Format the data before displaying
        formatted_data = format_dataframe(grouped_data)

        # Display the results in Streamlit
        st.subheader("Pitch Traits:")
        st.dataframe(formatted_data)
    except KeyError as ke:
        st.error(f"Key error encountered: {ke}. Please check the input data and column names.")
    except Exception as e:
        st.error(f"An error occurred while generating the pitch traits table: {e}")





def generate_plate_discipline_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on input parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        # Check if filtered data is empty
        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Total number of pitches for percentage calculations
        total_pitches = len(pitcher_data)

        # Function to calculate plate discipline metrics
        def calculate_metrics(df):
            # Determine pitches in the strike zone
            in_zone_pitches = calculate_in_zone(df)
            total_in_zone = len(in_zone_pitches)

            # Define swing-related conditions
            swing_conditions = ['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay']
            total_swings = df[df['PitchCall'].isin(swing_conditions)].shape[0]
            total_whiffs = df[df['PitchCall'] == 'StrikeSwinging'].shape[0]
            total_chase = df[
                (~df.index.isin(in_zone_pitches.index)) & 
                df['PitchCall'].isin(swing_conditions)
            ].shape[0]

            # Whiffs in the zone
            in_zone_whiffs = in_zone_pitches[in_zone_pitches['PitchCall'] == 'StrikeSwinging'].shape[0]

            # Define strike-related conditions
            strike_conditions = ['StrikeCalled', 'FoulBallFieldable', 'FoulBallNotFieldable', 'StrikeSwinging', 'InPlay']
            total_strikes = df[df['PitchCall'].isin(strike_conditions)].shape[0]

            # Calculate metrics
            metrics = {
                'InZone%': (total_in_zone / len(df)) * 100 if len(df) > 0 else 0,
                'Swing%': (total_swings / len(df)) * 100 if len(df) > 0 else 0,
                'Whiff%': (total_whiffs / total_swings) * 100 if total_swings > 0 else 0,
                'Chase%': (total_chase / total_swings) * 100 if total_swings > 0 else 0,
                'InZoneWhiff%': (in_zone_whiffs / total_in_zone) * 100 if total_in_zone > 0 else 0,
                'Strike%': (total_strikes / len(df)) * 100 if len(df) > 0 else 0
            }
            return metrics

        # Group data by pitch type and calculate metrics
        plate_discipline_data = pitcher_data.groupby('TaggedPitchType').apply(calculate_metrics).apply(pd.Series).reset_index()

        # Calculate pitch percentage for each pitch type
        plate_discipline_data['Count'] = pitcher_data.groupby('TaggedPitchType')['TaggedPitchType'].count().values
        plate_discipline_data['Pitch%'] = (plate_discipline_data['Count'] / total_pitches) * 100

        # Reorder columns for display
        plate_discipline_data = plate_discipline_data[['TaggedPitchType', 'Count', 'Pitch%', 'Strike%', 'InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%']]

        # Rename columns for readability
        rename_columns = {
            'TaggedPitchType': 'Pitch',
            'Count': 'Count',
            'Pitch%': 'Pitch%',
            'Strike%': 'Strike%',
            'InZone%': 'InZone%',
            'Swing%': 'Swing%',
            'Whiff%': 'Whiff%',
            'Chase%': 'Chase%',
            'InZoneWhiff%': 'InZoneWhiff%'
        }
        plate_discipline_data = plate_discipline_data.rename(columns=rename_columns)

        # Calculate aggregate "All" row
        in_zone_pitches = calculate_in_zone(pitcher_data)
        total_swings = pitcher_data[pitcher_data['PitchCall'].isin(['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay'])].shape[0]
        total_whiffs = pitcher_data[pitcher_data['PitchCall'] == 'StrikeSwinging'].shape[0]
        total_chase = pitcher_data[
            (~pitcher_data.index.isin(in_zone_pitches.index)) & 
            pitcher_data['PitchCall'].isin(['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay'])
        ].shape[0]
        in_zone_whiffs = in_zone_pitches[in_zone_pitches['PitchCall'] == 'StrikeSwinging'].shape[0]
        total_strikes = pitcher_data[pitcher_data['PitchCall'].isin(['StrikeCalled', 'FoulBallFieldable', 'FoulBallNotFieldable', 'StrikeSwinging', 'InPlay'])].shape[0]

        all_row = {
            'Pitch': 'All',
            'Count': total_pitches,  # Total number of pitches
            'Pitch%': 100.0,  # Percentage is 100 for aggregate
            'Strike%': (total_strikes / total_pitches) * 100,
            'InZone%': (in_zone_pitches.shape[0] / total_pitches) * 100,
            'Swing%': (total_swings / total_pitches) * 100,
            'Whiff%': (total_whiffs / total_swings) * 100 if total_swings > 0 else 0,
            'Chase%': (total_chase / total_swings) * 100 if total_swings > 0 else 0,
            'InZoneWhiff%': (in_zone_whiffs / in_zone_pitches.shape[0]) * 100 if in_zone_pitches.shape[0] > 0 else 0
        }

        # Append "All" row to the DataFrame
        all_row_df = pd.DataFrame([all_row])
        plate_discipline_data = pd.concat([plate_discipline_data, all_row_df], ignore_index=True)

        # Format the DataFrame for display
        formatted_data = format_dataframe(plate_discipline_data)

        # Display the results in Streamlit
        st.subheader("Plate Discipline:")
        st.dataframe(formatted_data)
    except Exception as e:
        st.error(f"An error occurred while generating the plate discipline table: {e}")


# Define a color dictionary for each pitch type
color_dict = {
    'Fastball': 'blue',
    'Sinker': 'gold',
    'Slider': 'green',
    'Curveball': 'red',
    'Cutter': 'orange',
    'ChangeUp': 'purple',
    'Splitter': 'teal',
    'Unknown': 'black',
    'Other': 'black'
}



import plotly.express as px
import plotly.graph_objects as go

import plotly.graph_objects as go

import plotly.graph_objects as go

def plot_pitch_movement(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on selected parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Drop NaN values for plotting
        movement_data = pitcher_data.dropna(subset=['InducedVertBreak', 'HorzBreak', 'RelSpeed', 'Date'])

        if movement_data.empty:
            st.write("No pitch movement data available for plotting.")
            return

        # Define Plotly color equivalents for pitch types
        plotly_color_dict = {
            'Fastball': 'royalblue',
            'Sinker': 'goldenrod',
            'Slider': 'mediumseagreen',
            'Curveball': 'firebrick',
            'Cutter': 'darkorange',
            'ChangeUp': 'mediumpurple',
            'Splitter': 'teal',
            'Unknown': 'black',
            'Other': 'black'
        }

        # Create Plotly figure
        fig = go.Figure()

        # Add **background reference lines** first so they appear **below** the scatter points
        fig.add_shape(
            type="line",
            x0=0, x1=0, y0=-25, y1=25,
            line=dict(color="black", width=2),
            layer="below"  # Keeps the line in the background
        )
        fig.add_shape(
            type="line",
            x0=-25, x1=25, y0=0, y1=0,
            line=dict(color="black", width=2),
            layer="below"  # Keeps the line in the background
        )

        # Ensure the black x-axis line stays visible
        fig.update_xaxes(
            zeroline=True, zerolinewidth=2, zerolinecolor='black'  # Forces the x-axis black line
        )
        fig.update_yaxes(
            zeroline=True, zerolinewidth=2, zerolinecolor='black'  # Forces the y-axis black line
        )

        # Get unique pitch types
        unique_pitch_types = movement_data['TaggedPitchType'].unique()

        for pitch_type in unique_pitch_types:
            pitch_data = movement_data[movement_data['TaggedPitchType'] == pitch_type]

            # Round numeric values for hover info
            pitch_data['RelSpeed'] = pitch_data['RelSpeed'].round(1)
            pitch_data['InducedVertBreak'] = pitch_data['InducedVertBreak'].round(1)
            pitch_data['HorzBreak'] = pitch_data['HorzBreak'].round(1)


            # Add scatter points **AFTER** the border lines to keep them on top
            fig.add_trace(go.Scatter(
                x=pitch_data['HorzBreak'],
                y=pitch_data['InducedVertBreak'],
                mode='markers',
                name=pitch_type,
                marker=dict(
                    size=9,  # Slightly larger for better visibility
                    color=plotly_color_dict.get(pitch_type, 'black'),
                    opacity=0.8,
                    line=dict(width=1, color="white")  # White edge for better contrast
                ),
                text=pitch_data.apply(
                    lambda row: f"Date: {row['Date']}<br>RelSpeed: {row['RelSpeed']}<br>iVB: {row['InducedVertBreak']}<br>HB: {row['HorzBreak']}<br>Pitch#: {row['PitchNo']}",
                    axis=1
                ),
                hoverinfo='text'
            ))

        # Customize layout
        fig.update_layout(
            title=f"Pitch Movement for {pitcher_name}",
            xaxis=dict(title="Horizontal Break (inches)", range=[-30, 30]),
            yaxis=dict(title="Induced Vertical Break (inches)", range=[-30, 30]),
            template="plotly_white",
            showlegend=True,
            width=800,
            height=700
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while generating the pitch movement graph: {e}")






# Function to generate the Batted Ball table
def generate_batted_ball_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on the provided filters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Categorize batted balls into types based on angle
        def categorize_batted_type(angle):
            if angle < 10:
                return "GroundBall"
            elif 10 <= angle < 25:
                return "LineDrive"
            elif 25 <= angle < 50:
                return "FlyBall"
            else:
                return "PopUp"

        # Create 'BattedType' column
        pitcher_data['BattedType'] = pitcher_data['Angle'].apply(categorize_batted_type)

        # Filter rows where PitchCall is 'InPlay' for BIP calculations
        batted_data = pitcher_data[pitcher_data['PitchCall'] == 'InPlay']

        # Group by pitch type and calculate metrics
        batted_ball_summary = batted_data.groupby('TaggedPitchType').agg(
            BIP=('PitchCall', 'size'),
            GB=('BattedType', lambda x: (x == "GroundBall").sum()),
            FB=('BattedType', lambda x: (x == "FlyBall").sum()),
            EV=('ExitSpeed', 'mean'),
            Hard=('ExitSpeed', lambda x: (x >= 95).sum()),
            Soft=('ExitSpeed', lambda x: (x < 95).sum())
        ).reset_index()

        # Ensure all pitch types are included
        unique_pitch_types = pitcher_data['TaggedPitchType'].unique()
        full_summary = pd.DataFrame({'TaggedPitchType': unique_pitch_types})
        batted_ball_summary = pd.merge(full_summary, batted_ball_summary, on='TaggedPitchType', how='left')

        # Fill missing values with defaults
        batted_ball_summary[['BIP', 'GB', 'FB', 'EV', 'Hard', 'Soft']] = batted_ball_summary[
            ['BIP', 'GB', 'FB', 'EV', 'Hard', 'Soft']
        ].fillna(0)

        # Add total pitch counts for each type
        pitch_counts = pitcher_data.groupby('TaggedPitchType')['PitchCall'].count().reset_index(name='Count')
        batted_ball_summary = pd.merge(batted_ball_summary, pitch_counts, on='TaggedPitchType', how='left')

        # Calculate percentages
        batted_ball_summary['GB%'] = ((batted_ball_summary['GB'] / batted_ball_summary['BIP']) * 100).fillna(0).round(1).astype(str) + '%'
        batted_ball_summary['FB%'] = ((batted_ball_summary['FB'] / batted_ball_summary['BIP']) * 100).fillna(0).round(1).astype(str) + '%'
        batted_ball_summary['Hard%'] = ((batted_ball_summary['Hard'] / batted_ball_summary['BIP']) * 100).fillna(0).round(1).astype(str) + '%'
        batted_ball_summary['Soft%'] = ((batted_ball_summary['Soft'] / batted_ball_summary['BIP']) * 100).fillna(0).round(1).astype(str) + '%'

        # Calculate Contact%
        def calculate_contact(df):
            swings = df[df['PitchCall'].isin(['StrikeSwinging', 'InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])].shape[0]
            contact = df[df['PitchCall'].isin(['InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])].shape[0]
            return (contact / swings * 100) if swings > 0 else 0

        contact_values = []
        for pitch_type in batted_ball_summary['TaggedPitchType']:
            contact_values.append(
                calculate_contact(pitcher_data[pitcher_data['TaggedPitchType'] == pitch_type])
            )
        batted_ball_summary['Contact%'] = [f"{round(val, 1)}%" for val in contact_values]

        # Drop intermediate columns
        batted_ball_summary = batted_ball_summary.drop(columns=['GB', 'FB', 'Hard', 'Soft'])

        # Rename columns for display
        rename_columns = {
            'TaggedPitchType': 'Pitch',
            'Count': 'Count',
            'BIP': 'BIP',
            'EV': 'EV',
            'GB%': 'GB%',
            'FB%': 'FB%',
            'Hard%': 'Hard%',
            'Soft%': 'Soft%',
            'Contact%': 'Contact%'
        }
        batted_ball_summary = batted_ball_summary.rename(columns=rename_columns)

        # Calculate "All" row
        all_row = {
            'Pitch': 'All',
            'Count': pitcher_data.shape[0],
            'BIP': batted_data.shape[0],
            'EV': batted_data['ExitSpeed'].mean() if batted_data.shape[0] > 0 else 0,
            'GB%': f"{round((batted_data['BattedType'] == 'GroundBall').sum() / batted_data.shape[0] * 100, 1) if batted_data.shape[0] > 0 else 0}%",
            'FB%': f"{round((batted_data['BattedType'] == 'FlyBall').sum() / batted_data.shape[0] * 100, 1) if batted_data.shape[0] > 0 else 0}%",
            'Hard%': f"{round((batted_data['ExitSpeed'] >= 95).sum() / batted_data.shape[0] * 100, 1) if batted_data.shape[0] > 0 else 0}%",
            'Soft%': f"{round((batted_data['ExitSpeed'] < 95).sum() / batted_data.shape[0] * 100, 1) if batted_data.shape[0] > 0 else 0}%",
            'Contact%': f"{round(calculate_contact(pitcher_data), 1)}%"
        }
        all_row_df = pd.DataFrame([all_row])
        batted_ball_summary = pd.concat([batted_ball_summary, all_row_df], ignore_index=True)

        # Format the data for display
        formatted_data = format_dataframe(batted_ball_summary)

        # Display the table in Streamlit
        st.subheader("Batted Ball Summary")
        st.dataframe(formatted_data)

    except Exception as e:
        st.error(f"Error generating batted ball table: {e}")



import plotly.express as px



def generate_rolling_line_graphs(
    rolling_df, pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date
):
    try:
        # Sidebar toggle to choose between Full Rolling or Pitch-by-Pitch
        if date_filter_option == "Single Date":
            view_option = st.sidebar.radio(
                "Select Rolling View:",
                options=["Full Dataset Rolling Averages", "Pitch-by-Pitch (Single Date)"],
                index=0  # Default to full dataset
            )
        else:
            view_option = "Full Dataset Rolling Averages"  # Default for other date selections

        # Filter only by pitcher name (keep full data for rolling averages)
        full_filtered_data = rolling_df[rolling_df['playerFullName'] == pitcher_name]

        if full_filtered_data.empty:
            st.write("No data available for the selected pitcher.")
            return

        # Ensure numeric conversion for selected metrics
        numeric_columns = {
            'Vel': 'Velocity',
            'IndVertBrk': 'iVB',
            'HorzBrk': 'HB',
            'Spin': 'Spin',
            'RelH (ft)': 'RelH',
            'Extension': 'Extension',
            'CLASS+': 'CLASS+',
        }
        for col in numeric_columns.keys():
            full_filtered_data[col] = pd.to_numeric(full_filtered_data[col], errors='coerce')

        # Convert Date column to datetime and drop NaN dates
        full_filtered_data['Date'] = pd.to_datetime(full_filtered_data['Date'], errors='coerce')
        full_filtered_data = full_filtered_data.dropna(subset=['Date'])

        # Rename pitch types for clarity
        pitch_type_mapping = {
            "4S": "Fastball",
            "SI": "Sinker",
            "FC": "Cutter",
            "SL": "Slider",
            "CU": "Curveball",
            "FS": "Splitter",
            "CH": "ChangeUp",
        }
        full_filtered_data['PitchType'] = full_filtered_data['PitchType'].map(pitch_type_mapping)

        # Get unique pitch types
        unique_pitch_types = full_filtered_data['PitchType'].unique()

        # Define color mapping
        color_dict = {
            'Fastball': 'blue',
            'Sinker': 'gold',
            'Slider': 'green',
            'Curveball': 'red',
            'Cutter': 'orange',
            'ChangeUp': 'purple',
            'Splitter': 'teal',
            'Unknown': 'black',
            'Other': 'black'
        }

        ### **Option 1: Full Dataset Rolling Averages**
        if view_option == "Full Dataset Rolling Averages":
            # Sort data by date for proper rolling trend
            rolling_data = (
                full_filtered_data.groupby(['Date', 'PitchType'])
                .agg({col: 'mean' for col in numeric_columns.keys()})
                .reset_index()
                .sort_values(by="Date")
            )

            st.subheader("Rolling Averages Across Full Database")

            for metric, metric_label in numeric_columns.items():
                fig = px.line(
                    rolling_data,
                    x="Date",
                    y=metric,
                    color="PitchType",
                    title=f"{metric_label} Rolling Averages by Pitch Type (Full Dataset)",
                    labels={"Date": "Date", metric: metric_label, "PitchType": "Pitch Type"},
                    color_discrete_map=color_dict,
                    hover_data={"Date": "|%B %d, %Y", metric: ":.2f"},
                )

                # Scatter points for each date
                for pitch_type in unique_pitch_types:
                    pitch_subset = rolling_data[rolling_data['PitchType'] == pitch_type]
                    fig.add_scatter(
                        x=pitch_subset['Date'],
                        y=pitch_subset[metric],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=color_dict.get(pitch_type, 'black'),
                            opacity=0.7
                        ),
                        name=f"{pitch_type} Dots",
                        showlegend=False
                    )

                # Highlight selected date(s)
                if date_filter_option == "Single Date" and selected_date:
                    selected_datetime = pd.to_datetime(selected_date)
                    fig.add_vrect(x0=selected_datetime, x1=selected_datetime, fillcolor="gray", opacity=0.3, line_width=0)
                elif date_filter_option == "Date Range" and start_date and end_date:
                    start_datetime, end_datetime = pd.to_datetime(start_date), pd.to_datetime(end_date)
                    fig.add_vrect(x0=start_datetime, x1=end_datetime, fillcolor="gray", opacity=0.3, line_width=0)

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=metric_label,
                    legend_title="Pitch Type",
                    template="plotly_white",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

        ### **Option 2: Pitch-by-Pitch View (Only for Single Date)**
        elif view_option == "Pitch-by-Pitch (Single Date)":
            # Filter data by selected date
            selected_datetime = pd.to_datetime(selected_date)
            pitch_data = full_filtered_data[full_filtered_data['Date'].dt.date == selected_datetime.date()]

            if pitch_data.empty:
                st.write("No data available for the selected date.")
                return

            # Ensure PitchNo is numeric and sort
            pitch_data['PitchNo'] = pd.to_numeric(pitch_data['PitchNo'], errors='coerce')
            pitch_data = pitch_data.dropna(subset=['PitchNo']).sort_values(by="PitchNo")

            st.subheader(f"Pitch-by-Pitch View for {selected_date.strftime('%B %d, %Y')}")

            for metric, metric_label in numeric_columns.items():
                fig = px.line(
                    pitch_data,
                    x="PitchNo",
                    y=metric,
                    color="PitchType",
                    title=f"{metric_label} Pitch-by-Pitch",
                    labels={"PitchNo": "Pitch Number", metric: metric_label, "PitchType": "Pitch Type"},
                    color_discrete_map=color_dict,
                    hover_data={"PitchNo": ":.0f", metric: ":.2f"},
                )

                # Scatter points for each pitch
                for pitch_type in unique_pitch_types:
                    pitch_subset = pitch_data[pitch_data['PitchType'] == pitch_type]
                    fig.add_scatter(
                        x=pitch_subset['PitchNo'],
                        y=pitch_subset[metric],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_dict.get(pitch_type, 'black')
                        ),
                        name=f"{pitch_type} Dots",
                        showlegend=False
                    )

                # Set x-axis to match smallest to largest pitch number for the day
                fig.update_xaxes(range=[pitch_data['PitchNo'].min() - 1, pitch_data['PitchNo'].max() + 1])

                fig.update_layout(
                    xaxis_title="Pitch Number",
                    yaxis_title=metric_label,
                    legend_title="Pitch Type",
                    template="plotly_white",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while generating rolling line graphs: {e}")



import plotly.graph_objects as go

def plot_release_and_approach_angles(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on selected parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Drop NaN values for plotting
        release_data = pitcher_data.dropna(subset=['HorzRelAngle', 'VertRelAngle'])
        approach_data = pitcher_data.dropna(subset=['HorzApprAngle', 'VertApprAngle'])

        if release_data.empty and approach_data.empty:
            st.write("No angle data available for plotting.")
            return

        # Define Plotly color equivalents for pitch types
        plotly_color_dict = {
            'Fastball': 'royalblue',
            'Sinker': 'goldenrod',
            'Slider': 'mediumseagreen',
            'Curveball': 'firebrick',
            'Cutter': 'darkorange',
            'ChangeUp': 'mediumpurple',
            'Splitter': 'teal',
            'Unknown': 'black',
            'Other': 'black'
        }

        # Function to create a scatter plot with bounding circles and average values
        def create_scatter_plot(data, x_col, y_col, title, x_lim, y_lim):
            fig = go.Figure()

            # Get unique pitch types
            unique_pitch_types = data['TaggedPitchType'].unique()

            for pitch_type in unique_pitch_types:
                pitch_type_data = data[data['TaggedPitchType'] == pitch_type]

                # Calculate mean and standard deviation for bounding circle
                mean_x = pitch_type_data[x_col].mean()
                mean_y = pitch_type_data[y_col].mean()
                std_dev_x = pitch_type_data[x_col].std()
                std_dev_y = pitch_type_data[y_col].std()

                # Format average values for legend
                avg_label = f"{pitch_type} ({mean_x:.1f}, {mean_y:.1f})"

                # Plot scatter points
                fig.add_trace(go.Scatter(
                    x=pitch_type_data[x_col],
                    y=pitch_type_data[y_col],
                    mode='markers',
                    name=avg_label,  # Use formatted label
                    marker=dict(
                        size=8,
                        color=plotly_color_dict.get(pitch_type, 'black'),
                        opacity=0.7
                    )
                ))

                # Draw bounding circle if data exists
                if not (pd.isna(mean_x) or pd.isna(mean_y) or pd.isna(std_dev_x) or pd.isna(std_dev_y)):
                    radius = max(std_dev_x, std_dev_y)  # Use the largest std dev
                    fig.add_shape(
                        type="circle",
                        xref="x", yref="y",
                        x0=mean_x - radius, y0=mean_y - radius,
                        x1=mean_x + radius, y1=mean_y + radius,
                        line=dict(color=plotly_color_dict.get(pitch_type, 'black'), width=2),
                        opacity=0.3
                    )

            # Customize layout with limits
            fig.update_layout(
                title=title,
                xaxis=dict(title=x_col, range=x_lim),
                yaxis=dict(title=y_col, range=y_lim),
                template="plotly_white",
                showlegend=True,
                width=800,
                height=700
            )

            return fig

        # Create and display the release angle plot
        if not release_data.empty:
            release_fig = create_scatter_plot(
                release_data, 
                'HorzRelAngle', 'VertRelAngle', 
                "Release Angles by Pitch Type", 
                x_lim=[-7.5, 7.5], 
                y_lim=[-5, 3]
            )
            st.plotly_chart(release_fig, use_container_width=True)

        # Create and display the approach angle plot
        if not approach_data.empty:
            approach_fig = create_scatter_plot(
                approach_data, 
                'HorzApprAngle', 'VertApprAngle', 
                "Approach Angles by Pitch Type", 
                x_lim=[-6, 6], 
                y_lim=[-12, 0]
            )
            st.plotly_chart(approach_fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while generating the angle plots: {e}")




tab1, tab2 = st.tabs(["Full Report", "Single Game Report"])




with tab1:
    # Generate heatmaps based on selections
    plot_heatmaps(
        pitcher_name, batter_side, strikes, balls,
        date_filter_option, selected_date, start_date, end_date,
        heatmap_type
    )

    generate_plate_discipline_table(
        pitcher_name, batter_side, strikes, balls,
        date_filter_option, selected_date, start_date, end_date
    )

    generate_pitch_traits_table(
        pitcher_name, batter_side, strikes, balls,
        date_filter_option, selected_date, start_date, end_date
    )

    generate_batted_ball_table(
        pitcher_name, batter_side, strikes, balls,
        date_filter_option, selected_date, start_date, end_date
    )

    plot_pitch_movement(
        pitcher_name, batter_side, strikes, balls,
        date_filter_option, selected_date, start_date, end_date
    )

    generate_rolling_line_graphs(
        rolling_df, pitcher_name, batter_side, strikes, balls,
        date_filter_option, selected_date, start_date, end_date
    )

    plot_release_and_approach_angles(
        pitcher_name, batter_side, strikes, balls,
        date_filter_option, selected_date, start_date, end_date
    )




def draw_single_game_report(pitcher_name, input_game_date, inngings_pitched, pitch_data, class_data):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    import numpy as np
    import pandas as pd
    from matplotlib.colors import to_hex, to_rgb
    from matplotlib import cm
    from matplotlib.lines import Line2D

    # === Helper functions ===
    def custom_blend_white_center(value, vmin, mid, vmax, low_color, high_color):
        if pd.isna(value):
            return "#D3D3D3"
        white_rgb = np.array([1.0, 1.0, 1.0])
        val = float(value)
        if val <= mid:
            norm = (val - vmin) / (mid - vmin)
            norm = min(max(norm, 0), 1)
            base_rgb = np.array(to_rgb(low_color))
            blended_rgb = base_rgb + (white_rgb - base_rgb) * norm
        else:
            norm = (val - mid) / (vmax - mid)
            norm = min(max(norm, 0), 1)
            base_rgb = np.array(to_rgb(high_color))
            blended_rgb = white_rgb + (base_rgb - white_rgb) * norm
        return to_hex(blended_rgb)

    def traffic_light_fip_color(fip):
        try:
            fip = float(str(fip).replace('%', '').strip())
            if np.isnan(fip): return "#D3D3D3"
        except: return "#D3D3D3"
        if fip <= 6:
            ratio = np.clip(fip / 6, 0, 1)
            red, green = ratio, 1.0
        else:
            ratio = np.clip((fip - 6) / 4, 0, 1)
            red, green = 1.0, 1.0 - ratio
        return to_hex((np.clip(red, 0, 1), np.clip(green, 0, 1), 0.0))

    def get_contrast_text_color(bg_hex):
        r, g, b = to_rgb(bg_hex)
        brightness = (r*299 + g*587 + b*114) / 1000
        return 'black' if brightness > 0.6 else 'white'

    def format_value(val, col=None):
        if col in ["SwStr%", "HardHit%", "% Thrown"]:
            return f"{val}%" if isinstance(val, (int, float)) else val
        return val

    # === Preprocess data ===
    parsed_game_date = pd.to_datetime(input_game_date)
    df = pitch_data[(pitch_data['Pitcher'] == pitcher_name) & (pitch_data['Date'] == parsed_game_date)]
    df_class = class_data[(class_data['playerFullName'] == pitcher_name) & (class_data['Date'] == parsed_game_date)]

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        ax.axis('off')
        return fig

    # === Calculate basic stats ===
    IP = inngings_pitched
    PA = len(df)
    R = df['RunsScored'].sum()
    ER = df['EarnedRuns'].sum() if 'EarnedRuns' in df.columns else R
    H = df[df['PlayResult'].isin(["Single", "Double", "Triple", "HomeRun"])].shape[0]
    K = df[df['KorBB'] == 'Strikeout'].shape[0]
    BB = df[df['KorBB'] == 'Walk'].shape[0]
    HBP = df[df['PitchCall'] == 'HitByPitch'].shape[0]
    HR = df[df['TaggedHitType'] == 'HomeRun'].shape[0]
    swings = df[df['PitchCall'].isin(['SwingingStrike', 'SwingingStrikeBlocked', 'Foul', 'InPlay'])]
    swstr = df[df['PitchCall'].isin(['SwingingStrike', 'SwingingStrikeBlocked'])]
    swstr_percent = round(100 * len(swstr) / len(swings), 1) if len(swings) > 0 else 0
    balls_in_play = df[df['PitchCall'] == 'InPlay']
    hard_hits = balls_in_play[balls_in_play['ExitSpeed'] >= 95]
    hard_hit_percent = round(100 * len(hard_hits) / len(balls_in_play), 1) if len(balls_in_play) > 0 else 0
    avg_class = int(df_class['CLASS+'].mean()) if not df_class.empty else 100
    fip_constant = 3.1
    FIP = round((13 * HR + 3 * (BB + HBP) - 2 * K) / IP + fip_constant, 2) if IP > 0 else np.nan

    # === Begin plotting ===
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(5, 3, height_ratios=[0.3, 2.2, 0.15, 2.5, 1.0])

    opponent = df['BatterTeam'].iloc[0] if 'BatterTeam' in df.columns and not df.empty else "Opponent"
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.set_title(
        f"{pitcher_name}  |  {parsed_game_date.date()} vs {opponent}",
        fontsize=20, weight='bold', loc='left', pad=20
    )

    box_score_data = [
        ["IP", IP], ["P", PA], ["R", R], ["ER", ER], ["H", H],
        ["K", K], ["BB", BB], ["HBP", HBP], ["HR", HR],
        ["CLASS+", avg_class], ["SwStr%", f"{swstr_percent}%"], ["HardHit%", f"{hard_hit_percent}%"], ["FIP", FIP]
    ]

    cell_values = [format_value(val) for _, val in box_score_data]
    col_headers = [label for label, _ in box_score_data]

    table = ax_title.table(
        cellText=[cell_values],
        colLabels=col_headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.1, 1.3)

    # === Continue plotting more content (release point, movement, heatmaps, etc.) ===
    # Keep extending with other axes (e.g., ax1 = fig.add_subplot(gs[1, 0]), etc.)
    pitch_colors = {
    "Fastball": "#1f77b4",     # Blue
    "Slider": "#2ca02c",       # Green
    "Curveball": "#d62728",    # Red
    "ChangeUp": "#9467bd",     # Purple
    "Sinker": "#ffdd57",       # Yellow
    "Cutter": "#ff7f0e",       # Orange
    "Splitter": "#17becf"      # Teal
}
fastball_blue = pitch_colors["Fastball"]
curveball_red = pitch_colors["Curveball"]
for j, label in enumerate(col_headers):
    cell = table[1, j]
    val = box_score_data[j][1]

    # Convert % strings to float if needed
    if isinstance(val, str) and val.endswith('%'):
        val = float(val.replace('%', ''))

    if label == "CLASS+":
        face = custom_blend_white_center(val, 70, 100, 120, fastball_blue, curveball_red)
    elif label == "SwStr%":
        face = custom_blend_white_center(val, 0, 13, 25, fastball_blue, curveball_red)
    elif label == "HardHit%":
        face = custom_blend_white_center(val, 0, 45, 100, curveball_red, fastball_blue)
    elif label == 'FIP':
        face = traffic_light_fip_color(val)
    else:
        continue

    cell.set_facecolor(face)
    cell.set_text_props(color=get_contrast_text_color(face))


table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.1, 1.3)





# === Release Point Plot (color-coded with mean markers) ===
ax1 = fig.add_subplot(gs[1, 0])
pitch_colors = {
    "Fastball": "#1f77b4",     # Blue
    "Slider": "#2ca02c",       # Green
    "Curveball": "#d62728",    # Red
    "ChangeUp": "#9467bd",     # Purple
    "Sinker": "#ffdd57",       # Yellow
    "Cutter": "#ff7f0e",       # Orange
    "Splitter": "#17becf"      # Teal
}


for pitch_type in df['TaggedPitchType'].unique():
    subset = df[df['TaggedPitchType'] == pitch_type]
    color = pitch_colors.get(pitch_type, "gray")

    # Plot individual pitches with lower opacity
    ax1.scatter(subset['RelSide'], subset['RelHeight'], color=color, alpha=0.4, label=pitch_type, s=25)

    # Plot average release point with full opacity + black edge
    avg_rel_side = subset['RelSide'].mean()
    avg_rel_height = subset['RelHeight'].mean()
    ax1.scatter(avg_rel_side, avg_rel_height, color=color, edgecolor='black', linewidth=1.2, s=100, zorder=5)

ax1.set_title("Release Point")
ax1.set_xlabel("RelSide")
ax1.set_ylabel("RelHeight")
ax1.axhline(0, color='gray', lw=0.5)
ax1.axvline(0, color='gray', lw=0.5)
ax1.set_ylim(3,7)
ax1.set_xlim(-3.5,3.5)

# === Movement Profile (updated with grid and averaged markers) ===
ax2 = fig.add_subplot(gs[1, 1])

# Draw grid lines first so they appear beneath points
for i in range(-30, 35, 5):  # horizontal gray grid lines
    ax2.axhline(i, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax2.axvline(i, color='gray', linestyle='--', linewidth=0.5, zorder=0)

# Origin lines (0,0) in black
ax2.axhline(0, color='black', linewidth=1, zorder=1)
ax2.axvline(0, color='black', linewidth=1, zorder=1)

# Plot pitches
for pitch in df['TaggedPitchType'].unique():
    subset = df[df['TaggedPitchType'] == pitch]
    color = pitch_colors.get(pitch, "gray")

    # Faded individual points
    ax2.scatter(subset['HorzBreak'], subset['InducedVertBreak'],
                color=color, alpha=0.4, label=pitch, s=25, zorder=2)

    # Average marker with black outline
    avg_x = subset['HorzBreak'].mean()
    avg_y = subset['InducedVertBreak'].mean()
    ax2.scatter(avg_x, avg_y, color=color, edgecolor='black',
                linewidth=1.2, s=100, zorder=3)

ax2.set_title("Movement Profile")
ax2.set_xlabel("Horz Break")
ax2.set_ylabel("IVB")
ax2.set_xlim(-30, 30)
ax2.set_ylim(-30, 30)

# === Rolling CLASS+ BY PITCH TYPE ===
ax3 = fig.add_subplot(gs[1, 2])
pitch_types = df_class['PitchType'].unique()

rolling_colors = {
    "4S": "#1f77b4",     # Blue
    "SL": "#2ca02c",       # Green
    "CU": "#d62728",    # Red
    "CH": "#9467bd",     # Purple
    "SI": "#ffdd57",       # Yellow
    "FC": "#ff7f0e",       # Orange
    "FS": "#17becf"      # Teal
}

for pitch_type in pitch_types:
    subset = df_class[df_class['PitchType'] == pitch_type].reset_index(drop=True)
    if len(subset) >= 5:
        rolling_avg = subset['CLASS+'].rolling(5).mean()
        color = rolling_colors.get(pitch_type, "gray")
        ax3.plot(rolling_avg, label=pitch_type, color=color, linewidth=2)


ax3.set_ylim(70, 130)
ax3.set_title("Rolling 5-Pitch CLASS+ by Type")
ax3.legend(fontsize=7, loc="lower right")


# === Strike Zone Visuals (3 for LHH, 3 for RHH) ===
from matplotlib import patches

# Plot containers
ax_zones = fig.add_subplot(gs[3, :])  # Full width for both sides
ax_zones.axis('off')


# Strike zone box dimensions
strike_zone_width = 17 / 12  # 1.41667 feet
zone_params = {'x': -strike_zone_width / 2, 'y': 1.5, 'width': strike_zone_width, 'height': 3.3775 - 1.5}
x_limits = [-2.5, 2.5]
y_limits = [0.5, 4.5]

# Subplot positions (relative to LHH/RHH subplot area)
zone_titles = ['Frequency', 'Whiff', 'HardHit']
zone_data_filters = [
    lambda df: df,  # Frequency
    lambda df: df[df['PitchCall'] == 'StrikeSwinging'],  # Whiff
    lambda df: df[(df['PitchCall'] == 'InPlay') & (df['ExitSpeed'] >= 95)]  # HardHit
]

def plot_zones(base_ax, side_df, side_title):
    for i, (label, data_filter) in enumerate(zip(zone_titles, zone_data_filters)):
        inset_ax = base_ax.inset_axes([i * 0.33 + 0.01, 0.02, 0.32, 0.96])


        zone_df = data_filter(side_df)
        for pitch_type in zone_df['TaggedPitchType'].unique():
            pt_df = zone_df[zone_df['TaggedPitchType'] == pitch_type]
            inset_ax.scatter(pt_df['PlateLocSide'], pt_df['PlateLocHeight'],
                             color=pitch_colors.get(pitch_type, 'gray'), s=20, alpha=0.8)

        # Strike zone box
        rect = patches.Rectangle((zone_params['x'], zone_params['y']),
                                 zone_params['width'], zone_params['height'],
                                 linewidth=1.5, edgecolor='black', facecolor='none')
        inset_ax.add_patch(rect)

        inset_ax.set_xlim(x_limits)
        inset_ax.set_ylim(y_limits)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_aspect('equal')
        title_text = f"{label}\n{side_title}" if label == 'Whiff' else label
        inset_ax.set_title(title_text, fontsize=9, weight='bold' if label == 'Whiff' else 'normal')

# Filter LHH and RHH
df_lhh = df[df['BatterSide'] == 'Left']
df_rhh = df[df['BatterSide'] == 'Right']

# Plot 3 strike zone heatmaps per side
def plot_zones_side_by_side(ax, df_lhh, df_rhh):
    side_titles = ['vs LHH', 'vs RHH']
    dfs = [df_lhh, df_rhh]
    
    for side_idx, side_df in enumerate(dfs):
        for i, (label, data_filter) in enumerate(zip(zone_titles, zone_data_filters)):
            # Positioning within the full-width ax_zones: [left, bottom, width, height]
            left = side_idx * 0.5 + i * (0.32 / 2) + 0.02
            bottom = 0.05
            width = 0.3 / 2
            height = 0.9
            
            inset_ax = ax.inset_axes([left, bottom, width, height])
            zone_df = data_filter(side_df)
            
            for pitch_type in zone_df['TaggedPitchType'].unique():
                pt_df = zone_df[zone_df['TaggedPitchType'] == pitch_type]
                inset_ax.scatter(pt_df['PlateLocSide'], pt_df['PlateLocHeight'],
                                 color=pitch_colors.get(pitch_type, 'gray'), s=20, alpha=0.8)

            # Strike zone box
            rect = patches.Rectangle((zone_params['x'], zone_params['y']),
                                     zone_params['width'], zone_params['height'],
                                     linewidth=1.5, edgecolor='black', facecolor='none')
            inset_ax.add_patch(rect)

            inset_ax.set_xlim(x_limits)
            inset_ax.set_ylim(y_limits)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_aspect('equal')
            title_text = f"{label}\n{side_titles[side_idx]}" 
            inset_ax.set_title(title_text, fontsize=9, weight='bold')


plot_zones_side_by_side(ax_zones, df_lhh, df_rhh)



# === Pitch Type Legend Below Zones ===
from matplotlib.lines import Line2D

# Create custom legend handles from pitch_colors
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=pitch)
    for pitch, color in pitch_colors.items()
    if pitch in df['TaggedPitchType'].unique()
]

# Add a new horizontal axis below the strike zone plots
legend_ax = fig.add_axes([0.25, 0.13, 0.5, 0.03])  # [left, bottom, width, height]
legend_ax.axis('off')
legend_ax.legend(handles=legend_handles, loc='center', ncol=len(legend_handles), frameon=False, fontsize=9)




# === Pitch Table ===
# === Pitch Table with Conditional Formatting ===
ax_table = fig.add_subplot(gs[4, :])
ax_table.axis('off')

# Sort the pitch summary by % Thrown descending before displaying
table_data = pitch_summary.reset_index().sort_values(by='% Thrown', ascending=False)
col_labels = ["Pitch", "%", "Vel (Max)", "Spin", "iVB", "HB", "VAA", "Ext", "VRA", "CLASS+", "SwStr%", "Hard%"]

cell_data = table_data.values.tolist()


tbl = ax_table.table(
    cellText=[[format_value(val, col_labels[j]) for j, val in enumerate(row)] for row in cell_data],
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)


tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.5)

# Use Fastball and Curveball colors for conditional colormaps
fastball_blue = pitch_colors["Fastball"]
curveball_red = pitch_colors["Curveball"]

for i, row in enumerate(cell_data):
    pitch_type = row[0]
    for j, val in enumerate(row):
        cell = tbl[i+1, j]
        if col_labels[j] == "Pitch":
            face = pitch_colors.get(pitch_type, "#E0E0E0")
        elif col_labels[j] == "CLASS+":
            face = custom_blend_white_center(val, 70, 100, 120, fastball_blue, curveball_red)
        elif col_labels[j] == "SwStr%":
            face = custom_blend_white_center(val, 0, 13, 25, fastball_blue, curveball_red)
        elif col_labels[j] == "Hard%":
            face = custom_blend_white_center(val, 0, 45, 100, curveball_red, fastball_blue)
        else:
            continue  # Skip if no coloring rule

        cell.set_facecolor(face)
        cell.set_text_props(color=get_contrast_text_color(face))

    

    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    return fig




    with tab2:
        import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        from your_visual_function_file import draw_single_game_report  # Update with your actual function file

        # === Load and preprocess data ===
        pitch_path = "2025_SEASON.csv"
        class_path = "2025_CLASS+_by_date.csv"
        pitch_df = pd.read_csv(pitch_path)
        class_df = pd.read_csv(class_path)

        pitch_df["Date"] = pd.to_datetime(pitch_df["Date"], errors="coerce")
        class_df["Date"] = pd.to_datetime(class_df["Date"], errors="coerce")

        # === UI Inputs ===
        st.header("Single Game Report")

        available_dates = sorted(pitch_df[pitch_df["PitcherTeam"] == "OLE_REB"]["Date"].dropna().unique())
        selected_date = st.date_input(
            "Select Game Date",
            value=available_dates[-1],
            min_value=min(available_dates),
            max_value=max(available_dates)
        )

        pitchers_on_date = sorted(
            pitch_df[
                (pitch_df["PitcherTeam"] == "OLE_REB") &
                (pitch_df["Date"] == pd.to_datetime(selected_date))
            ]["Pitcher"].unique()
        )
        selected_pitcher = st.selectbox("Select Pitcher", pitchers_on_date)

        ip = st.number_input("Enter Innings Pitched", min_value=0.0, max_value=9.0, step=0.1, value=1.0)

        # === Trigger Report ===
        if selected_pitcher and selected_date and ip > 0:
            fig = draw_single_game_report(
                pitcher_name=selected_pitcher,
                input_game_date=pd.to_datetime(selected_date),
                inngings_pitched=ip,
                pitch_data=pitch_df,
                class_data=class_df
            )
            st.pyplot(fig)
