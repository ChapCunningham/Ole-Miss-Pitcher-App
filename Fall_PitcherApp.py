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
                   'Extension', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle', 'ExitSpeed']

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
    rolling_df = pd.concat([fall_rolling_df, winter_rolling_df, spring_rolling_df], season_rolling_df)


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


# Add Season identifier to Fall CLASS+ dataset
class_plus_df['Season'] = 'Fall'

# Combine Fall and Winter CLASS+ datasets
all_class_plus_df = pd.concat([class_plus_df, winter_class_plus_df, spring_class_plus_df])



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
            VertApprAngle=('VertApprAngle', 'mean')
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
            'VertApprAngle': 'VAA'
        }
        grouped_data = grouped_data.rename(columns=rename_columns)

        # Convert all numeric columns to numeric format and coerce errors to NaN
        numeric_columns = ['Velo', 'iVB', 'HB', 'Spin', 'RelH', 'RelS', 'Ext', 'VAA']
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



def plot_pitch_movement(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on the selected parameters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Ensure InducedVertBreak and HorizontalBreak are available for plotting
        movement_data = pitcher_data.dropna(subset=['InducedVertBreak', 'HorzBreak'])

        if movement_data.empty:
            st.write("No pitch movement data available for plotting.")
            return

        # Set up the figure for pitch movement
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Set axis limits and labels
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_xlabel("Horizontal Break (inches)", fontsize=14)
        ax.set_ylabel("Induced Vertical Break (inches)", fontsize=14)

        # Add grid lines every 5 units
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add bold black lines through the origin
        plt.axhline(0, color='black', linewidth=2, zorder=1)  # Horizontal line through the origin
        plt.axvline(0, color='black', linewidth=2, zorder=1)  # Vertical line through the origin

        # Plot each pitch, colored by pitch type, with a higher z-order
        unique_pitch_types = movement_data['TaggedPitchType'].unique()
        for pitch_type in unique_pitch_types:
            pitch_type_data = movement_data[movement_data['TaggedPitchType'] == pitch_type]

            # Set color based on pitch type, default to black for unknown types
            color = color_dict.get(pitch_type, 'black')

            # Plot individual pitches
            plt.scatter(
                pitch_type_data['HorzBreak'],
                pitch_type_data['InducedVertBreak'],
                label=pitch_type,
                color=color,
                s=70,  # Adjust size of the markers
                alpha=0.7,
                zorder=2  # Higher z-order to plot above the lines
            )

            # Calculate the mean and standard deviation for clustering
            mean_horz = pitch_type_data['HorzBreak'].mean()
            mean_vert = pitch_type_data['InducedVertBreak'].mean()
            std_dev = np.sqrt(
                pitch_type_data['HorzBreak'].std()**2 + pitch_type_data['InducedVertBreak'].std()**2
            )

            # Draw a circle to represent the cluster area
            circle = plt.Circle((mean_horz, mean_vert), std_dev, color=color, alpha=0.3, zorder=1)
            ax.add_patch(circle)

        # Add a legend for pitch types
        plt.legend(title="Pitch Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        # Display the plot in Streamlit
        st.subheader(f"Pitch Movement Graph for {pitcher_name}")
        st.pyplot(plt)

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
        # Filter data based on the provided parameters
        filtered_data = rolling_df[rolling_df['playerFullName'] == pitcher_name]

        if filtered_data.empty:
            st.write("No data available for the selected pitcher.")
            return

        # Apply date filtering
        if date_filter_option == "Single Date" and selected_date:
            filtered_data = filtered_data[filtered_data['Date'] == pd.to_datetime(selected_date)]
        elif date_filter_option == "Date Range" and start_date and end_date:
            filtered_data = filtered_data[
                (filtered_data['Date'] >= pd.to_datetime(start_date)) &
                (filtered_data['Date'] <= pd.to_datetime(end_date))
            ]

        if filtered_data.empty:
            st.write("No data available for the selected date range.")
            return

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
        filtered_data['PitchType'] = filtered_data['PitchType'].map(pitch_type_mapping)

        # Ensure numeric conversion for the selected metrics
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
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

        # Group data by date and pitch type
        grouped_data = (
            filtered_data.groupby(['Date', 'PitchType'])
            .agg({col: 'mean' for col in numeric_columns.keys()})
            .reset_index()
        )

        # Get unique pitch types
        unique_pitch_types = grouped_data['PitchType'].unique()

        # Map colors using the color_dict
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

        # Plot rolling line graphs
        st.subheader("Interactive Rolling Averages by Pitch Type")

        for metric, metric_label in numeric_columns.items():
            fig = px.line(
                grouped_data,
                x="Date",
                y=metric,
                color="PitchType",
                title=f"{metric_label} Rolling Averages by Pitch Type",
                labels={"Date": "Date", metric: metric_label, "PitchType": "Pitch Type"},
                color_discrete_map=color_dict,  # Match colors with color_dict
                hover_data={"Date": "|%B %d, %Y", metric: ":.2f"},  # Format hover information
            )

            # Add scatter points for each date with data
            for pitch_type in unique_pitch_types:
                pitch_data = grouped_data[grouped_data['PitchType'] == pitch_type]
                fig.add_scatter(
                    x=pitch_data['Date'],
                    y=pitch_data[metric],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color_dict.get(pitch_type, 'black')  # Match marker color to line color
                    ),
                    name=f"{pitch_type} Dots",  # To differentiate scatter points in the legend
                    showlegend=False  # Hide extra legends for scatter points
                )

            # Customize layout
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=metric_label,
                legend_title="Pitch Type",
                template="plotly_white",
                hovermode="x unified",  # Show all hover info on the same vertical line
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while generating rolling line graphs: {e}")









# Generate heatmaps based on selections
# Generate heatmaps based on selections
plot_heatmaps(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date,
    heatmap_type  # Pass the selected heatmap type
)


# Generate and display the pitch traits and plate discipline tables
generate_plate_discipline_table(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date
)

generate_pitch_traits_table(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date
)

generate_batted_ball_table(
    pitcher_name,
    batter_side,
    strikes,
    balls,
    date_filter_option,
    selected_date,
    start_date,
    end_date
)


plot_pitch_movement(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date
)

# Generate rolling line graphs based on selected metrics and pitch types
generate_rolling_line_graphs(
    rolling_df,
    pitcher_name,
    batter_side,
    strikes,
    balls,
    date_filter_option,
    selected_date,
    start_date,
    end_date
)
