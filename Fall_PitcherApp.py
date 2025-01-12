import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import math
from datetime import datetime

# Load the real datase
file_path = 'FINAL FALL CSV 2024 - filtered_fall_trackman (1).csv'  # Replace with the correct ath in your Streamlit setup

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Date'])  # Parse 'Date' column as datetime

test_df = load_data(file_path)
test_df = test_df[test_df['PitcherTeam'] == 'OLE_REB']

# Ensure numeric conversion for the columns where aggregation will be done
numeric_columns = ['RelSpeed', 'SpinRate', 'Tilt', 'RelHeight', 'RelSide', 
                   'Extension', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle', 'ExitSpeed']

# Coerce non-numeric values to NaN
for col in numeric_columns:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# Streamlit app layout
st.title("OMBSB Fall Pitcher Reports")

# Sidebar for filters
st.sidebar.header("Filters")

# Dropdown widget to select the pitcher
pitcher_name = st.sidebar.selectbox(
    "Select Pitcher:",
    options=test_df['Pitcher'].unique()
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
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        value=[datetime.today(), datetime.today()]
    )

# Function to filter data based on the dropdown selections and date filters
def filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
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
        # Convert selected_date to datetime if necessary
        selected_datetime = pd.to_datetime(selected_date)
        pitcher_data = pitcher_data[pitcher_data['Date'].dt.date == selected_datetime.date()]
    elif date_filter_option == "Date Range" and start_date and end_date:
        # Convert start_date and end_date to datetime if necessary
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        pitcher_data = pitcher_data[
            (pitcher_data['Date'] >= start_datetime) & 
            (pitcher_data['Date'] <= end_datetime)
        ]
    
    return pitcher_data

# Function to create heatmaps for the selected pitcher, batter side, strikes, balls, and date filters
def plot_heatmaps(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
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
            
            if len(pitch_type_data) < 5:  # Switch to scatter plot for small data
                sns.scatterplot(
                    x=pitch_type_data['PlateLocSide'], 
                    y=pitch_type_data['PlateLocHeight'], 
                    ax=ax, 
                    color='blue'
                )
            else:
                bw_adjust_value = 0.5 if len(pitch_type_data) > 50 else 1  # Adjust bandwidth for small datasets
                sns.kdeplot(
                    x=pitch_type_data['PlateLocSide'], 
                    y=pitch_type_data['PlateLocHeight'], 
                    fill=True, 
                    cmap='Spectral_r', 
                    levels=6, 
                    ax=ax,
                    bw_adjust=bw_adjust_value
                )
            
            # Plot individual pitch locations as dots
            ax.scatter(
                pitch_type_data['PlateLocSide'], 
                pitch_type_data['PlateLocHeight'], 
                color='black',  # Color for the dots
                edgecolor='white',  # Add a white border to make dots stand out
                s=300,  # Size of the dots
                alpha=0.7  # Transparency to allow overlap
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
            ax.set_title(f"{pitch_type} ({pitcher_name})", fontsize=20)

            # Equal aspect ratio
            ax.set_aspect('equal', adjustable='box')
        
        # Remove any unused subplots
        for j in range(len(unique_pitch_types), len(axes)):
            fig.delaxes(axes[j])

        # Add a main title for all the heatmaps
        plt.suptitle(f"{pitcher_name} Heat Maps (Batter: {batter_side}, Strikes: {strikes}, Balls: {balls})", fontsize=30, fontweight='bold')
        
        # Adjust the layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for suptitle
        
        # Show the updated figure
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error generating heatmaps: {e}")

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
class_plus_file_path = "CLASS+ Trained on D1 Data - OM CLASS+ Ind Pitch.csv"  # Replace with the actual path
class_plus_df = pd.read_csv(class_plus_file_path)

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
class_plus_df["PitchType"] = class_plus_df["PitchType"].map(pitch_type_mapping)

# Function to generate the pitch traits table with CLASS+ scores
def generate_pitch_traits_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Group by 'TaggedPitchType' and calculate mean values for each group
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

        # Rename the columns
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

        # Merge with CLASS+ DataFrame
        class_plus_filtered = class_plus_df[class_plus_df["playerFullName"] == pitcher_name]
        grouped_data = pd.merge(
            grouped_data,
            class_plus_filtered[["PitchType", "CLASS+"]],  # Select only relevant columns
            how="left",
            left_on="Pitch",
            right_on="PitchType"
        )

        # Drop the extra 'PitchType' column from the merge and handle missing CLASS+ scores
        grouped_data = grouped_data.drop(columns=["PitchType"], errors="ignore")
        grouped_data["CLASS+"] = grouped_data["CLASS+"].fillna("N/A")

        # Sort by Count (most thrown to least thrown)
        grouped_data = grouped_data.sort_values(by='Count', ascending=False)

        # Format the data before displaying
        formatted_data = format_dataframe(grouped_data)

        # Display the table in Streamlit
        st.subheader("Pitch Traits:")
        st.dataframe(formatted_data)
    except Exception as e:
        st.write(f"Error generating pitch traits table: {e}")

# Function to generate the Plate Discipline table with "All" row
def generate_plate_discipline_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on the provided filters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Calculate total pitches
        total_pitches = len(pitcher_data)

        # Calculate InZone, Swing, Whiff, Chase, InZoneWhiff, and Strike percentages
        def calculate_metrics(df):
            in_zone_pitches = calculate_in_zone(df)
            total_in_zone = len(in_zone_pitches)

            # Define what constitutes a swing
            swing_conditions = ['StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable', 'InPlay']
            total_swings = df[df['PitchCall'].isin(swing_conditions)].shape[0]
            total_whiffs = df[df['PitchCall'] == 'StrikeSwinging'].shape[0]
            total_chase = df[
                (~df.index.isin(in_zone_pitches.index)) & 
                df['PitchCall'].isin(swing_conditions)
            ].shape[0]

            in_zone_whiffs = in_zone_pitches[in_zone_pitches['PitchCall'] == 'StrikeSwinging'].shape[0]

            # Define what constitutes a strike
            strike_conditions = ['StrikeCalled', 'FoulBallFieldable', 'FoulBallNotFieldable', 'StrikeSwinging', 'InPlay']
            total_strikes = df[df['PitchCall'].isin(strike_conditions)].shape[0]

            metrics = {
                'InZone%': (total_in_zone / len(df)) * 100 if len(df) > 0 else 0,
                'Swing%': (total_swings / len(df)) * 100 if len(df) > 0 else 0,
                'Whiff%': (total_whiffs / total_swings) * 100 if total_swings > 0 else 0,
                'Chase%': (total_chase / total_swings) * 100 if total_swings > 0 else 0,
                'InZoneWhiff%': (in_zone_whiffs / total_in_zone) * 100 if total_in_zone > 0 else 0,
                'Strike%': (total_strikes / len(df)) * 100 if len(df) > 0 else 0
            }
            return metrics

        # Group by 'TaggedPitchType' and calculate plate discipline metrics
        plate_discipline_data = pitcher_data.groupby('TaggedPitchType').apply(calculate_metrics).apply(pd.Series).reset_index()

        # Calculate the Pitch% column
        plate_discipline_data['Count'] = pitcher_data.groupby('TaggedPitchType')['TaggedPitchType'].count().values
        plate_discipline_data['Pitch%'] = (plate_discipline_data['Count'] / total_pitches) * 100

        # Reorder columns
        plate_discipline_data = plate_discipline_data[['TaggedPitchType', 'Count', 'Pitch%', 'Strike%', 'InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%']]

        # Rename columns
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

        # Calculate "All" row
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
            'Count': total_pitches,  # Total pitches
            'Pitch%': 100.0,  # Total percentage is 100%
            'Strike%': (total_strikes / total_pitches) * 100,
            'InZone%': (in_zone_pitches.shape[0] / total_pitches) * 100,
            'Swing%': (total_swings / total_pitches) * 100,
            'Whiff%': (total_whiffs / total_swings) * 100 if total_swings > 0 else 0,
            'Chase%': (total_chase / total_swings) * 100 if total_swings > 0 else 0,
            'InZoneWhiff%': (in_zone_whiffs / in_zone_pitches.shape[0]) * 100 if in_zone_pitches.shape[0] > 0 else 0
        }

        # Append "All" row to the DataFrame using pd.concat
        all_row_df = pd.DataFrame([all_row])
        plate_discipline_data = pd.concat([plate_discipline_data, all_row_df], ignore_index=True)

        # Format the data for display
        formatted_data = format_dataframe(plate_discipline_data)

        # Display the table in Streamlit
        st.subheader("Plate Discipline:")
        st.dataframe(formatted_data)
    except Exception as e:
        st.write(f"Error generating plate discipline table: {e}")



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

# Updated plot_pitch_movement function with color dictionary
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
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Set axis limits and labels
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_xlabel("Horizontal Break (inches)", fontsize=12)
        ax.set_ylabel("Induced Vertical Break (inches)", fontsize=12)
        
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
                s=50, 
                alpha=0.7,
                zorder=2  # Higher z-order to plot above the lines
            )

            # Calculate the mean and standard deviation for clustering
            mean_horz = pitch_type_data['HorzBreak'].mean()
            mean_vert = pitch_type_data['InducedVertBreak'].mean()
            std_dev = np.sqrt(pitch_type_data['HorzBreak'].std()**2 + pitch_type_data['InducedVertBreak'].std()**2)

            # Draw a circle to represent the cluster area
            circle = plt.Circle((mean_horz, mean_vert), std_dev, color=color, alpha=0.3, zorder=1)
            ax.add_patch(circle)

        # Add a legend for pitch types
        plt.legend(title="Pitch Type", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Display the plot in Streamlit
        st.subheader("Pitch Movement Graph:")
        st.pyplot(plt)
    except Exception as e:
        st.write(f"Error generating pitch movement graph: {e}")




# Function to generate the Batted Ball table
def generate_batted_ball_table(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date):
    try:
        # Filter data based on the provided filters
        pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls, date_filter_option, selected_date, start_date, end_date)

        if pitcher_data.empty:
            st.write("No data available for the selected parameters.")
            return

        # Filter rows where PitchCall is 'InPlay' to calculate BIP
        batted_data = pitcher_data[pitcher_data['PitchCall'] == 'InPlay']

        if batted_data.empty:
            st.write("No batted ball data available for the selected parameters.")
            return

        # Create a new column 'BattedType' based on the 'Angle' (launch angle)
        def categorize_batted_type(angle):
            if angle < 10:
                return "GroundBall"
            elif 10 <= angle < 25:
                return "LineDrive"
            elif 25 <= angle < 50:
                return "FlyBall"
            else:
                return "PopUp"

        batted_data['BattedType'] = batted_data['Angle'].apply(categorize_batted_type)

        # Group by pitch type and calculate metrics
        batted_ball_summary = batted_data.groupby('TaggedPitchType').agg(
            Count=('PitchCall', 'size'),  # Total count of pitches
            BIP=('PitchCall', 'size'),  # Count of balls in play
            GB=('BattedType', lambda x: (x == "GroundBall").sum()),  # Count of ground balls
            FB=('BattedType', lambda x: (x == "FlyBall").sum()),  # Count of fly balls
            EV=('ExitSpeed', 'mean'),  # Average exit velocity
            Hard=('ExitSpeed', lambda x: (x >= 95).sum()),  # Count of hard-hit balls
            Soft=('ExitSpeed', lambda x: (x < 95).sum())  # Count of soft-hit balls
        ).reset_index()

        # Add GB%, FB%, Hard%, Soft%, and Contact% columns
        batted_ball_summary['GB%'] = ((batted_ball_summary['GB'] / batted_ball_summary['BIP']) * 100).round(1).astype(str) + '%'
        batted_ball_summary['FB%'] = ((batted_ball_summary['FB'] / batted_ball_summary['BIP']) * 100).round(1).astype(str) + '%'
        batted_ball_summary['Hard%'] = ((batted_ball_summary['Hard'] / batted_ball_summary['BIP']) * 100).round(1).astype(str) + '%'
        batted_ball_summary['Soft%'] = ((batted_ball_summary['Soft'] / batted_ball_summary['BIP']) * 100).round(1).astype(str) + '%'

        # Calculate Contact% for each pitch type
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

        # Drop intermediate columns (GB, FB, Hard, Soft) after calculation
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

        # Calculate "All" row for totals and averages
        all_row = {
            'Pitch': 'All',
            'Count': pitcher_data.shape[0],  # Total count of pitches
            'BIP': batted_data.shape[0],  # Total BIP
            'EV': batted_data['ExitSpeed'].mean(),  # Overall average EV
            'GB%': f"{round((batted_data['BattedType'] == 'GroundBall').sum() / batted_data.shape[0] * 100, 1)}%",
            'FB%': f"{round((batted_data['BattedType'] == 'FlyBall').sum() / batted_data.shape[0] * 100, 1)}%",
            'Hard%': f"{round((batted_data['ExitSpeed'] >= 95).sum() / batted_data.shape[0] * 100, 1)}%",
            'Soft%': f"{round((batted_data['ExitSpeed'] < 95).sum() / batted_data.shape[0] * 100, 1)}%",
            'Contact%': f"{round(calculate_contact(pitcher_data), 1)}%"
        }

        # Append "All" row to the summary DataFrame using pd.concat()
        all_row_df = pd.DataFrame([all_row])
        batted_ball_summary = pd.concat([batted_ball_summary, all_row_df], ignore_index=True)

        # Format the data for display
        formatted_data = format_dataframe(batted_ball_summary)

        # Display the table in Streamlit
        st.subheader("Batted Ball:")
        st.dataframe(formatted_data)
    except Exception as e:
        st.write(f"Error generating batted ball table: {e}")





# Generate heatmaps based on selections
plot_heatmaps(
    pitcher_name, 
    batter_side, 
    strikes, 
    balls, 
    date_filter_option, 
    selected_date, 
    start_date, 
    end_date
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
