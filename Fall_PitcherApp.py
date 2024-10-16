import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import math

# Load the real dataset
file_path = 'AllTrackman_fall_2024_df.csv'  # Replace with the correct path in your Streamlit setup
test_df = pd.read_csv(file_path)

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
    options=['Right', 'Left', 'Both']  # Added 'Both' option
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
# Function to filter data based on the dropdown selections
def filter_data(pitcher_name, batter_side, strikes, balls):
    # Filter data for the selected pitcher
    pitcher_data = test_df[test_df['Pitcher'] == pitcher_name]

    # Apply filtering for batter side, including 'Both' option
    if batter_side != 'Both':
        pitcher_data = pitcher_data[pitcher_data['BatterSide'] == batter_side]
    
    # Apply filtering for strikes if 'All' is not selected
    if strikes != 'All':
        pitcher_data = pitcher_data[pitcher_data['Strikes'] == strikes]
    
    # Apply filtering for balls if 'All' is not selected
    if balls != 'All':
        pitcher_data = pitcher_data[pitcher_data['Balls'] == balls]
    
    return pitcher_data


def plot_heatmaps(pitcher_name, batter_side, strikes, balls):
    # Filter data for the selected pitcher and batter side
    pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls)
    
    # Remove rows where PlateLocSide or PlateLocHeight is NaN, for plotting purposes only
    plot_data = pitcher_data.dropna(subset=['PlateLocSide', 'PlateLocHeight'])
    
    # Check if plot_data is empty after dropping NaN values
    if plot_data.empty:
        st.write("No data available to plot for the selected parameters.")
        return

    # Get unique pitch types thrown by the selected pitcher
    unique_pitch_types = plot_data['TaggedPitchType'].unique()

    if len(unique_pitch_types) == 0:
        st.write("No pitch types available to plot for the selected parameters.")
        return

    # Proceed with plotting
    n_pitch_types = len(unique_pitch_types)
    plots_per_row = 3  # Set number of plots per row
    n_rows = math.ceil(n_pitch_types / plots_per_row)  # Calculate the number of rows needed
    
    # Adjust figure size dynamically
    fig_width = 12 * plots_per_row  # Set width based on number of plots per row
    fig_height = 16 * n_rows  # Set height to fit all rows

    # Create subplots with the appropriate number of rows and columns
    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(fig_width, fig_height))
    axes = axes.flatten()  # Flatten axes array for easier access

    # Loop over each unique pitch type and create heatmaps
  for i, (ax, pitch_type) in enumerate(zip(axes, unique_pitch_types)):
    pitch_type_data = plot_data[plot_data['TaggedPitchType'] == pitch_type]
    
      if not pitch_type_data.empty and len(pitch_type_data) > 5:  # Ensure there are more than 5 data points
        # Plot heatmap using kdeplot (kernel density estimation)
        sns.kdeplot(
            x=pitch_type_data['PlateLocSide'], 
            y=pitch_type_data['PlateLocHeight'], 
            fill=True, 
            cmap='Spectral_r', 
            levels=6, 
            ax=ax,
            bw_adjust=0.5  # Adjust bandwidth for smoothness
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
      else:
        # If there's not enough data, write a message
        ax.text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', verticalalignment='center', fontsize=12, transform=ax.transAxes)

        
        # Add strike zone as a rectangle with black edgecolor
        strike_zone_width = 17 / 12  # 1.41667 feet
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
        ax.set_title(f"{pitch_type}", fontsize=40)

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a main title for all the heatmaps
    plt.suptitle(f"{pitcher_name} Heat Maps (Batter: {batter_side}, Strikes: {strikes}, Balls: {balls})", fontsize=60, fontweight='bold')
    
    # Adjust the layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for suptitle
    
    # Show the updated figure
    st.pyplot(fig)





# Generate heatmaps based on selections
plot_heatmaps(pitcher_name, batter_side, strikes, balls)

# Function to calculate InZone% and Chase%
def calculate_in_zone(df):
    # Strike zone boundaries
    in_zone = df[(df['PlateLocHeight'] >= 1.5) & (df['PlateLocHeight'] <= 3.3775) & 
                 (df['PlateLocSide'] >= -0.708) & (df['PlateLocSide'] <= 0.708)]
    return in_zone

# Function to manually format the dataframe before displaying
# Function to manually format the dataframe before displaying
# Function to manually format the dataframe before displaying
def format_dataframe(df):
    # Ensure columns are numeric and fill NaN with 'N/A'
    df = df.copy()  # Create a copy to avoid warnings
    percent_columns = ['InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%']
    
    for col in df.columns:
        if col in percent_columns:
            df[col] = df[col].apply(lambda x: f"{round(x, 2)}%" if pd.notna(x) and isinstance(x, (int, float)) else 'N/A')  # Add % symbol to percentage columns
        elif df[col].dtype.kind in 'f':  # if it's a float type column
            df[col] = df[col].apply(lambda x: round(x, 2) if pd.notna(x) else 'N/A')
        else:
            df[col] = df[col].fillna('N/A')  # Fill NaN with N/A for non-float columns
    return df


# Function to generate the pitch traits table
def generate_pitch_traits_table(pitcher_name, batter_side, strikes, balls):
    pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls)

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
        VertApprAngle=('VertApprAngle', 'mean'),
        Tilt=('Tilt', 'mean'),
        ExitSpeed=('ExitSpeed', lambda x: x.mean() if x.notna().sum() > 0 else 'N/A')
    ).reset_index()

    # Sort by Count (most thrown to least thrown)
    grouped_data = grouped_data.sort_values(by='Count', ascending=False)

    # Format the data before displaying
    formatted_data = format_dataframe(grouped_data)

    # Display the table in Streamlit
    st.subheader("Pitch Traits:")
    st.dataframe(formatted_data)

# Function to generate the plate discipline table
def generate_plate_discipline_table(pitcher_name, batter_side, strikes, balls):
    pitcher_data = filter_data(pitcher_name, batter_side, strikes, balls)

    # Calculate total pitches
    total_pitches = len(pitcher_data)

    # Calculate InZone, Swing, Whiff, Chase, and InZoneWhiff percentages
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

    # Group by 'TaggedPitchType' and calculate plate discipline metrics
    plate_discipline_data = pitcher_data.groupby('TaggedPitchType').apply(calculate_metrics).apply(pd.Series).reset_index()
    
    # Calculate the Pitch% column
    plate_discipline_data['Count'] = pitcher_data.groupby('TaggedPitchType')['TaggedPitchType'].count().values
    plate_discipline_data['Pitch%'] = (plate_discipline_data['Count'] / total_pitches) * 100

    # Sort by Count (most thrown to least thrown)
    plate_discipline_data = plate_discipline_data.sort_values(by='Count', ascending=False)

    # Reorder columns
    plate_discipline_data = plate_discipline_data[['TaggedPitchType', 'Count', 'Pitch%', 'InZone%', 'Swing%', 'Whiff%', 'Chase%', 'InZoneWhiff%']]

    # Format the data before displaying
    formatted_data = format_dataframe(plate_discipline_data)

    # Display the table in Streamlit
    st.subheader("Plate Discipline:")
    st.dataframe(formatted_data)

# Generate and display the pitch traits and plate discipline tables
generate_pitch_traits_table(pitcher_name, batter_side, strikes, balls)
generate_plate_discipline_table(pitcher_name, batter_side, strikes, balls)
