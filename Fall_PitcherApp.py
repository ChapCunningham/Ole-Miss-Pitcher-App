import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd

# Load the real dataset
file_path = 'AllTrackman_fall_2024_df.csv'  # Replace with the correct path in your Streamlit setup
test_df = pd.read_csv(file_path)

# Streamlit app layout
st.title("Pitcher Heat Maps")

# Dropdown widget to select the pitcher
pitcher_name = st.selectbox(
    "Select Pitcher:",
    options=test_df['Pitcher'].unique()
)

# Dropdown widget to select the batter side (Right or Left)
batter_side = st.selectbox(
    "Select Batter Side:",
    options=['Right', 'Left']
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

# Function to create heatmaps for the selected pitcher, batter side, strikes, and balls
def plot_heatmaps(pitcher_name, batter_side, strikes, balls):
    # Filter data for the selected pitcher and batter side
    pitcher_data = test_df[
        (test_df['Pitcher'] == pitcher_name) &
        (test_df['BatterSide'] == batter_side)
    ]
    
    # Apply filtering for strikes if 'All' is not selected
    if strikes != 'All':
        pitcher_data = pitcher_data[pitcher_data['Strikes'] == strikes]
    
    # Apply filtering for balls if 'All' is not selected
    if balls != 'All':
        pitcher_data = pitcher_data[pitcher_data['Balls'] == balls]
    
    # Get unique pitch types thrown by the selected pitcher
    unique_pitch_types = pitcher_data['TaggedPitchType'].unique()

    # Create new subplots side by side based on the number of unique pitch types
    n_pitch_types = len(unique_pitch_types)
    fig, axes = plt.subplots(1, n_pitch_types, figsize=(5 * n_pitch_types, 6))
    
    if n_pitch_types == 1:
        axes = [axes]  # Ensure axes is a list if there's only one subplot

    # Loop over each unique pitch type and create heatmaps
    for ax, pitch_type in zip(axes, unique_pitch_types):
        pitch_type_data = pitcher_data[pitcher_data['TaggedPitchType'] == pitch_type]
        
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
        ax.set_title(f"{pitch_type}", fontsize=12)

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    # Add a main title for all the heatmaps
    plt.suptitle(f"{pitcher_name} Heat Maps (Batter: {batter_side}, Strikes: {strikes}, Balls: {balls})", fontsize=16, fontweight='bold')
    
    # Adjust the layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for suptitle
    
    # Show the updated figure
    st.pyplot(fig)

# Generate heatmaps based on selections
plot_heatmaps(pitcher_name, batter_side, strikes, balls)
