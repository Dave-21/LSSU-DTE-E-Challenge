import pandas as pd
import numpy as np
import datetime
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
tf.keras.utils.disable_interactive_logging() # Disable progress bar prints

DEBUG = 0

def difference_plot(dataframe, y1, y2, x, loss):
    
    # Assuming x is the index of the dataframe
    x = dataframe.index.tolist()
    actual_values = dataframe[y1].tolist()
    predicted_values = dataframe[y2].tolist()
    
    # Avoid division by zero by replacing zero actual values with a small number
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    actual_values[actual_values == 0] = 1e-6  # Small epsilon value to prevent division error

    # Calculate percentage error
    percentage_error = ((predicted_values - actual_values) / actual_values) * 100
    
    # Set up a custom colormap that centers around green for zero error
    colors = ['blue', 'green', 'red']  # Blue for under-prediction, Green for accurate, Red for over-prediction
    cmap = LinearSegmentedColormap.from_list('blue-green-red', colors)

    # Normalize with midpoint at 0% (perfect prediction)
    norm = TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)  # Adjusted to 50% under and over-prediction

    # Create a figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left Graph: Actual Values ----
    axes[0].plot(x, actual_values, color='green', marker='o', linestyle='None')
    axes[0].set_xlabel('Testing Hour Indices')
    axes[0].set_ylabel('Energy Consumption (kWh)')
    axes[0].set_title('Actual Energy Consumption')

    # ---- Right Graph: Predictions with Percentage Error ----
    scatter_pred = axes[1].scatter(x, predicted_values, c=percentage_error, cmap=cmap, norm=norm, marker='o')

    axes[1].set_xlabel('Testing Hour Indices')
    axes[1].set_ylabel('Energy Consumption (kWh)')
    axes[1].set_title('Prediction Error (%)')

    # Ensure both plots have the same y-axis scale
    y_min = min(min(actual_values), min(predicted_values))
    y_max = max(max(actual_values), max(predicted_values))
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    
    # Show the color gradient in a color bar with labels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], label="Prediction Error (%)")

    # Save the plot
    plt.savefig("C:\\DTE Archive Sept 2024\\Library AI\\results\\RJS_Model_Testing\\Model_Test" + str(filename) + "_Accuracy_" + str(accuracy_in_range) + "_MSE_" + str(loss) + ".png")

    # Clear the plot
    plt.clf()

# Load the saved model
model = tf.keras.models.load_model('Library_Model_May23-May24_82.keras')
print("Model loaded successfully!")

# Load new test data
filepath = 'Library_Data_June24-Jul24.csv' # Replace with actual file
filename = os.path.basename(filepath)
new_data = pd.read_csv(filepath)

# Drop unnecessary columns based on the training script
columns_to_delete = [
    'LSSU.Library.Heating.SNO-O.LSSU.Library.Heating.SNO-O.Trend - Present Value (Trend1)', 
    'Mass Flow.MASS FLOW Trend - Present Value (Trend1)',
    'LSSU.Library.Heating.SNO-SP.LSSU.Library.Heating.SNO-SP.Trend - Present Value (Trend1)', 
    'Library - Energy In', 'Library - Energy Out', 'Library - Energy Consumed',
    'Library - Condensate Temperature', 'Mass Flow.MASS FLOW Trend - Present Value (Trend1)', 
    'Total Neg Gals.HX2 Flow Total Negative Trend (Trend1)',
    'Total Net Gals.HX2 FLow Total Net Trend (Trend1)', 'Total Pos Gals.HX2 Flow Total Positive Trend (Trend1)',
    'Occupancy', 'LSSU.Library.Heating.HWDP-SP.LSSU.Library.Heating.HWDP-SP.Trend - Present Value (Trend1)', 
    ' outHumi', 'HX2HWR-T.HX2HWR-T Trend - Present Value (Trend1)',
    'HX2HWS-F.HX2 FLOW Trend - Present Value (Trend1)', 'HX2HWS-T.HX2HWS-T Trend - Present Value (Trend1)', 
    'LSSU.Library.Heating.OA-T.LSSU.Library.Heating.OA-T.Trend - Present Value (Trend1)',
    'LSSU.Library.Heating.SNOMLT-T.LSSU.Library.Heating.SNOMLT-T.Trend - Present Value (Trend1)', 
    ' gustspeed', ' dailygust', ' uvi', ' rainofhourly',
    ' eventrain', ' rainofdaily', ' rainofweekly', ' rainofmonthly', ' rainofyearly', ' solarrad',
    'Tomorrow Humidity - 1 Day Ago','Tomorrow Temperature - 1 Day Ago',
    'Tomorrow Cloud Amount - 1 Day Ago','Tomorrow Precipitation - 1 Day Ago',
    'Tomorrow Wind X - 1 Day Ago','Tomorrow Wind Y - 1 Day Ago','Library - Daily Max Energy',
    'Library - Energy Consumed Hourly (Kilowatts)'
]

for column in columns_to_delete:
    if column in new_data.columns:
        new_data = new_data.drop(column, axis=1)

# Convert 'Date / Time' column to datetime
new_data['Date / Time'] = pd.to_datetime(new_data['Date / Time'])

# Remove shutdown time data (keep data from 6 AM onward)
new_data = new_data[new_data['Date / Time'].dt.hour >= 5].reset_index(drop=True)

# Extract timestamps
timestamps = new_data.pop("Date / Time")
timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M:%S')

# Adding hourly time
new_data['Hour'] = timestamps.dt.hour

# NEW TIME TRANSFORMATIONS
# Compute day of the year (ignoring hour, minute, second)
day_of_year = timestamps.dt.dayofyear

# Compute total seconds in the day
seconds_in_day = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second

# Map days of the year to sin and cos (NO hourly influence)
new_data['Year sin'] = np.sin(day_of_year * (2 * np.pi / 365.2425))
new_data['Year cos'] = np.cos(day_of_year * (2 * np.pi / 365.2425))

# Map time of day to sin/cos (captures within-day cycles)
new_data['Day sin'] = np.sin(seconds_in_day * (2 * np.pi / 86400))
new_data['Day cos'] = np.cos(seconds_in_day * (2 * np.pi / 86400))

# OLD TIME TRANSFORMATIONS
# Setting up time variables
# day = 24*60*60
# year = 365.2425*day

# Encodes the time stamps as unix time stamp (seconds since 1970)
# timestamps = timestamps.map(pd.Timestamp.timestamp)

# Map days of the year to sin and cos (WITH hourly influence)
# new_data['Year sin'] = np.sin(timestamps * (2 * np.pi / year))
# new_data['Year cos'] = np.cos(timestamps * (2 * np.pi / year))

# Maping days to sin and cos
# new_data['Day sin'] = np.sin(timestamps * (2 * np.pi / day))
# new_data['Day cos'] = np.cos(timestamps * (2 * np.pi / day))

# Moving energy consumed back to the end of the df
energy_consumed_col = new_data.pop('Library - Tomorrow Max Energy')
new_data['Library - Tomorrow Max Energy'] = energy_consumed_col

# Separate features and target variable
X_new = new_data.drop('Library - Tomorrow Max Energy', axis=1)
y_new = new_data['Library - Tomorrow Max Energy']

# Load the scaler used in training
scaler = StandardScaler()
X_train = pd.read_csv('TrainingDataset.csv').drop(columns=['Unnamed: 0'])  # Load original training data for scaling
scaler.fit(X_train)  # Fit only on training data

# Scale new test data
X_new_scaled = scaler.transform(X_new)

if DEBUG:
    # Convert scaler's mean and std to a DataFrame for better comparison
    scaler_stats = pd.DataFrame({
        "Training Mean": scaler.mean_,
        "Training Std Dev": scaler.scale_,
    }, index=X_train.columns)  # Ensure columns match training data

    # Get test data mean/std and convert it to a DataFrame
    test_stats = pd.DataFrame({
        "Test Mean": np.mean(X_new, axis=0),
        "Test Std Dev": np.std(X_new, axis=0),
    }, index=X_new.columns)  # Ensure columns match test data

    # Merge both tables for side-by-side comparison
    comparison = scaler_stats.join(test_stats, how="outer")

    # Print to check differences
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Prevent line wrapping

    print(comparison)

# Create sequences for model input
sequence_length = 10
X_new_sequences = []
y_new_sequences = []

for i in range(len(X_new_scaled) - sequence_length + 1):
    X_new_sequences.append(X_new_scaled[i:i+sequence_length])
    y_new_sequences.append(y_new[i+sequence_length-1])

X_new_sequences = np.array(X_new_sequences)
y_new_sequences = np.array(y_new_sequences)

# Calculating testing MSE
print("Testing in progress, please wait...")
loss = model.evaluate(X_new_sequences, y_new_sequences)
print('Testing Mean Squared Error:', loss)
        
# Make predictions
predictions = model.predict(X_new_sequences)

# Reshape the arrays
predictions = predictions.reshape(-1)
y_new_sequences = y_new_sequences.reshape(-1)
                                
# Convert predictions to a DataFrame and save results
df = pd.DataFrame({
    'prediction': predictions,
    'actual': y_new_sequences
})

# Calculate the difference between the actual and predicted values
df['difference'] = df['prediction'].sub(df['actual']).abs()

# Calculate the number of values within a deviation amount (percent)
deviation_amount = 15
values_in_range = len(df.loc[((df['difference'] / df['actual']) * 100) < deviation_amount, 'difference'])
accuracy_in_range = (values_in_range / len(df)) * 100

print("Accuracy in range of 15%" + " : " + str(accuracy_in_range) + "%")
print(df.loc[((df['difference'] / df['actual']) * 100) < deviation_amount, :])

# Plot difference graph
difference_plot(df, 'actual', 'prediction', df.index, loss)

# Saving the results and Inputs to CSV files
X_new.to_csv("Tested_Model_Input_Data.csv", index=False)
df.to_csv("Predictions_Results.csv", index=False)
print("Predictions saved to Predictions_Results.csv!")
