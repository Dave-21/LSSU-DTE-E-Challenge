#Written by: Gage Hoornstra

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
import winsound # Play sound after finishing code

# Debug variable
# If set to 1, will enable extra print statements and graph elements to aid in debugging
DEBUG = 0

from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "RJS_Results"
GRAPHS  = RESULTS / "graphs"
BUCKETS = {
    "00-30": RESULTS / "accuracy_00-30",
    "30-50": RESULTS / "accuracy_30-50",
    "50-60": RESULTS / "accuracy_50-60",
    "60-70": RESULTS / "accuracy_60-70",
    "70-80": RESULTS / "accuracy_70-80",
    "80+":   RESULTS / "accuracy_80+",
}
MODELS = ROOT / "models"

# make sure all output dirs exist
for p in [RESULTS, GRAPHS, MODELS, *BUCKETS.values()]:
    p.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(GRAPHS / f"Trained_Model{filename}_Accuracy_{accuracy_in_range:.2f}_MSE_{loss:.2f}.png")

    # Clear the plot
    plt.clf()

# THIS FUNCTION IS UNDER CONSTRUCTION
def sequence_plot(dataframe, y1, y2, x, loss):

    actual_values = dataframe[y1].tolist()
    predicted_values = dataframe[y2].tolist()
    
    # Group by date and find max usage for each day
    actual_daily_max = actual_values.groupby(actual_values[date_column].dt.date).max()
    predicted_daily_max = predicted_values.groupby(predicted_values[date_column].dt.date).max()

    # Align data by index (dates)
    actual_daily_max, predicted_daily_max = actual_daily_max.align(predicted_daily_max, join='inner')

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(actual_daily_max))

    # Bar chart for actual and predicted max usage
    ax.bar(index, actual_daily_max.values, bar_width, label='Actual Max Usage')
    ax.bar(index + bar_width, predicted_daily_max.values, bar_width, label='Predicted Max Usage')

    # Labeling
    ax.set_xlabel('Date')
    ax.set_ylabel('Max Energy Usage')
    ax.set_title('Daily Max Energy Usage: Actual vs. Predicted')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(actual_daily_max.index.strftime('%Y-%m-%d'), rotation=45)  # Format dates

    ax.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(GRAPHS / f"Sequence_Epoch_{epoch}_Accuracy_{accuracy_in_range:.2f}_MSE_{loss:.2f}.png")
    
    # Clear the plot
    plt.clf()


# Create an empty list to store the training loss values during each epoch
train_loss = []

# Load and preprocess the data
#filepath = 'data/Library_Data_May23-May24.csv'
filepath = 'data/Library_Data_June24.csv'
filename = os.path.basename(filepath)
df = pd.read_csv(filepath)
columns_to_delete = ['LSSU.Library.Heating.SNO-O.LSSU.Library.Heating.SNO-O.Trend - Present Value (Trend1)', 'Mass Flow.MASS FLOW Trend - Present Value (Trend1)',
                     'LSSU.Library.Heating.SNO-SP.LSSU.Library.Heating.SNO-SP.Trend - Present Value (Trend1)', 'Library - Energy In', 'Library - Energy Out', 'Library - Energy Consumed',
                     'Library - Condensate Temperature', 'Mass Flow.MASS FLOW Trend - Present Value (Trend1)', 'Total Neg Gals.HX2 Flow Total Negative Trend (Trend1)',
                     'Total Net Gals.HX2 FLow Total Net Trend (Trend1)', 'Total Pos Gals.HX2 Flow Total Positive Trend (Trend1)',
                     'Occupancy', 'LSSU.Library.Heating.HWDP-SP.LSSU.Library.Heating.HWDP-SP.Trend - Present Value (Trend1)', ' outHumi', 'HX2HWR-T.HX2HWR-T Trend - Present Value (Trend1)',
                     'HX2HWS-F.HX2 FLOW Trend - Present Value (Trend1)', 'HX2HWS-T.HX2HWS-T Trend - Present Value (Trend1)', 'LSSU.Library.Heating.OA-T.LSSU.Library.Heating.OA-T.Trend - Present Value (Trend1)',
                     'LSSU.Library.Heating.SNOMLT-T.LSSU.Library.Heating.SNOMLT-T.Trend - Present Value (Trend1)', ' gustspeed', ' dailygust', ' uvi', ' rainofhourly',
                     ' eventrain', ' rainofdaily', ' rainofweekly', ' rainofmonthly', ' rainofyearly', ' solarrad',
                     'Tomorrow Humidity - 1 Day Ago','Tomorrow Temperature - 1 Day Ago','Tomorrow Cloud Amount - 1 Day Ago','Tomorrow Precipitation - 1 Day Ago',
                     'Tomorrow Wind X - 1 Day Ago','Tomorrow Wind Y - 1 Day Ago','Library - Daily Max Energy','Library - Energy Consumed Hourly (Kilowatts)',
                     #'Hourly Flow.Hourly Flow Trend - Present Value (Trend1)','Density.Density Trend - Present Value (Trend1)','Temperature 1.Temperature Trend - Present Value (Trend1)','Pressure.Pressure Trend - Present Value (Trend1)','LIBCOND-T.LIBCOND-T Trend - Present Value (Trend1)', 
]
# Check if a column exists in the DataFrame before dropping it
for column in columns_to_delete:
    if column in df.columns:
        df = df.drop(column, axis=1)




# --Feature engineering--

# Convert 'Date / Time' column to datetime
df['Date / Time'] = pd.to_datetime(df['Date / Time'])

# --Removing shutdown time data (from after 12am - 6am)-- 
df = df[df['Date / Time'].dt.hour >= 5]
df.reset_index(drop=True, inplace=True)

# Getting hourly data alone for analyzing
df.to_csv("gen_data/hourtest.csv")

# --Converting date / time to a usable format--

# Seperating date time and converting to time stamp
timestamps = df.pop("Date / Time")
timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M:%S') 

# Adding hourly time
df['Hour'] = timestamps.dt.hour
if(DEBUG):
    print(df['Hour'])

# NEW TIME TRANSFORMATIONS
# Compute day of the year (ignoring hour, minute, second)
day_of_year = timestamps.dt.dayofyear

# Compute total seconds in the day
seconds_in_day = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second

# Map days of the year to sin and cos (NO hourly influence)
df['Year sin'] = np.sin(day_of_year * (2 * np.pi / 365.2425))
df['Year cos'] = np.cos(day_of_year * (2 * np.pi / 365.2425))

# Map time of day to sin/cos (captures within-day cycles)
df['Day sin'] = np.sin(seconds_in_day * (2 * np.pi / 86400))
df['Day cos'] = np.cos(seconds_in_day * (2 * np.pi / 86400))

# OLD TIME TRANSFORMATIONS
# Encodes the time stamps as unix time stamp (seconds since 1970)
# timestamps = timestamps.map(pd.Timestamp.timestamp)

# Setting up time variables
# day = 24*60*60
# year = 365.2425*day

# Map days of the year to sin and cos (WITH hourly influence)
# df['Year sin'] = np.sin(timestamps * (2 * np.pi / year))
# df['Year cos'] = np.cos(timestamps * (2 * np.pi / year))

# Maping days to sin and cos
# df['Day sin'] = np.sin(timestamps * (2 * np.pi / day))
# df['Day cos'] = np.cos(timestamps * (2 * np.pi / day))

# Moving energy consumed back to the end of the df
energy_consumed_col = df.pop('Library - Tomorrow Max Energy')
df['Library - Tomorrow Max Energy'] = energy_consumed_col

with open("data/columns_in_use.txt", 'w') as file:
    for column_name in df.columns:
        file.write(column_name + '\n')

if DEBUG:
    print('Input Data Size:')
    print(len(df.index),'Rows')
    print(len(df.columns)-1,'Columns\n')
   
#randomizing order of df
#df = df.sample(frac = 1)

# Splitting inputs and target data
X = df.drop('Library - Tomorrow Max Energy', axis=1)
y = df['Library - Tomorrow Max Energy']

# Saving columns to a list to be used later
X_columns = X.columns.tolist()


# Scaling Data
scaler = StandardScaler()

# Convert the data into sequences for the RNN

X_scaled = scaler.fit_transform(X)

# ---Data Windowing---

sequence_length = 10
X_sequences = []
y_sequences = []

# Defining a sequence of data that predicts into the future by sequence length hours
for i in range(len(X_scaled) - sequence_length + 1):
    X_sequences.append(X_scaled[i:i+sequence_length])
    y_sequences.append(y[i+sequence_length-1])
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Defining a sequence of data that predicts into the current hour with previous data of sequence length hours

# for i in range(sequence_length, len(X_scaled)):
#     X_sequence = X_scaled[i-sequence_length:i]  # Extract historical data
#     #X_sequence = X_scaled[i:i+sequence_length]  # Extract future data
#     y_value = y[i]  # Current target value
#     X_sequences.append(X_sequence)
#     y_sequences.append(y_value)

# X_sequences = np.array(X_sequences)
# y_sequences = np.array(y_sequences)

# Gettting a stretch of sequential data for later testing
X_sequential = X_sequences[120:120+72]
y_sequential = y_sequences[120:120+72]

# For getting non shuffled data
# def non_shuffling_train_test_split(X, y, test_size=0.2):
#     i = int((1 - test_size) * X.shape[0]) + 1
#     X_train, X_test = np.split(X, [i])
#     y_train, y_test = np.split(y, [i])
#     return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X_sequences, y_sequences, test_size=0.1)
# Split the data into training and testing sets, by default this randomizes the order of the data
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.1)


# Parameters all in one place to make tuning easier for the neural net
layer_one_size = 128
layer_two_size = 64
layer_three_size = 32
activation_method = 'relu'
learning_rate = 0.0010
epoch_amount = [200]
batch = 48
dropout_rate = 0.10
loops = 10

print('Training in progress, please wait...')
X.to_csv('gen_data/TrainingDataset.csv')
y.to_csv('gen_data/TrainingAnswers.csv')

# If you wish to test the same parameters multiple times then increase range to the desired quantity
for iter in range(loops):
    print('Run ',iter+1,' out of ',loops)
    # Testing every desired amount of epochs
    for epoch_index,epoch in enumerate(epoch_amount):
        # Saving a backup copy of the df to be reloaded after every run of the ai
        df_copy = df

        # Creating a AI model with three BiLSTM layers each seperated by a dropout layer.
        # BiLSTM layers excel at analyzing long term dependencies between inputs and
        # dropout layers help prevent overfitting by randomly input a 0 in for dropout_rate %
        # of inputs in the X_train dataset

        model = tf.keras.Sequential([
            
            # Input layer
            tf.keras.layers.Input(shape=(sequence_length, X_train.shape[2])),
    
            # First Bidirectional LSTM layer (change return_sequences to False when using dense layers after this layer)
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, activation=activation_method, return_sequences=False)),
            tf.keras.layers.Dropout(dropout_rate),

            # Flatten (Uncomment when using dense layers)
            tf.keras.layers.Flatten(),

            # Dense layers (Uncomment these and comment out the 2nd and 3rd BiLSTM Layers when using)
            tf.keras.layers.Dense(units=layer_one_size,activation=activation_method),
            tf.keras.layers.Dense(units=layer_two_size,activation=activation_method),
            tf.keras.layers.Dense(units=layer_three_size,activation=activation_method),
            
            # Second Bidirectional LSTM layer
            #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation=activation_method, return_sequences=True)),
            #tf.keras.layers.Dropout(dropout_rate),
            
            # Third Bidirectional LSTM layer
            #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation=activation_method, return_sequences=False)),
            #tf.keras.layers.Dropout(dropout_rate),
            
            # Final Dense layer for output
            tf.keras.layers.Dense(units=1, activation=activation_method),
        ])



        # Declare the optimizer with adjustable learning rate
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

        #Tensorboard Log Saving
        #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # patience = 2
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
        #                                                 patience=patience,
        #                                                 mode='min')
        # Train the model
        #model.fit(X_train, y_train, epochs=epoch_amount, batch_size=batch)
        
        # This code will make testing various epoch lengths faster by using previous weights file to resume from
        # The last epoch.
        # If it is anything but shortest amount of epochs then load the previous weights
        if epoch_index >= 1:
            model.load_weights('model_weights.h5')
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, initial_epoch=epoch_amount[epoch_index-1])
                    #,callbacks=[tensorboard_callback])     # Remove comment and previous ) to enable tensorboard logging
        # If it is the shortest epoch length then do not
        else:
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch)
                    #,callbacks=[tensorboard_callback])     # Remove comment and previous ) to enable tensorboard logging
        train_loss.extend(history.history['loss'])
        


        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print('Mean Squared Error:', loss)


        # Make predictions
        predictions = model.predict(X_test,verbose=None) # Change X_test to sequential to test sequential data

        # Reshape the arrays
        predictions = predictions.reshape(-1)
        y_test = y_test.reshape(-1)

        # Loading x test np array to df
        # X_test = X_test.reshape(-1)
        # X_test_df = pd.DataFrame(X_test, columns=X_columns)
        # print(X_test_df)

        # # Reversing the sine and cosine calculations for time
        # timestamps = np.arcsin(X_test['Day sin'])*(day / (2 * np.pi))
        # timestamps_cos = np.arccos(X_test['Day cos'])*(day / (2 * np.pi))
        # timestamps_year = np.arcsin(X_test['Year sin'])*(year / (2 * np.pi))
        # timestamps_year_cos = np.arccos(X_test['Year cos'])*(year / (2 * np.pi))
        
        
        # recovered_timestamps = [timestamps, timestamps_cos, timestamps_year, timestamps_year_cos]
        
        # print(recovered_timestamps[0])

        # Create a DataFrame with predictions and actual values
        df = pd.DataFrame({'prediction': predictions, 'actual': y_test}) # Change y_test to sequential to test sequential data

        # Calculate the difference between the actual and predicted values
        df['difference'] = df['prediction'].sub(df['actual']).abs()

        # Calculate the number of values within a deviation amount (percent)
        deviation_amount = 15
        values_in_range = len(df.loc[((df['difference'] / df['actual']) * 100) < deviation_amount, 'difference'])
        accuracy_in_range = (values_in_range / len(df)) * 100

        print("Accuracy in range of 15%" + " : " + str(accuracy_in_range) + "%")
        print(df.loc[((df['difference'] / df['actual']) * 100) < deviation_amount, :])
        
        # Plot the convergence graph
        plt.plot(range(1, epoch+1), train_loss[:epoch])
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss(MSE)')
        plt.title('Training Loss Convergence')
        plt.xlim(0, epoch)
        plt.ylim(0, 200) # Change back to 200
        #plt.show()
        plt.savefig(GRAPHS / f"Convergence_Epoch{epoch}_Accuracy_{accuracy_in_range:.2f}_MSE_{loss:.2f}.png")

        # Clear the plot
        plt.clf()
        
        # Plot difference graph
        difference_plot(df, 'actual', 'prediction', df.index, loss)

        # Plot sequence graph (THIS FUNCTION IS UNDER CONSTRUCTION)
        #sequence_plot(df, 'actual', 'prediction', df.index, loss)
        
        # Categorizing results based on accuracy
        def bucket_for(acc):
            if acc < 30:  return "00-30"
            if acc < 50:  return "30-50"
            if acc < 60:  return "50-60"
            if acc < 70:  return "60-70"
            if acc < 80:  return "70-80"
            return "80+"

        b = bucket_for(accuracy_in_range)
        out_csv = BUCKETS[b] / (
            f"BidirLSTM_accuracy_{accuracy_in_range:.2f}_mse_{loss:.2f}_"
            f"layers_{layer_one_size}_{layer_two_size}_{layer_three_size}_"
            f"am_{activation_method}_lr_{learning_rate}_epochs_{epoch_amount}_batchsz_{batch}_results.csv"
        )
        df.to_csv(out_csv, index=False)
        print(f"Wrote results => {out_csv}")

        df = df_copy
        model_out = MODELS / f"Library_Model_EnergyInputs_Dropout_SmallerFlat_{iter+1}.keras"
        model.save(model_out, overwrite=True)
        print(f"Saved model => {model_out}")
        #model.save('Library_Model_EnergyInputs_Dropout_SmallerFlat_' + str(iter+1) + '.keras',overwrite=True)
        #tf.keras.utils.plot_model(model,to_file='modelGraph_WindAvg_West_loop_' + str(iter+1) + '.png',show_shapes=True,show_layer_names=True)
        #print(model.layers[0].get_weights())

X.to_csv('gen_data/Model_Input_Data.csv')

if DEBUG:
    winsound.PlaySound('SystemAsterisk',winsound.SND_ALIAS) # Make computer beep when finished running

# Printing how many in each category
# print("Best Count: " + str(sum(len(files) for _, _, files in os.walk('results\\Best Settings Testing\\Best\\')) - best_count))
# print("Good Count: " + str(sum(len(files) for _, _, files in os.walk('results\\Best Settings Testing\\Good\\')) - good_count))
# print("Decent Count: " + str(sum(len(files) for _, _, files in os.walk('results\\Best Settings Testing\\Decent\\')) - decent_count))
# print("Bad Count: " + str(sum(len(files) for _, _, files in os.walk('results\\Best Settings Testing\\Bad\\')) - bad_count))
# print("Garbage Count: " + str(sum(len(files) for _, _, files in os.walk('results\\Best Settings Testing\\Garbage\\')) - garbage_count))
