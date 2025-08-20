#Written by: Gage Hoornstra

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt


# Create an empty list to store the training loss values during each epoch
train_loss = []

# Load and preprocess the data
df = pd.read_csv('Library_Data.csv')
columns_to_delete = ['Density.Density Trend - Present Value (Trend1)', 'LSSU.Library.Heating.SNO-O.LSSU.Library.Heating.SNO-O.Trend - Present Value (Trend1)', 'Mass Flow.MASS FLOW Trend - Present Value (Trend1)',
                     'LSSU.Library.Heating.SNO-SP.LSSU.Library.Heating.SNO-SP.Trend - Present Value (Trend1)', 'Library - Energy In', 'Library - Energy Out', 'Library - Energy Consumed',
                     'Temperature 1.Temperature Trend - Present Value (Trend1)', 'Pressure.Pressure Trend - Present Value (Trend1)', 'Density.Density Trend - Present Value (Trend1)',
                     'Library - Condensate Temperature', 'Mass Flow.MASS FLOW Trend - Present Value (Trend1)', 'Total Neg Gals.HX2 Flow Total Negative Trend (Trend1)',
                     'Total Net Gals.HX2 FLow Total Net Trend (Trend1)', 'Total Pos Gals.HX2 Flow Total Positive Trend (Trend1)', 'LIBCOND-T.LIBCOND-T Trend - Present Value (Trend1)',
                     'Occupancy', 'LSSU.Library.Heating.HWDP-SP.LSSU.Library.Heating.HWDP-SP.Trend - Present Value (Trend1)', ' outHumi', 'HX2HWR-T.HX2HWR-T Trend - Present Value (Trend1)',
                     'HX2HWS-F.HX2 FLOW Trend - Present Value (Trend1)', 'HX2HWS-T.HX2HWS-T Trend - Present Value (Trend1)', 'LSSU.Library.Heating.OA-T.LSSU.Library.Heating.OA-T.Trend - Present Value (Trend1)',
                     'LSSU.Library.Heating.SNOMLT-T.LSSU.Library.Heating.SNOMLT-T.Trend - Present Value (Trend1)', ' gustspeed', ' dailygust', ' uvi', ' rainofhourly',
                     ' eventrain', ' rainofdaily', ' rainofweekly', ' rainofmonthly', ' rainofyearly', ' solarrad'
]
# Check if a column exists in the DataFrame before dropping it
for column in columns_to_delete:
    if column in df.columns:
        df = df.drop(column, axis=1)

# --Feature engineering--

# Convert wind direction and speed into wind vectors MIGHT WANT TO TAKE MAX DAILY VALUES AT SOME POINT
wind_speed_cols = [' HR 0 Wind Speed', ' HR 24 Wind Speed', ' HR 48 Wind Speed', ' HR 72 Wind Speed', ' HR 96 Wind Speed', ' HR 120 Wind Speed']
wind_dir_cols = [' HR 0 Wind Direction', ' HR 24 Wind Direction', ' HR 48 Wind Direction', ' HR 72 Wind Direction', ' HR 96 Wind Direction', ' HR 120 Wind Direction',]

for i in range(len(wind_speed_cols)):
    # Get wind speed
    wv = df.pop(wind_speed_cols[i])
    
    # Convert from mph to m/s
    wv = wv * 0.44704
    
    # Convert to radians
    wd_rad = df.pop(wind_dir_cols[i])*np.pi / 180

    # Making Column names
    wx_str = "HR " + str(i * 24) + " Wind X"
    wy_str = "HR " + str(i * 24) + " Wind Y"

    # Calculate the wind x and y components
    df[wx_str] = (wv*np.cos(wd_rad)).round(4)
    df[wy_str] = (wv*np.sin(wd_rad)).round(4)

# Removing some uneeded data

columns_to_delete = [' HR 0 Humidity', ' HR 0 Temperature', ' HR 0 Cloud Amount', ' HR 0 Precipitation', 'HR 0 Wind X', 'HR 0 Wind Y',
                     ' HR 72 Humidity', ' HR 72 Temperature', ' HR 72 Cloud Amount', ' HR 72 Precipitation', 'HR 72 Wind X', 'HR 72 Wind Y',
                     ' HR 48 Humidity', ' HR 48 Temperature', ' HR 48 Cloud Amount', ' HR 48 Precipitation', 'HR 48 Wind X', 'HR 48 Wind Y',
                     ' HR 96 Humidity', ' HR 96 Temperature', ' HR 96 Cloud Amount', ' HR 96 Precipitation', 'HR 96 Wind X', 'HR 96 Wind Y',
                     ' HR 120 Humidity', ' HR 120 Temperature', ' HR 120 Cloud Amount', ' HR 120 Precipitation', 'HR 120 Wind X', 'HR 120 Wind Y',
]

df = df.drop(columns_to_delete, axis=1)

# --Converting date / time to a usable format--

# Seperating date time and converting to time stamp
timestamps = df.pop("Date / Time")
timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M:%S')

# Adding hourly time
df['Hour'] = timestamps.dt.hour
print(df['Hour'])


timestamps = timestamps.map(pd.Timestamp.timestamp)

# Setting up time variables
day = 24*60*60
year = 365.2425*day

# Maping days and year to sin and cos LOOK INTO THIS MORE
df['Day sin'] = np.sin(timestamps * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamps * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamps * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamps * (2 * np.pi / year))

# Moving energy consumed back to the end of the df
energy_consumed_col = df.pop('Library - Energy Consumed Hourly (Kilowatts)')
df['Library - Energy Consumed Hourly (Kilowatts)'] = energy_consumed_col

with open("columns_in_use.txt", 'w') as file:
    for column_name in df.columns:
        file.write(column_name + '\n')

   
#randomizing order of df
#df = df.sample(frac = 1)

# Splitting inputs and target data
X = df.drop('Library - Energy Consumed Hourly (Kilowatts)', axis=1)
y = df['Library - Energy Consumed Hourly (Kilowatts)']

# Saving columns to a list to be used later
X_columns = X.columns.tolist()


# Scaling Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the data into sequences for the RNN
# Define the length of each sequence
# sequence_length = 10
# X_sequences = []
# y_sequences = []
# for i in range(len(X_scaled) - sequence_length + 1):
#     X_sequences.append(X_scaled[i:i+sequence_length])
#     y_sequences.append(y[i+sequence_length-1])
# X_sequences = np.array(X_sequences)
# y_sequences = np.array(y_sequences)
sequence_length = 10
X_sequences = []
y_sequences = []

for i in range(sequence_length, len(X_scaled)):
    X_sequence = X_scaled[i-sequence_length:i]  # Extract historical data
    y_value = y[i]  # Current target value
    X_sequences.append(X_sequence)
    y_sequences.append(y_value)

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Gettting a stretch of sequential data for later testing
X_sequential = X_sequences[120:120+72]
y_sequential = y_sequences[120:120+72]
# For getting inscrambled data
# def non_shuffling_train_test_split(X, y, test_size=0.2):
#     i = int((1 - test_size) * X.shape[0]) + 1
#     X_train, X_test = np.split(X, [i])
#     y_train, y_test = np.split(y, [i])
#     return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X_sequences, y_sequences, test_size=0.1)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.1)

# Parameters all in onc place to make tuning easier for the neural net
layer_sizes = [32, 64, 128]
activation_methods = ['relu']
learning_rates = [0.001]
epoch_amount = [10, 50]
batch = 24
dropout_rate = 0.20
    
#trying every combination of settings
for layer_one_size in layer_sizes:
    for layer_two_size in layer_sizes:
        for layer_three_size in layer_sizes:
            for activation_method in activation_methods:
                for learning_rate in learning_rates:
                    for epoch_index,epoch in enumerate(epoch_amount):
                        df_copy = df
                        # Making the RNN model
                        # model = tf.keras.Sequential([
                        #     tf.keras.layers.LSTM(layer_one_size, activation=activation_method, input_shape=(sequence_length, X_train.shape[2]), return_sequences=True),
                        #     tf.keras.layers.LSTM(layer_two_size, activation=activation_method),
                        #     tf.keras.layers.Dense(1)
                        # ])
                        model = tf.keras.Sequential([
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer_one_size, activation=activation_method, return_sequences=True), input_shape=(sequence_length, X_train.shape[2])),
                            tf.keras.layers.Dropout(dropout_rate),
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer_two_size, activation=activation_method, return_sequences=True)),
                            tf.keras.layers.Dropout(dropout_rate),
                            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(layer_three_size, activation=activation_method)),
                            tf.keras.layers.Dropout(dropout_rate),
                            tf.keras.layers.Dense(1)
                        ])



                        # Declare the optimizer with adjustable learning rate
                        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                        model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

                        # patience = 2
                        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                        #                                                 patience=patience,
                        #                                                 mode='min')
                        # Train the model
                        #model.fit(X_train, y_train, epochs=epoch_amount, batch_size=batch)
                        if epoch_index >= 1:
                            model.load_weights('model_weights.h5')
                            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch, initial_epoch=epoch_amount[epoch_index-1])
                        else:
                            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch)
                        train_loss.extend(history.history['loss'])
                        


                        # Evaluate the model
                        loss = model.evaluate(X_test, y_test)
                        print('Mean Squared Error:', loss)


                        # Make predictions
                        predictions = model.predict(X_test) # Change X_test to sequential to test sequential data

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
                        deviation_amount = 10
                        values_in_range = len(df.loc[((df['difference'] / df['actual']) * 100) < deviation_amount, 'difference'])
                        accuracy_in_range = (values_in_range / len(df)) * 100

                        print("Accuracy in range of 10%" + " : " + str(accuracy_in_range) + "%")
                        print(df.loc[((df['difference'] / df['actual']) * 100) < deviation_amount, :])
                        
                        # Plot the convergence graph
                        # plt.plot(range(1, epoch+1), train_loss[:epoch])
                        # plt.xlabel('Epoch')
                        # plt.ylabel('Training Loss(MSE)')
                        # plt.title('Training Loss Convergence')
                        # if epoch_index >= 1:
                        #     plt.ylim(0, 600)
                        # #plt.show()
                        # plt.savefig("results\\final report\\Conv and Predictions Seven\\Convergence_Epoch" + str(epoch) + "_Accuracy" + str(accuracy_in_range) + "_MSE" + str(loss) + ".png")
                        # # Clear the plot
                        # plt.clf()
                        
                        
                        # Plot difference graph
                        #plot(df, 'actual', 'prediction', df.index, loss)
                        
                        # Categorizing results based on accuracy
                        if(accuracy_in_range < 30):
                            df.to_csv("C:\\DTE Archive Sept 2024\\Library AI\\results\\SettingsList_1\\accuracy_00-30\\BidirLSTM_accuracy_" + str(accuracy_in_range) + "_mse_" + str(loss) + "_layers sizes_" + str(layer_one_size) + "_" + str(layer_two_size) + "_" + str(layer_three_size) + "_am_" + str(activation_method) + "_lr_" + str(learning_rate) + "_epochs_" + str(epoch_amount) + "_batchsz_" + str(batch) + "_" + "results.csv", index=False)
                        elif(accuracy_in_range >= 30 and accuracy_in_range < 50):
                            df.to_csv("C:\\DTE Archive Sept 2024\\Library AI\\results\\SettingsList_1\\accuracy_30-50\\BidirLSTM_accuracy_" + str(accuracy_in_range) + "_mse_" + str(loss) + "_layers sizes_" + str(layer_one_size) + "_" + str(layer_two_size) + "_" + str(layer_three_size) + "_am_" + str(activation_method) + "_lr_" + str(learning_rate) + "_epochs_" + str(epoch_amount) + "_batchsz_" + str(batch) + "_" + "results.csv", index=False)
                        elif(accuracy_in_range >= 50 and accuracy_in_range < 60):
                            df.to_csv("C:\\DTE Archive Sept 2024\\Library AI\\results\\SettingsList_1\\accuracy_50-60\\BidirLSTM_accuracy_" + str(accuracy_in_range) + "_mse_" + str(loss) + "_layers sizes_" + str(layer_one_size) + "_" + str(layer_two_size) + "_" + str(layer_three_size) + "_am_" + str(activation_method) + "_lr_" + str(learning_rate) + "_epochs_" + str(epoch_amount) + "_batchsz_" + str(batch) + "_" + "results.csv", index=False)
                        elif(accuracy_in_range >= 60 and accuracy_in_range < 70):
                            df.to_csv("C:\\DTE Archive Sept 2024\\Library AI\\results\\SettingsList_1\\accuracy_60-70\\BidirLSTM_accuracy_" + str(accuracy_in_range) + "_mse_" + str(loss) + "_layers sizes_" + str(layer_one_size) + "_" + str(layer_two_size) + "_" + str(layer_three_size) + "_am_" + str(activation_method) + "_lr_" + str(learning_rate) + "_epochs_" + str(epoch_amount) + "_batchsz_" + str(batch) + "_" + "results.csv", index=False)
                        elif(accuracy_in_range >= 70 and accuracy_in_range < 80):
                            df.to_csv("C:\\DTE Archive Sept 2024\\Library AI\\results\\SettingsList_1\\accuracy_70-80\\BidirLSTM_accuracy_" + str(accuracy_in_range) + "_mse_" + str(loss) + "_layers sizes_" + str(layer_one_size) + "_" + str(layer_two_size) + "_" + str(layer_three_size) + "_am_" + str(activation_method) + "_lr_" + str(learning_rate) + "_epochs_" + str(epoch_amount) + "_batchsz_" + str(batch) + "_" + "results.csv", index=False)
                        elif(accuracy_in_range >= 80):
                            df.to_csv("C:\\DTE Archive Sept 2024\\Library AI\\results\\SettingsList_1\\accuracy_70+\\BidirLSTM_accuracy_" + str(accuracy_in_range) + "_mse_" + str(loss) + "_layers sizes_" + str(layer_one_size) + "_" + str(layer_two_size) + "_" + str(layer_three_size) + "_am_" + str(activation_method) + "_lr_" + str(learning_rate) + "_epochs_" + str(epoch_amount) + "_batchsz_" + str(batch) + "_" + "results.csv", index=False)
                            
                        df = df_copy
                        model.save_weights('model_weights.h5')
