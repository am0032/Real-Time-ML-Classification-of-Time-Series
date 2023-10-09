import serial
import tkinter as tk
from tkinter import  Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.signal import argrelextrema
from sklearn.feature_selection import mutual_info_regression
import time
from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from tkinter import filedialog
import pandas as pd
import igraph as ig
from scipy.stats import entropy 
import os
import joblib


# Initialize variables for logging and calculation
logging = False
x = []

# Function to start and stop logging
def toggle_logging():
    global logging, x
    if logging:
        logging = False
        log_button.config(text="Start Logging")
    else:
        logging = True
        log_button.config(text="Stop Logging")
        x = []  # Reset the x array when starting logging again

# Function to read data from Arduino and update the plot
def update_plot():
    data = []
    while len(data) < 100:  # Adjust the number of data points you want to display
        try:
            line = ser.readline().decode().strip()
            value = int(line)
            # Normalize the value to be between 0 and 1
            normalized_value = value / 1023.0  # Assuming Arduino uses 10-bit ADC
            data.append(normalized_value)

            # Log the value if logging is enabled
            if logging:
                x.append(normalized_value)
        except ValueError:
            continue

    # Update the canvas with new data
    ax.clear()
    ax.plot(data, linestyle="-", marker="")
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Real-Time Analog Signal (Normalized)')
    ax.set_ylim(-0.1, 1.1)  # Set y-axis limits to -0.1 and 1.1
    canvas.draw()

    # Schedule the next update
    root.after(10, update_plot)  # Update every 100 milliseconds

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Create a list to store the file names
file_names = []

# Create an empty list to store DataFrames for each file
data_frames = []

# Load the machine learning model
model_path = filedialog.askopenfilename(
    title="Select pre-trained model file",
    filetypes=[("Joblib files", "*.joblib")]
)


# Function to calculate the average mutual information function
def calculate_avg_mutual_info():
    global x
    if len(x) > 0:

        x1 = x.copy()
        lenx1 = len(x1)
        print(lenx1)
        # Normalize the data components
        x1 = scaler.fit_transform(np.array(x1).reshape(-1, 1)).flatten()

        # Initialize an empty list to store the average mutual information values
        ami_list = []

        # Initialize an empty list to store the tau values
        tau_list = []
        
        # Loop over different values of tau (the delay parameter)
        def calculate_mutual_info(tau, x1):

            # Apply the delay embedding technique to create set B as x(ti+tau)
            xB = np.roll(x1, -tau)

            # Calculate the mutual information between set A (x(ti)) and set B (x(ti+tau))
            mi = mutual_info_regression(x1.reshape(-1, 1), xB.reshape(-1, 1))

            return tau, mi[0]

        start_time2 = time.time()  # Start timing

        # Create a list of tau values to loop over
        tau_values = list(range(1, 100))

        # Use Parallel and delayed to parallelize the mutual information calculation
        results = Parallel(n_jobs=-1)(delayed(calculate_mutual_info)(tau, x1) for tau in tau_values)

        # Extract the tau and mutual information values from the results
        tau_list, ami_list = zip(*results)

        # Convert the lists to numpy arrays
        ami_array = np.array(ami_list)
        tau_array = np.array(tau_list)

        # Find the indices of local minima
        local_min_indices = argrelextrema(ami_array, np.less)
        # Find the index of the first local minimum
        first_local_min_index = local_min_indices[0][0]
        # Print the value of tau and the average mutual information at the first local minimum
        tau_min = tau = tau_array[first_local_min_index]
        ami_min = ami_array[first_local_min_index]
        print(f"The first local minimum occurs at tau = {tau_min}")
        print(f"The average mutual information at the first local minimum is {ami_min}")

        x = x1

        # Compute recurrence plot threshold based on standard deviation of x
        std_x = np.std(x)
        eps = 0.4 * std_x

        # Define the parameters for FNN calculation
        dim_max = 15  # Maximum embedding dimension to consider
        Rtol = 15
        Atol = 2
        start_time3 = time.time()

        def calculate_fnn_dimension(d, x, tau, Rtol, Atol):
            def FNN(d, x, tau, Rtol, Atol):
                def reconstruct(x, dim, tau):
                    m = len(x) - (dim - 1) * tau
                    return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])

                def findneighbors(rec1):
                    n_neighbors = 2
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(rec1)
                    distance, inn = nbrs.kneighbors(rec1)
                    return inn[:, 1], distance[:, 1]

                rec1 = reconstruct(x[:-tau], d, tau)
                rec2 = reconstruct(x, d + 1, tau)
                inn, distance = findneighbors(rec1)

                R1 = np.abs(rec2[:, -1] - rec2[inn, -1]) / distance > Rtol
                R2 = np.linalg.norm(rec2 - rec2[inn], axis=1) / np.std(x) > Atol
                R3 = R1 | R2
                return np.mean(R1), np.mean(R2), np.mean(R3)

            fnn_R1, fnn_R2, fnn_R3 = FNN(d, x, tau, Rtol, Atol)
            return fnn_R3  # Return FNN Ratio (R3) for dimension d

        # Calculate FNN for different dimensions in parallel
        dim = np.arange(1, dim_max + 1)
        results = Parallel(n_jobs=-1)(delayed(calculate_fnn_dimension)(d, x, tau, Rtol, Atol) for d in dim)

        # Extract the FNN values for each dimension
        fnn_values_R3 = np.array(results)

    
        # Find the knee point or point of maximum change in slope
        from kneed import KneeLocator

        knee_locator = KneeLocator(dim, fnn_values_R3, curve='convex', direction='decreasing')
        knee_point = knee_locator.elbow

        # Highlight the knee point on the plot
        plt.scatter(knee_point, fnn_values_R3[knee_point], color='red', marker='o', label='Knee Point')
        plt.legend()

        # The knee point and corresponding dimension
        print("Knee Point Dimension:", dim[knee_point])
        m = dim[knee_point]
        end_time3 = time.time()  # End timing
        elapsed_time3 = end_time3 - start_time3  # Calculate elapsed time
        print(elapsed_time3)
        print("Time it took for FNN: " + str(elapsed_time3))

        # Create TimeSeries object
        time_series = TimeSeries(x, embedding_dimension=m, time_delay=tau)
        import pyrqa.settings as settings

        # Configure RQA settings
        rqa_settings = settings.Settings(
            time_series,
            analysis_type=Classic(),
            neighbourhood=FixedRadius(eps),
            similarity_measure=EuclideanMetric(),
            theiler_corrector=1
        )

        # Perform RQA computation
        computation = RQAComputation.create(rqa_settings, verbose=True)
        result = computation.run()

        computation = RPComputation.create(rqa_settings)
        result2 = computation.run()
    
        
        filename = 'recurrence_plot.png'  # Include param_type in the filename
        full_file_path = os.path.join(model_path, filename)
        ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse, full_file_path)

        # result is the output of RecurrencePlotComputation
        L = result2.recurrence_matrix_reverse[::-1]

        # Create an igraph Graph from the adjacency matrix
        G = ig.Graph.Adjacency(L, mode='undirected')
        # Remove self-loops from the graph
        G.simplify()

        print("Number of nodes:", G.vcount())
        print("Number of edges:", G.ecount())

        # Calculate clustering coefficient
        clustering_coefficient = G.transitivity_undirected()

        avg_path_length = G.average_path_length()

        # Calculate diagonal line lengths
        diagonal_line_lengths = np.diff(np.where(L))

        # Calculate the probabilities of each diagonal line length
        unique_lengths, length_counts = np.unique(diagonal_line_lengths, return_counts=True)
        probabilities = length_counts / np.sum(length_counts)

        # Calculate Shannon entropy of diagonal line lengths
        diagonal_line_entropy = entropy(probabilities, base=2)

        # After processing each file, create a DataFrame for the measure values
        measure_df = pd.DataFrame({

            'Recurrence Rate': [result.recurrence_rate],
            'Determinism': [result.determinism],
            'Laminarity': [result.laminarity],
            'average path length': [avg_path_length],
            'Clustering Coefficient': [clustering_coefficient],
            'Diagonal Line Entropy': [diagonal_line_entropy]
        })

        # Append the measure DataFrame to the list
        data_frames.append(measure_df)

        # After processing all files, concatenate the DataFrames in the list
        RQA_NETWORK_data = pd.concat(data_frames, ignore_index=False)

        
        # Use the RQA_NETWORK_data DataFrame for predictions
        X = RQA_NETWORK_data

        loaded_model = joblib.load(model_path)

        # Make predictions
        predictions = loaded_model.predict(X)

        # Print the predicted class
        print("Predicted Class:", predictions[0])
        # Display the predicted class in a label
        max_label.config(text=f"Predicted Class: {predictions[0]}")
        end_time2 = time.time()  # End timing
        elapsed_time2 = end_time2 - start_time2  # Calculate elapsed time

        print("Time it took to make prediction: " + str(elapsed_time2))

# Create a Tkinter window
root = tk.Tk()
root.title("Timseries Oscilloscope and predictor")



# Create a Matplotlib figure and canvas
fig, ax = plt.subplots(figsize=(6, 10))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
baud_rate=9600
# Try connecting to COM ports 1-6
ser = None
for port_num in range(0, 10):
    try:
        port_name = f"COM{port_num}"
        ser = serial.Serial(port_name, baud_rate)
        break  # Exit the loop if successful
    except serial.SerialException:
        pass

if ser is None:
    print("Failed to connect to any COM port (COM1 to COM6). Please check your Arduino connection.")
    root.destroy()
else:
    print(f"Connected to {ser.name}")

# Create buttons
log_button = Button(root, text="Start Logging", command=toggle_logging)
log_button.pack()
max_button = Button(root, text="Predict Behaviour", command=calculate_avg_mutual_info)
max_button.pack()
max_label = tk.Label(root, text="")
max_label.pack()



# Start the initial update of the plot
update_plot()

# Start the Tkinter main loop
root.mainloop()
