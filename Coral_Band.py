"""
CORAL BREATH DETECTION SCRIPT w/ Elastic Band Data

WORKING VERSION

Purpose
	Gather sensor data in real time and perform predictions to actuate blowers

Outputs
	Real time actuation to blowers
	Raw data from sensors stored over period of classification
	Stored classification data

Change Log
	10.15 - Created initial script
	10.21 - Added ADS1015
	10.22 - Using "fake data" to verify that model can change predictions based on real time readings
    10.23 - Added min max scaling and second proximity sensor
    10.23 - Added derivatives of Force and Proximity sensors
    10.31 - Added data collection from wearable band instead of proximity and force sensors
    10.31 - Real time scaling and filtering added
            NOTE: need to retrain model
    11.3 - Added real time adjusting of scaling range (as peaks trend smaller over time)
    11.4 - Added class switch - still need to test
    11.4 - Moved to independent repo
    11.11 - Added class switch in real time
    11.12 - Improved min / max scaling

"""

# Import Libraries
import ADS1015
from tflite_runtime.interpreter import Interpreter
import time
import pandas as pd
import numpy as np

# Create helper functions
def movingAvg(Class, windowSize):
    """
    Create moving average to remove single outliers

    Args:
        Class (list): List of inhale / exhale classes
        Windowsize (int): Size of moving average window

    Returns:
        filteredClass (list): Filtered list of classes
    """
    
    filteredClass = []
    for i in range(0, len(Class)):

        if i < windowSize - 1:
            filteredClass.append(Class[i])

        elif Class[i] != Class[i - 1]:
            if sum(Class[i - (windowSize - 1):(i + 1)]) / windowSize > 1:
                filteredClass.append(2)

            elif sum(Class[i - (windowSize - 1):(i + 1)]) / windowSize < 1:
                filteredClass.append(0)

            else:
                filteredClass.append(Class[i])

        elif Class[i] == Class[i - 1]:
            filteredClass.append(Class[i])
            
    return filteredClass

# Create objects
adc = ADS1015.ADS1015()

# Create storage lists
Sensor_1_Data = []
Sensor_2_Data = []
Sensor_3_Data = []
Sensor_4_Data = []

Range_1 = []
Range_2 = []
Range_3 = []
Range_4 = []

d_Sensor_1_Data = []
d_Sensor_2_Data = []
d_Sensor_3_Data = []
d_Sensor_4_Data = []

RescaleRange = 200
RawData_1 = []
RawData_2 = []
RawData_3 = []
RawData_4 = []

Raw_Data = []
Predictions = []
FilteredPredictions = []

timeTracker = []

# Kalman Filter Parameters
mea_e = 0.05
est_e = 0.05
q = 0.05

d_mea_e = 0.07
d_est_e = 0.07
d_q = 0.08

# Initiatize timer and loop counter
startTime = time.time()
i = 0

# Instantiate TF Lite Model
interpreter = Interpreter(model_path = "model_5.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('Model Initiatized')

for i in range(0, 5000):

    # Read sensor values
    Sensor_1 = adc.read_adc(0, gain = 4)
    Sensor_2 = adc.read_adc(1, gain = 4)
    Sensor_3 = adc.read_adc(2, gain = 4)
    Sensor_4 = adc.read_adc(3, gain = 4)

    if i < 201:
        Range_1.append(Sensor_1)
        Range_2.append(Sensor_2)
        Range_3.append(Sensor_3)
        Range_4.append(Sensor_4)

        if i == 200:
                Min_1 = min(Range_1) - 600
                Min_2 = min(Range_2) - 600
                Min_3 = min(Range_3) - 600
                Min_4 = min(Range_4) - 600

                Max_1 = min(Range_1) + 1000
                Max_2 = min(Range_2) + 1000
                Max_3 = min(Range_3) + 1000
                Max_4 = min(Range_4) + 1000

    else:

        # Store raw data first
        RawData_1.append(Sensor_1)
        RawData_2.append(Sensor_2)
        RawData_3.append(Sensor_3)
        RawData_4.append(Sensor_4)

        if Sensor_1 > Max_1:
            Sensor_1 = 1
        elif Sensor_1 < Min_1:
            Sensor_1 = 0
        else:
            Sensor_1 = (Sensor_1 - Min_1) / (Max_1 - Min_1)

        if Sensor_2 > Max_2:
            Sensor_2 = 1
        elif Sensor_2 < Min_2:
            Sensor_2 = 0
        else:
            Sensor_2 = (Sensor_2 - Min_2) / (Max_2 - Min_2)

        if Sensor_3 > Max_3:
            Sensor_3 = 1
        elif Sensor_3 < Min_3:
            Sensor_3 = 0
        else:
            Sensor_3 = (Sensor_3 - Min_3) / (Max_3 - Min_3)

        if Sensor_4 > Max_4:
            Sensor_4 = 1
        elif Sensor_4 < Min_4:
            Sensor_4 = 0
        else:
            Sensor_4 = (Sensor_4 - Min_4) / (Max_4 - Min_4)

        # Kalman Filtering
        # For the first iteration of real data collection,
        # set the last estimate to the first value collected
        if i == 201:
            last_estimate_1 = Sensor_1
            last_estimate_2 = Sensor_2
            last_estimate_3 = Sensor_3
            last_estimate_4 = Sensor_4

            d_last_estimate_1 = 0
            d_last_estimate_2 = 0
            d_last_estimate_3 = 0
            d_last_estimate_4 = 0

            err_measure_1 = mea_e
            err_estimate_1 = est_e
            err_measure_2 = mea_e
            err_estimate_2 = est_e
            err_measure_3 = mea_e
            err_estimate_3 = est_e
            err_measure_4 = mea_e
            err_estimate_4 = est_e

            d_err_measure_1 = d_mea_e
            d_err_estimate_1 = d_est_e
            d_err_measure_2 = d_mea_e
            d_err_estimate_2 = d_est_e
            d_err_measure_3 = d_mea_e
            d_err_estimate_3 = d_est_e
            d_err_measure_4 = d_mea_e
            d_err_estimate_4 = d_est_e

            Sensor_1_Data.append(Sensor_1)
            Sensor_2_Data.append(Sensor_2)
            Sensor_3_Data.append(Sensor_3)
            Sensor_4_Data.append(Sensor_4)

            d_Sensor_1_Data.append(0)
            d_Sensor_2_Data.append(0)
            d_Sensor_3_Data.append(0)
            d_Sensor_4_Data.append(0)

            timeTracker.append(time.time())

        # Perform Kalman filter and update last_estimate with current_estimate value
        else:
            kalman_gain_1 = err_estimate_1 / (err_estimate_1 + err_measure_1)
            current_estimate_1 = last_estimate_1 + kalman_gain_1 * (Sensor_1 - last_estimate_1)
            err_estimate_1 =  (1.0 - kalman_gain_1) * err_estimate_1 + abs(last_estimate_1 - current_estimate_1) * q
            Sensor_1_Data.append(current_estimate_1)
            last_estimate_1 = current_estimate_1

            kalman_gain_2 = err_estimate_2 / (err_estimate_2 + err_measure_2)
            current_estimate_2 = last_estimate_2 + kalman_gain_2 * (Sensor_2 - last_estimate_2)
            err_estimate_2 =  (1.0 - kalman_gain_2) * err_estimate_2 + abs(last_estimate_2 - current_estimate_2) * q
            Sensor_2_Data.append(current_estimate_2)
            last_estimate_2 = current_estimate_2

            kalman_gain_3 = err_estimate_3 / (err_estimate_3 + err_measure_3)
            current_estimate_3 = last_estimate_3 + kalman_gain_3 * (Sensor_3 - last_estimate_3)
            err_estimate_3 =  (1.0 - kalman_gain_3) * err_estimate_3 + abs(last_estimate_3 - current_estimate_3) * q
            Sensor_3_Data.append(current_estimate_3)
            last_estimate_3 = current_estimate_3

            kalman_gain_4 = err_estimate_4 / (err_estimate_4 + err_measure_4)
            current_estimate_4 = last_estimate_4 + kalman_gain_4 * (Sensor_4 - last_estimate_4)
            err_estimate_4 =  (1.0 - kalman_gain_4) * err_estimate_4 + abs(last_estimate_4 - current_estimate_4) * q
            Sensor_4_Data.append(current_estimate_4)
            last_estimate_4 = current_estimate_4

            timeTracker.append(time.time())

            # Gather derivatives
            if len(Sensor_1_Data) > 1:
                d_Sensor_1 = (Sensor_1_Data[-1] - Sensor_1_Data[-2]) / (timeTracker[-1] - timeTracker[-2])
            else:
                d_Sensor_1 = 0

            if len(Sensor_2_Data) > 1:
                d_Sensor_2 = (Sensor_2_Data[-1] - Sensor_2_Data[-2]) / (timeTracker[-1] - timeTracker[-2])
            else:
                d_Sensor_2 = 0

            if len(Sensor_3_Data) > 1:
                d_Sensor_3 = (Sensor_3_Data[-1] - Sensor_3_Data[-2]) / (timeTracker[-1] - timeTracker[-2])
            else:
                d_Sensor_3 = 0

            if len(Sensor_4_Data) > 1:
                d_Sensor_4 = (Sensor_4_Data[-1] - Sensor_4_Data[-2]) / (timeTracker[-1] - timeTracker[-2])
            else:
                d_Sensor_4 = 0

            d_kalman_gain_1 = d_err_estimate_1 / (d_err_estimate_1 + d_err_measure_1)
            d_current_estimate_1 = d_last_estimate_1 + d_kalman_gain_1 * (d_Sensor_1 - d_last_estimate_1)
            d_err_estimate_1 =  (1.0 - d_kalman_gain_1) * d_err_estimate_1 + abs(d_last_estimate_1 - d_current_estimate_1) * d_q
            d_Sensor_1_Data.append(d_current_estimate_1)
            d_last_estimate_1 = d_current_estimate_1

            d_kalman_gain_2 = d_err_estimate_2 / (d_err_estimate_2 + d_err_measure_2)
            d_current_estimate_2 = d_last_estimate_2 + d_kalman_gain_2 * (d_Sensor_2 - d_last_estimate_2)
            d_err_estimate_2 =  (1.0 - d_kalman_gain_2) * d_err_estimate_2 + abs(d_last_estimate_2 - d_current_estimate_2) * d_q
            d_Sensor_2_Data.append(d_current_estimate_2)
            d_last_estimate_2 = d_current_estimate_2

            d_kalman_gain_3 = d_err_estimate_3 / (d_err_estimate_3 + d_err_measure_3)
            d_current_estimate_3 = d_last_estimate_3 + d_kalman_gain_3 * (d_Sensor_3 - d_last_estimate_3)
            d_err_estimate_3 =  (1.0 - d_kalman_gain_3) * d_err_estimate_3 + abs(d_last_estimate_3 - d_current_estimate_3) * d_q
            d_Sensor_3_Data.append(d_current_estimate_3)
            d_last_estimate_3 = d_current_estimate_3

            d_kalman_gain_4 = d_err_estimate_4 / (d_err_estimate_4 + d_err_measure_4)
            d_current_estimate_4 = d_last_estimate_4 + d_kalman_gain_4 * (d_Sensor_4 - d_last_estimate_4)
            d_err_estimate_4 =  (1.0 - d_kalman_gain_4) * d_err_estimate_4 + abs(d_last_estimate_4 - d_current_estimate_4) * d_q
            d_Sensor_4_Data.append(d_current_estimate_4)
            d_last_estimate_4 = d_current_estimate_4

            # If the min / max of the last 20 data points is far away from the current min / max, rescale
            if len(Sensor_1_Data) > RescaleRange:

                if 1 - max(Sensor_1_Data[-RescaleRange:-1]) > 0.2 or 1 - max(Sensor_1_Data[-RescaleRange:-1]) < 0.1: 
                    Max_1 = max(RawData_1[-RescaleRange:-1]) + 500

                if 1 - max(Sensor_2_Data[-RescaleRange:-1]) > 0.2 or 1 - max(Sensor_2_Data[-RescaleRange:-1]) < 0.1: 
                    Max_2 = max(RawData_2[-RescaleRange:-1]) + 500

                if 1 - max(Sensor_3_Data[-RescaleRange:-1]) > 0.2 or 1 - max(Sensor_3_Data[-RescaleRange:-1]) < 0.1: 
                    Max_3 = max(RawData_3[-RescaleRange:-1]) + 500

                if 1 - max(Sensor_4_Data[-RescaleRange:-1]) > 0.2 or 1 - max(Sensor_4_Data[-RescaleRange:-1]) < 0.1: 
                    Max_4 = max(RawData_4[-RescaleRange:-1]) + 500

                if min(Sensor_1_Data[-RescaleRange:-1]) < 0.15: 
                    Min_1 = min(RawData_1[-RescaleRange:-1]) - 500

                if min(Sensor_2_Data[-RescaleRange:-1]) < 0.15: 
                    Min_2 = min(RawData_2[-RescaleRange:-1]) - 500

                if min(Sensor_3_Data[-RescaleRange:-1]) < 0.15: 
                    Min_3 = min(RawData_3[-RescaleRange:-1]) - 500

                if min(Sensor_4_Data[-RescaleRange:-1]) < 0.15: 
                    Min_4 = min(RawData_4[-RescaleRange:-1]) - 500

            # Pull real time data into input tensor for classification
            data = np.float32([[current_estimate_1, current_estimate_2, current_estimate_3, current_estimate_4, d_current_estimate_1, d_current_estimate_2, d_current_estimate_3, d_current_estimate_4]])
            Raw_Data.append(data)
            input_data = data
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            if len(Predictions) < 50:
                Current_Class = np.argmax(interpreter.get_tensor(output_details[0]['index']))  
                Predictions.append(Current_Class)

            else:

                start = time.time()

                Current_Class = np.argmax(interpreter.get_tensor(output_details[0]['index']))  
                Predictions.append(Current_Class)

                if Current_Class == 1:
                    FilteredPredictions.append(1)

                elif Current_Class == 0:
                    FilteredPredictions.append(0)

                elif Current_Class == 2:
                    if len(FilteredPredictions) > 1:
                        FilteredPredictions.append(FilteredPredictions[-1])
                    else:
                        FilteredPredictions.append(2)

                FilteredPredictions = movingAvg(FilteredPredictions, 3)
                duration = round((time.time() - start) * 1000, 5)

                if FilteredPredictions[-1] == 1:
                    print('EXHALE')

                elif FilteredPredictions[-1] == 0:
                    print('INHALE')

                elif FilteredPredictions[-1] == 2:
                    print('REST')

                print(duration, 'ms')
                print('   ')
                print('~~~~~~~')
                print('   ')


