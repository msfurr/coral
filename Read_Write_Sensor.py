"""
Read Write Sensor

WORKING VERSION

DESCRIPTION
	Reads values from ADS1015 and prints to console
    Comment / Uncomment last section to write to CSV file for EDA

CHANGE LOG
    10.30 - Attempt at new scaling function
    10.30 - Real time kalman filters
    10.31 - Changing to absolute time
    10.31 - Real time kalman filters on derivative data
    11.3 - Adjusting min / max scaling depending on stored recent values
    11.12 - Improved min / max scaling

"""

# Import libraries
import time
import ADS1015
import pandas as pd
import statistics

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

RescaleRange = 370
RawData_1 = []
RawData_2 = []
RawData_3 = []
RawData_4 = []

PreFilter_1 = []
PreFilter_2 = []
PreFilter_3 = []
PreFilter_4 = []

PreScale_1 = []
PreScale_2 = []
PreScale_3 = []
PreScale_4 = []

timeTracker = []

# Kalman Filter Parameters
mea_e = 0.05
est_e = 0.05
q = 0.05

d_mea_e = 0.05
d_est_e = 0.05
d_q = 0.05

# Initiatize timer
startTime = time.time()

print('Reading Google Coral Data values, press Ctrl-C to quit...')

# Print column headers
print('|  1    |   2   |   3   |  4  |   Time   |   d_1   |   d_2   |   d_3   |   d_4   |    t   |'.format(*range(2)))
print('-' * 60)

for i in range(0, 10000):
    # Read all the ADC channel values in a list
    values = [0]*9

    # Read sensor values
    Sensor_1 = adc.read_adc(0, gain = 4)
    Sensor_2 = adc.read_adc(1, gain = 4)
    Sensor_3 = adc.read_adc(2, gain = 4)
    Sensor_4 = adc.read_adc(3, gain = 4)

    NoScale_1 = Sensor_1
    NoScale_2 = Sensor_2
    NoScale_3 = Sensor_3
    NoScale_4 = Sensor_4

    # Create a range for scaling based on first 10 values
    if i < 401:
        Range_1.append(Sensor_1)
        Range_2.append(Sensor_2)
        Range_3.append(Sensor_3)
        Range_4.append(Sensor_4)

        # At the end of data gathering set the min and max of the range
        if i == 400:
            Min_1 = min(Range_1) - 400
            Min_2 = min(Range_2) - 400
            Min_3 = min(Range_3) - 400
            Min_4 = min(Range_4) - 400

            Max_1 = max(Range_1) + 400
            Max_2 = max(Range_2) + 400
            Max_3 = max(Range_3) + 400
            Max_4 = max(Range_4) + 400

            NoScale_Min_1 = Min_1
            NoScale_Min_2 = Min_2
            NoScale_Min_3 = Min_3
            NoScale_Min_4 = Min_4

            NoScale_Max_1 = Max_1
            NoScale_Max_2 = Max_2
            NoScale_Max_3 = Max_3
            NoScale_Max_4 = Max_4

    # Begin data collection with scaling
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

        if NoScale_1 > NoScale_Max_1:
            NoScale_1 = 1
        elif NoScale_1 < NoScale_Min_1:
            NoScale_1 = 0
        else:
            NoScale_1 = (NoScale_1 - NoScale_Min_1) / (NoScale_Max_1 - NoScale_Min_1)

        if NoScale_2 > NoScale_Max_2:
            NoScale_2 = 1
        elif NoScale_2 < NoScale_Min_2:
            NoScale_2 = 0
        else:
            NoScale_2 = (NoScale_2 - NoScale_Min_2) / (NoScale_Max_2 - NoScale_Min_2)

        if NoScale_3 > NoScale_Max_3:
            NoScale_3 = 1
        elif NoScale_3 < NoScale_Min_3:
            NoScale_3 = 0
        else:
            NoScale_3 = (NoScale_3 - NoScale_Min_3) / (NoScale_Max_3 - NoScale_Min_3)

        if NoScale_4 > NoScale_Max_4:
            NoScale_4 = 1
        elif NoScale_4 < NoScale_Min_4:
            NoScale_4 = 0
        else:
            NoScale_4 = (NoScale_4 - NoScale_Min_4) / (NoScale_Max_4 - NoScale_Min_4)

        PreFilter_1.append(NoScale_1)
        PreFilter_2.append(NoScale_2)
        PreFilter_3.append(NoScale_3)
        PreFilter_4.append(NoScale_4)

        # Kalman Filtering
        # For the first iteration of real data collection,
        # set the last estimate to the first value collected
        if i == 401:
            last_estimate_1 = Sensor_1
            last_estimate_2 = Sensor_2
            last_estimate_3 = Sensor_3
            last_estimate_4 = Sensor_4

            d_last_estimate_1 = 0
            d_last_estimate_2 = 0
            d_last_estimate_3 = 0
            d_last_estimate_4 = 0

            last_estimate_1_NoScale = NoScale_1
            last_estimate_2_NoScale = NoScale_2
            last_estimate_3_NoScale = NoScale_3
            last_estimate_4_NoScale = NoScale_4

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

            err_measure_1_NoScale = mea_e
            err_estimate_1_NoScale = est_e
            err_measure_2_NoScale = mea_e
            err_estimate_2_NoScale = est_e
            err_measure_3_NoScale = mea_e
            err_estimate_3_NoScale = est_e
            err_measure_4_NoScale = mea_e
            err_estimate_4_NoScale = est_e

            Sensor_1_Data.append(Sensor_1)
            Sensor_2_Data.append(Sensor_2)
            Sensor_3_Data.append(Sensor_3)
            Sensor_4_Data.append(Sensor_4)

            d_Sensor_1_Data.append(0)
            d_Sensor_2_Data.append(0)
            d_Sensor_3_Data.append(0)
            d_Sensor_4_Data.append(0)

            PreScale_1.append(NoScale_1)
            PreScale_2.append(NoScale_2)
            PreScale_3.append(NoScale_3)
            PreScale_4.append(NoScale_4)

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

            k_gain_1 = err_estimate_1_NoScale / (err_estimate_1_NoScale + err_measure_1_NoScale)
            current_estimate_1_NoScale = last_estimate_1_NoScale + k_gain_1 * (NoScale_1 - last_estimate_1_NoScale)
            err_estimate_1_NoScale =  (1.0 - k_gain_1) * err_estimate_1_NoScale + abs(last_estimate_1_NoScale - current_estimate_1_NoScale) * q
            PreScale_1.append(current_estimate_1_NoScale)
            last_estimate_1_NoScale = current_estimate_1_NoScale

            k_gain_2 = err_estimate_2_NoScale / (err_estimate_2_NoScale + err_measure_2_NoScale)
            current_estimate_2_NoScale = last_estimate_2_NoScale + k_gain_2 * (NoScale_2 - last_estimate_2_NoScale)
            err_estimate_2_NoScale =  (1.0 - k_gain_2) * err_estimate_2_NoScale + abs(last_estimate_2_NoScale - current_estimate_2_NoScale) * q
            PreScale_2.append(current_estimate_2_NoScale)
            last_estimate_2_NoScale = current_estimate_2_NoScale

            k_gain_3 = err_estimate_3_NoScale / (err_estimate_3_NoScale + err_measure_3_NoScale)
            current_estimate_3_NoScale = last_estimate_3_NoScale + k_gain_3 * (NoScale_3 - last_estimate_3_NoScale)
            err_estimate_3_NoScale =  (1.0 - k_gain_3) * err_estimate_3_NoScale + abs(last_estimate_3_NoScale - current_estimate_3_NoScale) * q
            PreScale_3.append(current_estimate_3_NoScale)
            last_estimate_3_NoScale = current_estimate_3_NoScale

            k_gain_4 = err_estimate_4_NoScale / (err_estimate_4_NoScale + err_measure_4_NoScale)
            current_estimate_4_NoScale = last_estimate_4_NoScale + k_gain_4 * (NoScale_4 - last_estimate_4_NoScale)
            err_estimate_4_NoScale =  (1.0 - k_gain_4) * err_estimate_4_NoScale + abs(last_estimate_4_NoScale - current_estimate_4_NoScale) * q
            PreScale_4.append(current_estimate_4_NoScale)
            last_estimate_4_NoScale = current_estimate_4_NoScale

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

                if 1 - max(Sensor_1_Data[-RescaleRange:]) > 0.4 or 1 - max(Sensor_1_Data[-RescaleRange:-1]) < 0.1:
                    Max_1 = max(RawData_1[-RescaleRange:-1]) + 200
                    Min_1 = min(RawData_1[-RescaleRange:-1]) - 100

                if 1 - max(Sensor_2_Data[-RescaleRange:]) > 0.4 or 1 - max(Sensor_2_Data[-RescaleRange:-1]) < 0.1:
                    Max_2 = max(RawData_2[-RescaleRange:-1]) + 200
                    Min_2 = min(RawData_2[-RescaleRange:-1]) - 100

                if 1 - max(Sensor_3_Data[-RescaleRange:]) > 0.4 or 1 - max(Sensor_3_Data[-RescaleRange:-1]) < 0.1:
                    Max_3 = max(RawData_3[-RescaleRange:-1]) + 200
                    Min_3 = min(RawData_3[-RescaleRange:-1]) - 100

                if 1 - max(Sensor_4_Data[-RescaleRange:]) > 0.4 or 1 - max(Sensor_4_Data[-RescaleRange:-1]) < 0.1:
                    Max_4 = max(RawData_4[-RescaleRange:-1]) + 200
                    Min_4 = min(RawData_4[-RescaleRange:-1]) - 100
                

            # Gather values for displaying
            values[0] = Sensor_1
            values[1] = Sensor_2
            values[2] = Sensor_3
            values[3] = Sensor_4
            values[4] = d_Sensor_1
            values[5] = d_Sensor_2
            values[6] = d_Sensor_3
            values[7] = d_Sensor_4
            values[8] = timeTracker[-1]
            print('|', '%.4f'%values[0], ' |', '%.4f'%values[1], '|', '%.4f'%values[2], '|', '%.4f'%values[3], '|', '%.4f'%values[4], '|', '%.4f'%values[5], '|', '%.4f'%values[6], '|', '%.4f'%values[7], '|', '%.4f'%values[8], '|')

    # Pause for display
    time.sleep(0)

UnfilteredData = {'timeTracker': timeTracker, 'Sensor 1': PreFilter_1, 'Sensor 2': PreFilter_2, 'Sensor 3': PreFilter_3, 'Sensor 4': PreFilter_4}
UnscaledData = {'timeTracker': timeTracker, 'Sensor 1': PreScale_1, 'Sensor 2': PreScale_2, 'Sensor 3': PreScale_3, 'Sensor 4': PreScale_4}
SensorData = {'timeTracker': timeTracker, 'Sensor 1': Sensor_1_Data, 'Sensor 2': Sensor_2_Data, 'Sensor 3': Sensor_3_Data, 'Sensor 4': Sensor_4_Data, 'd_Sensor 1': d_Sensor_1_Data, 'd_Sensor 2': d_Sensor_2_Data, 'd_Sensor 3': d_Sensor_3_Data, 'd_Sensor 4': d_Sensor_4_Data}

Unfiltered = pd.DataFrame(data = UnfilteredData)
Unscaled = pd.DataFrame(data = UnscaledData)
Results = pd.DataFrame(data = SensorData)

export_csv = Unfiltered.to_csv(r'/home/mendel/coral/Unfiltered.csv', header = True, index = None)
export_csv = Unscaled.to_csv(r'/home/mendel/coral/Unscaled.csv', header = True, index = None)
export_csv = Results.to_csv(r'/home/mendel/coral/Results.csv', header = True, index = None)

# Pull command for file
# mdt pull /home/mendel/coral/Results.csv /Users/mikefurr/Documents/GitHub/coral_host/Jupyter_Notebooks
# mdt pull /home/mendel/coral/Unfiltered.csv /Users/mikefurr/Documents/GitHub/coral_host/Jupyter_Notebooks
# mdt pull /home/mendel/coral/Unscaled.csv /Users/mikefurr/Documents/GitHub/coral_host/Jupyter_Notebooks

# mdt pull /home/mendel/coral/Results.csv C:\Users\msfur\Documents\Raw_Data
# exec(open('Read_Write_Sensor.py').read())
# END

