"""
WORKING VERSION

Purpose:
    Performs classification on existing raw data to better understand current
    model
    
Outputs:
    List of class results produced by TFLite model inference from sample data
"""

import numpy as np
from tflite_runtime.interpreter import Interpreter
import time
import pandas as pd
    
#%%

def main():

    # Gather data from text file
    data = np.loadtxt('X_test.txt')
    data = np.float32([data])
    
    # Setup interpreter for inference
    interpreter = Interpreter(model_path = "model_4.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    start = time.time()
    
    results= []
    for i in range(0, len(data[0])):
        input_data = data[0][[i]]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        results.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
    
    duration = time.time() - start
    print(duration / len(data[0]))
    return results
    
results = main()
export_csv = pd.DataFrame(results).to_csv(r'/home/mendel/coral/coral_results.csv', header = True, index = None)

