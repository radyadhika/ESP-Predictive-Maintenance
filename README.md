# Streamlit Hierarchical Multivariate Time Series Classification for ESP Failure

## General Information
A simple [streamlit](https://streamlit.io) app to predict information regarding ESP Failure. This application has the following feature:  
a. Accepts multiple well inputs, either in the same Excel file or different Excel file.  
b. Shows different types of graphs, including the original sensors, derivatives of the sensors, rolling means of the sensors, and directions of the sensors.  
c. All graphs and charts are interactive.  
d. Shows statistics of the most likely prediction and least likely predictions.

### Data Format
The input data format should have 5 columns in order, which are named UniqueID, DateTillFail, IntakePressure, DischargePressure, IntakeTemp, MotorTemp, VibrationX. The explanation for each columns is as follows:
- UniqueID            : the unique identifier of ESP well.
- DateTillFail        : the number of days until the ESP will fail, which serves as an indicator of time for the data.
- IntakePressure      : sensor value for intake pressure (psi).
- DischargePressure   : sensor value for discharge pressure (psi).
- IntakeTemp          : sensor value for intake temperature (degF).
- MotorTemp           : sensor value for motor temperature (degF).
- VibrationX          : sensor value for vibration X (G).

## Live App
To view the application, simply go to this website [here](https://hmtsc-esp-failure-prediction.streamlit.app/). This application is hosted by [streamlit](https://streamlit.io).

## Installing the App Locally
### Install the required modules
```
pip install streamlit
pip install plotly
pip install numpy
pip install pandas
pip install scikit-learn
pip install scikit-learn-intelex
pip install openpyxl
pip install hiclass
```
### Running the App
```
streamlit run https://raw.githubusercontent.com/adamzainuri01/HMTSC_ESP_Prediction_Streamlit/HMTSC_Streamlit.py
```

## Required Module
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- scikit-learn-intelex
- openpyxl
- hiclass
