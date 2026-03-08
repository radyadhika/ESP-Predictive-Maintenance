import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pickle

from sklearn.preprocessing import MaxAbsScaler

st.set_page_config(page_title='ESP Prediction',layout="wide", page_icon=':oil_drum:')

st.markdown('<p style="font-family: Roboto; font-size: 40px; text-align: center;"><b>Hierarchical Multivariate Time Series Classification Approach on ESP Predictive Maintenance</b></p>', unsafe_allow_html=True)

upload_well = st.file_uploader('Select well(s) sensor data: (.xlsx)', accept_multiple_files=True)

if upload_well:
    list_well = {}

    for i in upload_well:
        df_well = pd.read_excel(i)
        if len(np.unique(df_well.UniqueID)) > 1:
            for i in np.unique(df_well.UniqueID):
                list_well[i] = df_well[df_well['UniqueID'] == i]
        else:
            uniqueid = np.unique(df_well.UniqueID)[0]
            list_well[uniqueid] = df_well

    if len(list(list_well.keys())) > 1 :
        choose_well = st.selectbox('Choose Well: ', list(list_well.keys()))
        df_well = list_well[choose_well]
    else:
        df_well = list(list_well.values())[0]

    sensorcolumns = ['IntakePressure', 'DischargePressure', 'IntakeTemp', 'MotorTemp', 'VibrationX']

    for i in sensorcolumns:
        df_well[i + '_rolling_mean'] = df_well[i].rolling(window=5).mean()
        df_well[i + '_derivative'] = np.gradient(df_well[i], df_well['DateTillFail'])
    for i in [i + '_rolling_mean' for i in sensorcolumns]:
        df_well[i.split("_")[0] + '_direction'] = df_well[i].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    df_well = df_well.ffill().bfill()

    st.markdown(f'<p style="font-family: Roboto; font-size: 30px; text-align: center;"><b>{np.unique(df_well.UniqueID)[0]} Sensor</b></p>', unsafe_allow_html=True)

    graph_type = st.radio('Select sensor type: ', ["Original", "Derivative", "Rolling Mean", "Direction"],
                          captions = ["Original sensor value without preprocessing", "Sensor derivative with respect to time", "Average value of sensor within 5 periods", "Up/down/constant of sensor value compared to value before"], horizontal=True, label_visibility='collapsed')

    colors = px.colors.qualitative.Plotly  # List of colors from Plotly's qualitative color scale

    if graph_type == "Original":
        fig1 = px.line(x=df_well['DateTillFail'], y=df_well['IntakePressure'], labels={
            "x": "DTF", "y": "Intake Pressure"}, title='Intake Pressure Sensor', color_discrete_sequence=[colors[0]])

        fig2 = px.line(x=df_well['DateTillFail'], y=df_well['DischargePressure'], labels={
            "x": "DTF", "y": "Discharge Pressure"}, title='Discharge Pressure Sensor', color_discrete_sequence=[colors[1]])

        fig3 = px.line(x=df_well['DateTillFail'], y=df_well['IntakeTemp'], labels={
            "x": "DTF", "y": "Intake Temperature"}, title='Intake Temperature Sensor', color_discrete_sequence=[colors[2]])

        fig4 = px.line(x=df_well['DateTillFail'], y=df_well['MotorTemp'], labels={
            "x": "DTF", "y": "Motor Temperature"}, title='Motor Temperature Sensor', color_discrete_sequence=[colors[3]])

        fig5 = px.line(x=df_well['DateTillFail'], y=df_well['VibrationX'], labels={
            "x": "DTF", "y": "Vibration X"}, title='Vibration Sensor', color_discrete_sequence=[colors[4]])
    elif graph_type == "Derivative":
        fig1 = px.line(x=df_well['DateTillFail'], y=df_well['IntakePressure_derivative'], labels={
            "x": "DTF", "y": "Intake Pressure"}, title='Intake Pressure Sensor', color_discrete_sequence=[colors[0]])

        fig2 = px.line(x=df_well['DateTillFail'], y=df_well['DischargePressure_derivative'], labels={
            "x": "DTF", "y": "Discharge Pressure"}, title='Discharge Pressure Sensor', color_discrete_sequence=[colors[1]])

        fig3 = px.line(x=df_well['DateTillFail'], y=df_well['IntakeTemp_derivative'], labels={
            "x": "DTF", "y": "Intake Temperature"}, title='Intake Temperature Sensor', color_discrete_sequence=[colors[2]])

        fig4 = px.line(x=df_well['DateTillFail'], y=df_well['MotorTemp_derivative'], labels={
            "x": "DTF", "y": "Motor Temperature"}, title='Motor Temperature Sensor', color_discrete_sequence=[colors[3]])

        fig5 = px.line(x=df_well['DateTillFail'], y=df_well['VibrationX_derivative'], labels={
            "x": "DTF", "y": "Vibration X"}, title='Vibration Sensor', color_discrete_sequence=[colors[4]])
    elif graph_type == "Rolling Mean":
        fig1 = px.line(x=df_well['DateTillFail'], y=df_well['IntakePressure_rolling_mean'], labels={
            "x": "DTF", "y": "Intake Pressure"}, title='Intake Pressure Sensor', color_discrete_sequence=[colors[0]])

        fig2 = px.line(x=df_well['DateTillFail'], y=df_well['DischargePressure_rolling_mean'], labels={
            "x": "DTF", "y": "Discharge Pressure"}, title='Discharge Pressure Sensor', color_discrete_sequence=[colors[1]])

        fig3 = px.line(x=df_well['DateTillFail'], y=df_well['IntakeTemp_rolling_mean'], labels={
            "x": "DTF", "y": "Intake Temperature"}, title='Intake Temperature Sensor', color_discrete_sequence=[colors[2]])

        fig4 = px.line(x=df_well['DateTillFail'], y=df_well['MotorTemp_rolling_mean'], labels={
            "x": "DTF", "y": "Motor Temperature"}, title='Motor Temperature Sensor', color_discrete_sequence=[colors[3]])

        fig5 = px.line(x=df_well['DateTillFail'], y=df_well['VibrationX_rolling_mean'], labels={
            "x": "DTF", "y": "Vibration X"}, title='Vibration Sensor', color_discrete_sequence=[colors[4]])
    elif graph_type == "Direction":
        fig1 = px.line(x=df_well['DateTillFail'], y=df_well['IntakePressure_direction'], labels={
            "x": "DTF", "y": "Intake Pressure"}, title='Intake Pressure Sensor', color_discrete_sequence=[colors[0]])

        fig2 = px.line(x=df_well['DateTillFail'], y=df_well['DischargePressure_direction'], labels={
            "x": "DTF", "y": "Discharge Pressure"}, title='Discharge Pressure Sensor', color_discrete_sequence=[colors[1]])

        fig3 = px.line(x=df_well['DateTillFail'], y=df_well['IntakeTemp_direction'], labels={
            "x": "DTF", "y": "Intake Temperature"}, title='Intake Temperature Sensor', color_discrete_sequence=[colors[2]])

        fig4 = px.line(x=df_well['DateTillFail'], y=df_well['MotorTemp_direction'], labels={
            "x": "DTF", "y": "Motor Temperature"}, title='Motor Temperature Sensor', color_discrete_sequence=[colors[3]])

        fig5 = px.line(x=df_well['DateTillFail'], y=df_well['VibrationX_direction'], labels={
            "x": "DTF", "y": "Vibration X"}, title='Vibration Sensor', color_discrete_sequence=[colors[4]])

    fig1.update_xaxes(autorange="reversed")
    fig1.update_layout(title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}}, height=400)
    fig2.update_xaxes(autorange="reversed")
    fig2.update_layout(title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}}, height=400)
    fig3.update_xaxes(autorange="reversed")
    fig3.update_layout(title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}}, height=400)
    fig4.update_xaxes(autorange="reversed")
    fig4.update_layout(title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}}, height=400)
    fig5.update_xaxes(autorange="reversed")
    fig5.update_layout(title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}}, height=400)

    plot1, plot2, plot3 = st.columns(3)
    dummy_plot1, plot4, plot5, dummy_plot2 = st.columns((0.75, 1.5, 1.5, 0.75))

    plot1.plotly_chart(fig1, use_container_width=True)
    plot2.plotly_chart(fig2, use_container_width=True)
    plot3.plotly_chart(fig3, use_container_width=True)
    plot4.plotly_chart(fig4, use_container_width=True)
    plot5.plotly_chart(fig5, use_container_width=True)

    st.markdown(f'<p style="font-family: Roboto; font-size: 30px; text-align: center;"><b>ESP Failure Prediction</b></p>', unsafe_allow_html=True)

    for i in sensorcolumns:
        scaler_original = MaxAbsScaler()
        df_well[i] = scaler_original.fit_transform(np.array(df_well[i]).reshape(-1,1))
        scaler_rolling = MaxAbsScaler()
        df_well[i + '_rolling_mean'] = scaler_rolling.fit_transform(np.array(df_well[i + '_rolling_mean']).reshape(-1,1))
        scaler_derivative = MaxAbsScaler()
        df_well[i + '_derivative'] = scaler_derivative.fit_transform(np.array(df_well[i + '_derivative']).reshape(-1,1))

    with open('hmlc_model_2_split.pkl', 'rb') as f:
        hmlc_model = pickle.load(f)

    predictions = hmlc_model.predict(df_well.drop(['UniqueID'], axis=1))

    predictions[:, 2] = np.array([x.split('_')[1] for x in predictions[:, 2]])
    predictions[:, 3] = np.array([x.split('_')[2] for x in predictions[:, 3]])

    predictions = pd.DataFrame(predictions, columns=['GeneralFailureDescriptor', 'FailureItem', 'FailureItemSpecific','DetailedFailureDescriptor'])
    predictions['Combined'] = predictions['GeneralFailureDescriptor'] + '_' + predictions['FailureItem'] + '_' + predictions['FailureItemSpecific'] + '_' + predictions['DetailedFailureDescriptor']

    if len(np.unique(predictions['Combined'])) == 1:
        st.markdown(f'<p style="font-family: Roboto; font-size: 30px; text-align: center;"><b>Most Likely</b></p>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric('Probability', str(len(predictions[predictions['Combined'] == np.unique(predictions['Combined'])[0]])*100 / len(predictions)) + " %")
        col2.metric('Type', np.unique(predictions['Combined'])[0].split("_")[0])
        col3.metric('Item', np.unique(predictions['Combined'])[0].split("_")[1])
        col4.metric('Specific', np.unique(predictions['Combined'])[0].split("_")[2])
        col5.metric('Detailed', np.unique(predictions['Combined'])[0].split("_")[3])

    else:
        ml, ll = st.columns(2)

        ml.markdown(f'<p style="font-family: Roboto; font-size: 30px; text-align: center;"><b>Most Likely</b></p>', unsafe_allow_html=True)

        col1, col2 = ml.columns((1,2))
        col3, col4, col5 = ml.columns(3)

        col1.metric('Probability', str(round(len(predictions[predictions['Combined'] == predictions.groupby('Combined').size().idxmax()])*100 / len(predictions), 2)) + " %")
        col2.metric('Type', predictions.groupby('Combined').size().idxmax().split("_")[0])
        col3.metric('Item', predictions.groupby('Combined').size().idxmax().split("_")[1])
        col4.metric('Specific', predictions.groupby('Combined').size().idxmax().split("_")[2])
        col5.metric('Detailed', predictions.groupby('Combined').size().idxmax().split("_")[3])

        ll.markdown(f'<p style="font-family: Roboto; font-size: 30px; text-align: center;"><b>Least Likely</b></p>', unsafe_allow_html=True)

        fig = px.pie(names=np.unique(predictions[predictions['Combined'] != predictions.groupby('Combined').size().idxmax()].Combined), values=predictions[predictions['Combined'] != predictions.groupby('Combined').size().idxmax()].value_counts()*100 / len(predictions))

        fig.update_traces(textposition='inside',
                     text=(predictions[predictions['Combined'] != predictions.groupby('Combined').size().idxmax()].value_counts()*100 / len(predictions)).map("{:.2f}%".format),
                     textinfo='text', insidetextorientation='horizontal')

        fig.update_layout(font=dict(size=20), legend=dict(orientation="h",  yanchor="top",  y=-0.2, xanchor="center", x=0.5))

        ll.plotly_chart(fig,use_container_width = True)

st.markdown("""<style>.stRadio [role=radiogroup]{align-items: center;justify-content: center;}</style>""",unsafe_allow_html=True)
st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)
with st.sidebar:
    css_style = """
        <style>
            .sidebar-text {
                line-height: 1.5; /* Adjust the value as needed */
            }
        </style>
        """
    st.markdown(css_style, unsafe_allow_html=True)

    st.markdown('<p style="font-family: Roboto; font-size: 20px; text-align: left;"><b>General Information</b></p>', unsafe_allow_html=True)
    st.write("A simple streamlit app to predict information regarding ESP Failure. This application has the following feature:")
    st.markdown("""<ul class=\"sidebar-text\"><li>Accepts multiple well inputs, either in the same Excel file or different Excel file.</li>
        <li>Shows different types of graphs, including the original sensors, derivatives of the sensors, rolling means of the sensors, and directions of the sensors.</li>
        <li>All graphs and charts are interactive.</li>
        <li>Shows statistics of the most likely prediction and least likely predictions.</li>
        </ul>""", unsafe_allow_html=True)

    st.markdown('<p style="font-family: Roboto; font-size: 20px; text-align: left;"><b>Data Format</b></p>', unsafe_allow_html=True)
    st.write("The input data format should have 5 columns in order, which are named UniqueID, DateTillFail, IntakePressure, DischargePressure, IntakeTemp, MotorTemp, VibrationX. The explanation for each columns is as follows:")
    st.markdown("""<ul class="sidebar-text"><li>UniqueID : the unique identifier of ESP well.</li>
        <li>DateTillFail : the number of days until the ESP will fail, which serves as an indicator of time for the data.</li>
        <li>IntakePressure : sensor value for intake pressure (psi).</li>
        <li>DischargePressure : sensor value for discharge pressure (psi).</li>
        <li>IntakeTemp : sensor value for intake temperature (degF).</li>
        <li>MotorTemp : sensor value for motor temperature (degF).</li>
        <li>VibrationX : sensor value for vibration X (G).</li>
        </ul>""", unsafe_allow_html=True)

    data = np.random.randn(5, 6)
    columns = ["DateTillFail", "IntakePressure", "DischargePressure", "IntakeTemp", "MotorTemp", "VibrationX"]
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, 'UniqueID', 'WellA')

    st.dataframe(df, hide_index=True)
    st.markdown(
        """
        <div style="margin-top: -20px; font-size: 12px;">
            *The numbers in this DataFrame are random.
        </div>
        """,
        unsafe_allow_html=True
    )
