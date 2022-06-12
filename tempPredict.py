import pandas as pd
from neuralprophet import NeuralProphet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


df = pd.read_csv('temp - Copy.csv')
df['time'] = pd.to_datetime(df['time'])
df.drop_duplicates(inplace=True)
data = df[['time', 'temperature']] 
data.dropna(inplace=True)
data.columns = ['ds', 'y']
print(data.head())
print(df.head())
#train model
m = NeuralProphet(
    changepoints_range=0.95,
    n_changepoints=30,
    trend_reg=1,
    weekly_seasonality=False,
    daily_seasonality=10,
)
metrics = m.fit(data, freq='10min')
#forcast away
future = m.make_future_dataframe(data, periods=60//10*24*2, n_historic_predictions=True)
forecast = m.predict(future)
fig = m.plot(forecast)
# fig_comp = m.plot_components(forecast)
fig_param = m.plot_parameters()
