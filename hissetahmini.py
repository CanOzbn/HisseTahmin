import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

hisse_senedi = 'DOAS.IS'

adese = yf.Ticker(hisse_senedi)
hist_df = adese.history(period="4y", interval="1d", auto_adjust=True)


hist_df.index = hist_df.index.tz_localize(None)

veri = hist_df.reset_index()[["Date", "Close"]]  
veri.columns = ['ds', 'y']

model = Prophet(changepoint_prior_scale=0.5, 
                seasonality_prior_scale=10.0,
                holidays_prior_scale=20.0,
                seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True)

model.fit(veri)

gelecek_tarihler = model.make_future_dataframe(periods=720)
tahminler = model.predict(gelecek_tarihler)

fig = model.plot(tahminler, figsize=(10, 5))

plt.fill_between(tahminler['ds'], tahminler['yhat_lower'], tahminler['yhat_upper'], color='gray', alpha=0.2)

plt.title(f'"{hisse_senedi}" - Gelecek Tahmin Fiyatı', y=0.94)

plt.xlabel('Date', loc='left')
plt.ylabel('Close Price', loc='top')

plt.legend()
plt.show()