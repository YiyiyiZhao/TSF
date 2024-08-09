import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


keys={"mf": "Manufacturer-controllable Factors", "of":"Organization- or user-controllable Factors"}
method="ARIMA"

for kk in keys:
    key=keys[kk]
    df = pd.read_csv(f"data_no_outliers/data_{kk}.csv")
    df["unique_id"] = "1"
    df.columns = ["ds", "y", "unique_id"]
    df["ds"] = pd.to_datetime(df["ds"])
    Y_train_df = df[df.ds <= '2021-12-01']
    Y_test_df = df[df.ds > '2021-12-01']

    season_length = 12
    horizon = len(Y_test_df)
    models = [AutoARIMA(season_length=season_length)]
    sf = StatsForecast(df=Y_train_df, models=models, freq='MS', n_jobs=-1)
    sf.fit()


    Y_hat_df = sf.forecast(horizon, fitted=True)
    Y_hat_df = Y_hat_df.reset_index()
    Y_hat_df['unique_id'] = Y_hat_df['unique_id'].astype(int)
    Y_test_df['unique_id'] = Y_test_df['unique_id'].astype(int)
    Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])

    mae_score = mean_absolute_error(Y_test_df['y'], Y_hat_df['AutoARIMA'])
    rmse_score = mean_squared_error(Y_test_df['y'], Y_hat_df['AutoARIMA'],squared=False)
    print(f"Mean absolute error: {mae_score:.4f}")
    print(f"Root mean squared error: {rmse_score:.4f}")
    Y_hat_df.to_csv(f"prediction/{method}_{kk}.csv", index=False)



    sf_future = StatsForecast(df=df, models=models, freq='MS', n_jobs=-1)
    sf_future.fit()
    Y_future_df = sf_future.forecast(23, fitted=True)
    Y_future_df = Y_future_df.reset_index()

    Y_future_df['unique_id'] = Y_future_df['unique_id'].astype(int)

    Y_future_df.to_csv(f"forecast/{method}_{kk}.csv", index=False)
    Y_future_df.set_index('ds', inplace=True)
    Y_train_df.set_index('ds', inplace=True)
    Y_test_df.set_index('ds', inplace=True)
    Y_hat_df.set_index('ds', inplace=True)


    plt.figure(figsize=(12, 6))
    plt.plot(Y_train_df.index, Y_train_df['y'], label='Train', color='blue')
    plt.plot(Y_test_df.index, Y_test_df['y'], label='Test', color='green')
    plt.plot(Y_hat_df.index, Y_hat_df['AutoARIMA'], label='ARIMA Pred', color='orange')
    plt.plot(Y_future_df.index, Y_future_df['AutoARIMA'], label='ARIMA Future', color='maroon')
    vertical_line_date = Y_test_df.index[0]
    plt.axvline(x=vertical_line_date, color='gray', linestyle='--')
    vertical_line_date = Y_future_df.index[0]
    plt.axvline(x=vertical_line_date, color='gray', linestyle='--')
    plt.legend(loc='upper right',fontsize='large')
    plt.title(f'ARIMA for {key}: MAE = {mae_score:.4f}, RMSE = {rmse_score:.4f}.' , fontsize='x-large')
    #Train: 2009-01-01 to 2021-12-01; Test: 2022-01-01 to 2024-01-01. Predict Future: 2024-02-01 to 2025-12-01',
    plt.xlabel('Time', fontsize='large')
    plt.ylabel(f'{key}', fontsize='large')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f'./plots/{method}_{kk}.png', dpi=300)
