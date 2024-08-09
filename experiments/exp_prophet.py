import pdb
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error



keys={"mf": "Manufacturer-controllable Factors", "of":"Organization- or user-controllable Factors"}
method="Prophet"
for kk in keys:
    key=keys[kk]
    df = pd.read_csv(f"data_no_outliers/data_{kk}.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    Y_train_df = df[df.ds <= '2021-12-01']
    Y_test_df = df[df.ds > '2021-12-01']

    #train
    m = Prophet()
    m.fit(Y_train_df)
    #test
    pred = m.make_future_dataframe(periods=25,freq='MS')
    predictions = m.predict(pred)
    Y_hat_df=predictions[-25:]
    Y_hat_df=Y_hat_df[["ds", "yhat"]]

    mae_score = mean_absolute_error(Y_test_df['y'], Y_hat_df['yhat'])
    rmse_score = mean_squared_error(Y_test_df['y'], Y_hat_df['yhat'],squared=False)

    print(f"Mean absolute error: {mae_score:.4f}")
    print(f"Root mean squared error: {rmse_score:.4f}")
    Y_hat_df.to_csv(f"prediction/{method}_{kk}.csv", index=False)

    #Future
    m = Prophet()
    m.fit(df)
    pred = m.make_future_dataframe(periods=23,freq='MS')
    predictions = m.predict(pred)
    Y_future_df =predictions[-23:]
    Y_future_df=Y_future_df[["ds", "yhat"]]

    Y_future_df.to_csv(f"forecast/{method}_{kk}.csv", index=False)
    Y_train_df.set_index("ds", inplace=True)
    Y_test_df.set_index("ds", inplace=True)
    Y_hat_df.set_index("ds", inplace=True)
    Y_future_df.set_index("ds", inplace=True)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(Y_train_df.index, Y_train_df['y'], label='Train', color='blue')
    plt.plot(Y_test_df.index, Y_test_df['y'], label='Test', color='green')
    plt.plot(Y_hat_df.index, Y_hat_df['yhat'], label='Prophet Pred', color='orange')
    plt.plot(Y_future_df.index, Y_future_df['yhat'], label='Prophet Future', color='maroon')


    vertical_line_date = Y_test_df.index[0]
    plt.axvline(x=vertical_line_date, color='gray', linestyle='--')


    vertical_line_date = Y_future_df.index[0]
    plt.axvline(x=vertical_line_date, color='gray', linestyle='--')

    plt.legend(loc='upper right',fontsize='large')
    plt.title(f'Prophet for {key}: MAE = {mae_score:.4f}, RMSE = {rmse_score:.4f}.' , fontsize='x-large')
    #Train: 2009-01-01 to 2021-12-01; Test: 2022-01-01 to 2024-01-01. Predict Future: 2024-02-01 to 2025-12-01',
    plt.xlabel('Time', fontsize='large')
    plt.ylabel(f'{key}', fontsize='large')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    plt.savefig(f'./plots/{method}_{kk}.png', dpi=300)