import matplotlib.pyplot as plt
import pandas as pd

keys={"mf": "Manufacturer-controllable Factors", "of":"Organization- or user-controllable Factors"}
for kk in keys:
    key=keys[kk]

    df=pd.read_csv(f"data_raw/data_{kk}.csv")
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df['y'], model='additive', period=12)

    # 检测残差中的异常值
    residuals = decomposition.resid
    outliers = (residuals > residuals.mean() + 3 * residuals.std()) | (residuals < residuals.mean() - 3 * residuals.std())
    residuals_mean = residuals.mean()
    residuals[outliers] = residuals_mean
    reconstructed_series = decomposition.trend + decomposition.seasonal + residuals

    df_recon = pd.DataFrame(index=df.index)
    df_recon['y'] = reconstructed_series

    # 确定要填充的行的索引
    first_n_rows = 6
    last_n_rows = 6

    # 获取 df_recon 的开始和结束的索引
    start_index = df_recon.index[:first_n_rows]
    end_index = df_recon.index[-last_n_rows:]

    # 从 df 中获取对应索引的值
    values_to_fill_start = df.loc[start_index, 'y']
    values_to_fill_end = df.loc[end_index, 'y']

    # 填充 df_recon 中的值
    df_recon.loc[start_index, 'y'] = values_to_fill_start
    df_recon.loc[end_index, 'y'] = values_to_fill_end


    # 可视化处理前后的对比
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['y'], label='Original', color='blue')
    plt.plot(df_recon.index, df_recon['y'], label='Processed', linestyle='--', color='red')
    plt.title(f"Outlier Processing for {key}",fontsize='x-large')  # 添加图表标题
    plt.xlabel("Time",fontsize='large')  # 添加 X 轴标签
    plt.ylabel(f"{key}",fontsize='large')  # 假设 y_label 是你想要设置的 Y 轴标签，替换成你的标签名
    plt.legend(fontsize='large')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f'./data_no_outliers/{kk}.png', dpi=300)

    df_recon.to_csv(f"data_no_outliers/data_{kk}.csv")