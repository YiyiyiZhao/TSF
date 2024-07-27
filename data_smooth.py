import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import zscore


def smooth_df(df, keys, window_size=4, threshold=2):
    for key in keys:
        df[key + '_ZScore'] = zscore(df[key])
        rolling_mean = df[key].rolling(window=window_size, center=True).mean()
        df[key + '_Smoothed'] = df.apply(
            lambda row: rolling_mean[row.name] if abs(row[key + '_ZScore']) > threshold else row[key], axis=1)
        df.drop(key + '_ZScore', axis=1, inplace=True)
    return df


def plot_data(df, keys, out_dir):
    annotations = {'MF': 'Manufacturer-controllable factors',
                   'OF': 'Organization- or user-controllable factors',
                   'MNU': 'Manufacturer-influenced non-use',
                   'ONU': 'Organization- or user-controllable non-use'}

    df['Year'] = df['Year'].astype(str)

    for key in keys:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 6))

        # Plotting the original data
        axes[0].plot(df['Year'], df[key], label='Original Data', color='blue', alpha=0.7, linestyle='--', marker='*')
        axes[0].set_title(f'Original Data: {annotations[key]}')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel(f'{key} Values')
        axes[0].grid(True)
        tick_spacing = 4
        axes[0].set_xticks(df['Year'][::tick_spacing])
        axes[0].set_xlim([df['Year'].min(), df['Year'].max()])
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

        # Plotting the smoothed data
        axes[1].plot(df['Year'], df[key + '_Smoothed'], label='Smoothed Data', color='green', alpha=0.7, linestyle='--',marker='*')
        axes[1].set_title(f'Smoothed Data: {annotations[key]}')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel(f'{key} Values')
        axes[1].grid(True)
        tick_spacing = 4
        axes[1].set_xticks(df['Year'][::tick_spacing])
        axes[1].set_xlim([df['Year'].min(), df['Year'].max()])
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(f'./{out_dir}/{key}.jpg')
        plt.show()
    return None


# Load the data
file_path = 'data.csv'  # Replace with your file path
df = pd.read_csv(file_path)
for window_size in [2,3,4,5,6]:
    out_dir=f"./smoothed/ws{window_size}"
    os.makedirs(out_dir, exist_ok=True)
    df = smooth_df(df, ['MF', 'OF', 'MNU', 'ONU'],window_size)
    plot_data(df, ['MF', 'OF', 'MNU', 'ONU'],out_dir)
    df.to_csv(f"{out_dir}/data.csv", index=False)
