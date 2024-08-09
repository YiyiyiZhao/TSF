import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

gpt2_config = GPT2Config.from_pretrained('../llms/gpt2')
gpt2 = GPT2Model.from_pretrained('../llms/gpt2', config=gpt2_config)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('../llms/gpt2')
print("Model Loaded!")
prompt_prefix = "The dataset includes monthly event counts and may exhibit yearly seasonal patterns. The predicted number of events should be greater than zero."

method = "TimeLLM"
keys = {"mf": "Manufacturer-controllable Factors", "of": "Organization- or user-controllable Factors"}
for kk in keys:
    key = keys[kk]
    df = pd.read_csv(f"data_no_outliers/data_{kk}.csv")
    df["unique_id"] = "1"
    df.columns = ["ds", "y", "unique_id"]
    df["ds"] = pd.to_datetime(df["ds"])
    Y_train_df = df[df.ds <= '2021-12-01']
    Y_test_df = df[df.ds > '2021-12-01']

    horizon = len(Y_test_df)
    models = [TimeLLM(h=horizon, input_size=6, scaler_type='standard',batch_size=8,llm=gpt2,llm_config=gpt2_config, llm_tokenizer=gpt2_tokenizer, prompt_prefix=prompt_prefix,windows_batch_size=8, max_steps=300, val_check_steps=25,llm_output_hidden_states=False,top_k=3,patch_len=12)]
    print("Model Loaded!")
    nf = NeuralForecast(models=models, freq='MS')
    nf.fit(df=Y_train_df)

    Y_hat_df = nf.predict()
    Y_hat_df = Y_hat_df.reset_index()

    mae_score = mean_absolute_error(Y_test_df['y'], Y_hat_df[f'{method}'])
    rmse_score = mean_squared_error(Y_test_df['y'], Y_hat_df[f'{method}'], squared=False)
    print(f"Mean absolute error: {mae_score:.4f}")
    print(f"Root mean squared error: {rmse_score:.4f}")
    Y_hat_df.to_csv(f"prediction/{method}_{kk}.csv", index=False)
    # Future
    models = [TimeLLM(h=23, input_size=6, scaler_type='standard',batch_size=8, llm=gpt2, llm_config=gpt2_config, llm_tokenizer=gpt2_tokenizer,
             prompt_prefix=prompt_prefix, windows_batch_size=8, max_steps=300, val_check_steps=25,
             llm_output_hidden_states=False, top_k=3, patch_len=12)]
    print("Model Loaded!")
    nf = NeuralForecast(models=models, freq='MS')
    nf.fit(df=df)
    Y_future_df = nf.predict()
    Y_future_df.to_csv(f"forecast/{method}_{kk}.csv", index=False)

    Y_train_df.set_index("ds", inplace=True)
    Y_test_df.set_index("ds", inplace=True)
    Y_hat_df.set_index("ds", inplace=True)
    Y_future_df.set_index("ds", inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(Y_train_df.index, Y_train_df['y'], label='Train', color='blue')
    plt.plot(Y_test_df.index, Y_test_df['y'], label='Test', color='green')
    plt.plot(Y_hat_df.index, Y_hat_df[f'{method}'], label=f'{method} Pred', color='orange')
    plt.plot(Y_future_df.index, Y_future_df[f'{method}'], label=f'{method} Future', color='maroon')

    vertical_line_date = Y_test_df.index[0]
    plt.axvline(x=vertical_line_date, color='gray', linestyle='--')

    vertical_line_date = Y_future_df.index[0]
    plt.axvline(x=vertical_line_date, color='gray', linestyle='--')

    plt.legend(loc="upper right", fontsize='large')
    plt.title(f'{method} for {key}: MAE = {mae_score:.4f}, RMSE = {rmse_score:.4f}.', fontsize='x-large')
    # Train: 2009-01-01 to 2021-12-01; Test: 2022-01-01 to 2024-01-01. Predict Future: 2024-02-01 to 2025-12-01',
    plt.xlabel('Time', fontsize='large')
    plt.ylabel(f'{key}', fontsize='large')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./plots/{method}_{kk}.png', dpi=300)
