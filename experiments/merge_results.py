import pandas as pd
import os



methods = ['ARIMA',  'Prophet', 'DLinear', 'NLinear','TCN', 'LSTM', 'Autoformer',  'PatchTST','TimeLLM']

kks = ['mf', 'of']

for directory_path in ['prediction', 'forecast']:
    if directory_path == 'prediction':
        ordered_columns = ['ds', 'y']
        ordered_columns.extend(methods)
        for kk in kks:
            columns_list = []
            ds_column = None
            y_column = None
            for file in os.listdir(directory_path):
                if file.endswith('.csv'):
                    method, f_kk = file.replace('.csv','').split('_')
                    if f_kk==kk:
                        df = pd.read_csv(os.path.join(directory_path, file))
                        columns_list.append(df[method])
                        if method == 'ARIMA':
                            ds_column = df['ds']
                            y_column = df['y']
            final_columns = pd.concat(columns_list, axis=1)
            final_df = pd.concat([ds_column, y_column, final_columns], axis=1)
            final_df = final_df[ordered_columns]
            final_df=final_df.round(5)
            final_df.to_csv(os.path.join("merged", f'prediction_{kk}.csv'), index=False)
    else:
        ordered_columns = ['ds']
        ordered_columns.extend(methods)
        for kk in kks:
            columns_list = []
            ds_column = None
            y_column = None
            for file in os.listdir(directory_path):
                if file.endswith('.csv'):
                    method, f_kk = file.replace('.csv','').split('_')
                    if f_kk==kk:
                        df = pd.read_csv(os.path.join(directory_path, file))
                        columns_list.append(df[method])
                        if method == 'ARIMA':
                            ds_column = df['ds']
            final_columns = pd.concat(columns_list, axis=1)
            final_df = pd.concat([ds_column, final_columns], axis=1)
            final_df = final_df[ordered_columns]
            final_df=final_df.round(5)
            final_df.to_csv(os.path.join("merged", f'forecast_{kk}.csv'), index=False)
