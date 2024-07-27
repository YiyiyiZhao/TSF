
import pandas as pd
import pdb
#read data
df = pd.read_excel('Data.xlsx')
df.rename(columns={
    'Manufacturer-controllable factors': 'MF',
    'Organization- or user-controllable factors': 'OF',
    'Manufacturer-influenced non-use': 'MNU',
    'Organization- or user-controllable non-use': 'ONU'},
    inplace=True)
#df.to_csv('v0.csv', index=False)

#merge data
df_up = df.groupby('Year').sum().reset_index()
df_up = df_up.drop(df_up.index[0:2])
df_up = df_up.reset_index(drop=True)
#df_up.to_csv('v1.csv', index=False)

#fill blank
new_rows = []
years=range(2009,2024)
months=range(1,13)
for y in years:
    for m in months:
        y_m = f"{y}{m:02d}"  # 使用格式化字符串确保月份为两位数
        if int(y_m) not in df_up['Year'].to_list():
            new_rows.append({'Year': int(y_m), 'MF': 0, 'OF': 0, 'MNU': 0, 'ONU': 0})

new_rows_df = pd.DataFrame(new_rows)
if not new_rows_df.empty:
    df = pd.concat([df_up, new_rows_df], ignore_index=True)
    df['Year'] = df['Year'].astype(int)
    df.sort_values(by='Year', inplace=True)
    df.to_csv('data.csv', index=False)