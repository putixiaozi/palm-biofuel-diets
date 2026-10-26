"""Extract FAOSTAT Food Balance Sheet (new methodology) items needed for oil and fat targets."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import paths
import pandas as pd
F = paths.REPO + 'data_raw/faostat/FoodBalanceSheets_E_All_Data_NOFLAG.csv'
OUT = paths.REPO + 'data_derived/fbs_oil_fat_2010_2023.csv'
oils = list(range(2571, 2583)) + [2586]
items = [2501, 2901, 2914, 2946] + oils
elems = [511, 664, 684, 645, 5142, 5154, 5511, 5611, 5911, 5072, 5301, 5521, 5123, 5131, 5170, 5171]
df = pd.read_csv(F, encoding='latin1')
df = df[(df['Area Code'] < 5000) & df['Item Code'].isin(items) & df['Element Code'].isin(elems)]
yc = [c for c in df.columns if c.startswith('Y')]
long = df.melt(id_vars=['Area Code', 'Area Code (M49)', 'Area', 'Item Code', 'Item', 'Element Code', 'Element', 'Unit'],
               value_vars=yc, var_name='year', value_name='value')
long['year'] = long.year.str[1:].astype(int)
long['m49'] = long['Area Code (M49)'].str.strip("'").astype(int)
long.dropna(subset=['value']).drop(columns=['Area Code (M49)']).to_csv(OUT, index=False)
print(long.shape, long.Area.nunique())
