import pandas as pd

df = pd.read_csv('C:/Users/kaiso/PycharmProjects/BPAI/Data/la_county_prepared.csv')

print("Mean Logerror by ADI State Rank:")
adi_stats = df.groupby('ADI_STATERNK')['logerror'].agg(['mean', 'median', 'count']).sort_index()
print(adi_stats)

print("\nMean Logerror by Price Decile:")
df['price_decile'] = pd.qcut(df['taxvaluedollarcnt'], 10, labels=False) + 1
price_stats = df.groupby('price_decile')['logerror'].agg(['mean', 'median', 'count']).sort_index()
print(price_stats)
