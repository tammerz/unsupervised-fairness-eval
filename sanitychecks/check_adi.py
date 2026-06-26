import pandas as pd

df = pd.read_csv('C:/Users/kaiso/PycharmProjects/BPAI/Data/la_county_prepared.csv')

print("Mean Tax Value by ADI State Rank:")
adi_stats = df.groupby('ADI_STATERNK')['taxvaluedollarcnt'].agg(['mean', 'median', 'count']).sort_index()
print(adi_stats)

print("\nCorrelation between ADI and Tax Value:")
print(df[['ADI_STATERNK', 'taxvaluedollarcnt']].corr())
