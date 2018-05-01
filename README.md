# PyPME
Assisting utility functions for calculation of Private Equity PME measures using Python. Natively compatible with Preqin data format.

*Example code:*
```python
for idx, fund in cf_funds.iterrows():
  fund_cf = cf_data[cf_data['Fund ID'] == fund['Fund ID']].copy()
  cf_funds.loc[idx,'local_KSPME'] = pypme.KS_PME(fund_cf["Transaction Date"],fund_cf["Transaction Amount"],fund_cf["Transaction Category"],local_index.iloc[:,0],local_index.iloc[:,1])
  cf_funds.loc[idx,'DirectAlpha'] = pypme.Direct_Alpha_PME(fund_cf["Transaction Date"],fund_cf["Transaction Amount"],fund_cf["Transaction Category"],local_index.iloc[:,0],local_index.iloc[:,1])
  cf_funds.loc[idx,'PME+'] = pypme.PME_PLUS(fund_cf["Transaction Date"],fund_cf["Transaction Amount"],fund_cf["Transaction Category"],local_index.iloc[:,0],local_index.iloc[:,1])                      
```
Where `cf_data` is the cashflow data imported from Preqin using `pd.read_csv('preqin_cf_file.csv')`, and `local_index` is a 2-column DataFrame in the format [Date|Index_Value].

The code above iterates over all unique funds in the file, and assigns the metric on a per-fund basis, using `local_index` as the basis for discount factor.

The library automatically finds the closest index dates for a given cash flow, and thus day-for-day index data is not required for the code to run properly (albeit, better resolution does obviously increase accuracy).
