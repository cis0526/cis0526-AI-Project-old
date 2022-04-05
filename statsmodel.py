import pandas as pd
import statsmodels.api as sm

df = pd.DataFrame({
'height': [1.47, 1.50, 1.52],
'mass': [52.21, 53.12, 54.48],
})
print( df )
#    height   mass
# 0    1.47  52.21
# 1    1.50  53.12
# 2    1.52  54.48

df2 = sm.add_constant(df)
print( df2 )