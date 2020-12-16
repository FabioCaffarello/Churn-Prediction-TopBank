import warnings

import numpy  as np
import pandas as pd


warnings.filterwarnings("ignore")
np.random.seed(0)


dfRaw = pd.read_csv('../../01-Data/churn.csv')


dfRaw.head()





df01 = dfRaw.copy()


df01.columns


print(f'Number of Rows: {df01.shape[0]}')
print(f'Number of Columns: {df01.shape[1]}')


df01.dtypes


## Convert Objects type to categorical types >> more performatic
df01[df01.select_dtypes(include=['object']).columns] = df01.select_dtypes(include=['object']).astype('category')

## Convert Boolean integer columns to Boolean
df01[['HasCrCard', 'IsActiveMember', 'Exited']] = df01[['HasCrCard', 'IsActiveMember', 'Exited']].astype('bool')


df01.dtypes


df01.isnull().sum()


# Numerical Attributes
numAttributes = df01.select_dtypes(include=['int64', 'float64'])
NotNumerial = ['RowNumber', 'CustomerId']
numAttributes = numAttributes[numAttributes.columns[~numAttributes.columns.isin(NotNumerial)]]

#Categorical Attributes
catAttributes = df01.select_dtypes(include=['category'])

#Boolean Attributes
boolAttributes = df01.select_dtypes(include=['bool'])


### Central Tendency -> Mean, Median
ct1 = pd.DataFrame(numAttributes.apply(np.mean)).T
ct2 = pd.DataFrame(numAttributes.apply(np.median)).T

### Dispersion -> std, min, max, range, skew, kurtosis
d1 = pd.DataFrame(numAttributes.apply(np.std)).T
d2 = pd.DataFrame(numAttributes.apply(min)).T
d3 = pd.DataFrame(numAttributes.apply(max)).T
d4 = pd.DataFrame(numAttributes.apply(lambda x: x.max() - x.min())).T
d5 = pd.DataFrame(numAttributes.apply(lambda x: x.skew())).T
d6 = pd.DataFrame(numAttributes.apply(lambda x: x.kurtosis())).T

# Concatenate
m = pd.concat([d2, d3, d4, ct1, ct2, d1, d5, d6]).T.reset_index()
m.columns = ['Attributes', 'Min', 'Max', 'Range', 'Mean', 'Median', 'Std', 'Skew', 'Kurtosis']
m


def jupyter_settings():
    get_ipython().run_line_magic("matplotlib", " inline")
    get_ipython().run_line_magic("pylab", " inline")
    
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    
    display( HTML('<style>.container { width:100% get_ipython().getoutput("important; }</style>'))")
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option('display.expand_frame_repr', False)
    
    sns.set()


plt.rcParams['figure.figsize']


get_ipython().run_line_magic("matplotlib", " inline")


from IPython.display      import Image
from IPython.core.display import HTML 


jupyter_settings()


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


sns.displot(df01['CreditScore'], kde=False)
plt.show()


catAttributes.apply(lambda x: x.unique().shape[0])
