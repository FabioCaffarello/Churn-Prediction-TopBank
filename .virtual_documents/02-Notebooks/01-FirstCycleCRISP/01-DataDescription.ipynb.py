import warnings

import numpy  as np
import pandas as pd

import seaborn           as sns
import matplotlib.pyplot as plt


from IPython.display      import Image
from IPython.core.display import HTML


warnings.filterwarnings("ignore")
np.random.seed(0)


dfRaw = pd.read_csv('../../01-Data/churn.csv')


dfRaw.head()


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


jupyter_settings()


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


### Central Tendency -> Mean, Median, Mode
ct1 = pd.DataFrame(numAttributes.apply(np.mean)).T
ct2 = pd.DataFrame(numAttributes.apply(np.median)).T
ct3 = pd.DataFrame(stats.mode(numAttributes)[0])
ct3.columns = ct1.columns

### Dispersion -> std, min, max, range, skew, kurtosis, rsd
d1 = pd.DataFrame(numAttributes.apply(np.std)).T
d2 = pd.DataFrame(numAttributes.apply(min)).T
d3 = pd.DataFrame(numAttributes.apply(max)).T
d4 = pd.DataFrame(numAttributes.apply(lambda x: x.max() - x.min())).T
d5 = pd.DataFrame(numAttributes.apply(lambda x: x.skew())).T
d6 = pd.DataFrame(numAttributes.apply(lambda x: x.kurtosis())).T
d7 = d1 / ct1



# Concatenate
m = pd.concat([d2, d3, d4, ct1, ct2,ct3, d1, d7, d5, d6]).T.reset_index()
m.columns = ['Attributes', 'Min', 'Max', 'Range', 'Mean', 'Median', 'Mode', 'Std','Relative Std', 'Skew', 'Kurtosis']
m


count = 1
attributesToInspect = m['Attributes'].tolist()
rows= len(attributesToInspect)
plt.figure(figsize=(25,5*rows))
for i in attributesToInspect:
    plt.subplot(rows, 2, count)
    sns.histplot(x=i, bins=25, data=df01)
    count += 1
    
    plt.subplot(rows, 2, count)
    sns.histplot(x=i, bins=25, hue='Exited', data=df01)
    count += 1
plt.show()


catAttributes.apply(lambda x: x.unique().shape[0])


sns.countplot(x='Geography', data=df01)

count = 1
attributesToInspect = ['Geography', 'Gender']
rows= len(attributesToInspect)
for i in attributesToInspect:
    plt.subplot(rows, 2, count)
    sns.countplot(x=i, data=df01)
    count += 1
    
    plt.subplot(rows, 2, count)
    sns.countplot(x=i, hue='Exited', data=df01)
    count += 1
plt.show()


dfBool = pd.DataFrame(boolAttributes.apply(lambda x: x.sum())).rename(columns={0:"Yes"})
dfBool['No'] = dfBool['Yes'].apply(lambda row: df01.shape[0] - row)
dfBool.head()


df01.to_feather('00-Data/FeatherData/df01.feather')
