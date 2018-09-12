
# coding: utf-8

# In[26]:


#Realiza os imports necessários
import numpy  as  np
import pandas as  pd
import matplotlib.pyplot as plt


# In[27]:


#Lê dataset por meio do csv
data = pd.read_csv('precos_casa_california.csv')


# In[28]:


#Imprime as primeiras n linhas do dataset
data.head()


# In[29]:


#Imprime todas as informações sobre o dataset como: tipos de coluna, valores não nulos e uso da memória.
data.info()


# In[30]:


#Valores nulos do dataset são substituídos por NaN.
data = data.replace(' ',np.nan)


# In[31]:


#Remova os valores ausentes (NaN).
data = data.dropna()


# In[32]:


#Retorna as contagens de valores exclusivos da coluna ocean_proximity.
data['ocean_proximity'].value_counts()


# In[33]:


#Mapea os campos da coluna ocean_proximity, associando um objeto a um valor (0 = mais proximo)
dicionario = {'NEAR BAY':0,'<1H OCEAN':1,'INLAND':2,'NEAR OCEAN':3,'ISLAND':4}


# In[34]:


#Valores da coluna ocean_proximity do dataset são substituídos pelos valores do dicionario.
data = data.replace({"ocean_proximity":dicionario})


# In[35]:


#Converte todo o dataset para tipo numerico
data = data.apply(pd.to_numeric)


# In[39]:


#Gera estatísticas do dataset, como: contagem, media, min, max.
data.describe().transpose()


# In[70]:


#Imprime um histograma (representação dos dados) do dataset
get_ipython().run_line_magic('matplotlib', 'inline')
data.hist(bins=50,figsize=(15,10))
plt.show()


# In[72]:


data_labels = data['median_house_value']
data = data.drop('median_house_value',axis=1)

data_prepared = data
y_train = data_labels.copy()


# In[73]:


#Treinando e Construindo Modelo de Regressão Linear
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_prepared, data_labels, test_size = 0.2, random_state = 42)
model = LinearRegression()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)


# In[74]:


print(score)

