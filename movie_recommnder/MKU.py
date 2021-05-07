#!/usr/bin/env python
# coding: utf-8

# In[ ]:


MOVIE RECOMMENDATION SYSTEM


# IMPORTS RESOLVEMENT

# In[204]:


get_ipython().system('pip install -U -q PyDrive')
from oauth2client.client import GoogleCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[205]:


#https://drive.google.com/file/d/1gIA9AC8wNx2kdepuP2Y7K5J7LnihRD1V/view?usp=sharing
downloaded = drive.CreateFile({'id':'1gIA9AC8wNx2kdepuP2Y7K5J7LnihRD1V'}) 
downloaded.GetContentFile('Ratings.csv') 
#https://drive.google.com/file/d/11duMozcad56OxTZtGrW3nY2Ir0hKU8ul/view?usp=sharing
downloaded = drive.CreateFile({'id':'11duMozcad56OxTZtGrW3nY2Ir0hKU8ul'}) 
downloaded.GetContentFile('Teleplay.csv')
#https://drive.google.com/file/d/1TFQWYDQYwBf3cEtVsU6BtUnUZrwGOreH/view?usp=sharing
downloaded = drive.CreateFile({'id':'1TFQWYDQYwBf3cEtVsU6BtUnUZrwGOreH'}) 
downloaded.GetContentFile('New_Teleplay.csv')


# In[74]:



import pandas as pd
df=pd.read_csv('Teleplay.csv')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
v = TfidfVectorizer()
df=df.fillna("")
x = v.fit_transform(df['genre'])

#count_matrix = cv.fit_transform(x)
count_matrix = x
tp=pd.read_csv("Teleplay.csv")
tp=tp.drop('name',axis=1)
tp


# In[75]:


tp['type'].unique()


# In[76]:


rt=pd.read_csv('Ratings.csv')
rt.head()


# Removing -1 values from Ratings.csv

# In[77]:


rt.loc[rt['rating']==-1,'rating']= None
rt.head()


# Handling missing values

# In[78]:


tp.isna().sum()


# In[79]:


gen=tp['genre'].fillna("")
tp=tp.drop('genre',axis=1)
tp['genre']=gen
tp.head()


# In[80]:


print(tp['type'].value_counts())
typ=tp['type'].fillna(tp['type'].value_counts().idxmax())
tp=tp.drop('type',axis=1)
tp['type']=typ
tp.head()


# In[84]:


tp[tp['episodes']=='Unknown']=None
g=tp['episodes'].dropna()
tp=tp.drop('episodes',axis=1)
tp['episodes']=g
tp.head()


# In[90]:


tp=tp.dropna()
tp.isna().sum()


# Handling Outliers

# In[96]:


outliers=[]
def detect_outliers(data):
    
    threshold=3
    mean = np.mean(data)
    std =np.std(data)
    
    
    for i in data:
        z_score= (i - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

len(outlier_pt)


# In[103]:


outlier_pt=detect_outliers(tp['members'].astype(int))
tp.drop(tp[tp['members'].isin(outlier_pt)].index,inplace=True)

outlier_pt=detect_outliers(tp['episodes'].astype(int))
tp.drop(tp[tp['episodes'].isin(outlier_pt)].index,inplace=True)
tp


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
typ_label=le.fit_transform(tp['type'])
tp=tp.drop('type',axis=1)
tp['type']=typ_label
tp.head()


# In[131]:


tp=tp.reset_index(drop=True)
tp.head()


# In[141]:


from sklearn.preprocessing import OneHotEncoder
  
onehotencoder = OneHotEncoder()
r=pd.DataFrame(tp['type'])
#data = np.array(columnTransformer.fit_transform(r), dtype = np.str)
rated_dummies = pd.get_dummies(r.type)
type_names={}
for i in range(6):
  type_names[i]=le.inverse_transform([i])[0]
rated_dummies=rated_dummies.rename(columns=type_names)
rated_dummies.head()


# In[143]:


for r in rated_dummies.columns:
  tp[r]=rated_dummies[r]
tp.head()


# In[145]:


tp=tp.drop('type',axis=1)
tp.head()


# Add average rating from the users to tp dataframe

# In[149]:


rt=rt.dropna().reset_index(drop=True)
rt.head()


# In[155]:


avg_rt=pd.DataFrame(rt.groupby('teleplay_id').mean()['rating'])
df=pd.merge(tp, avg_rt, how='inner', left_on = 'teleplay_id', right_on = 'teleplay_id')
df=df.rename(columns = {"rating_x": "rating","rating_y":"average rating"})
df.head()


# In[164]:


from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
x = v.fit_transform(df['genre'])
gen_idf=pd.DataFrame(x.toarray())
gen_idf.head()


# In[192]:


from sklearn.preprocessing import MinMaxScaler

# create a scaler object
scaler = MinMaxScaler()
# fit and transform the data
df_norm = pd.DataFrame(scaler.fit_transform(df[['members','episodes']]), columns=['members','episodes'])

for r in rated_dummies.columns:
  df_norm[r]=rated_dummies[r]


# In[193]:


df_norm=df_norm.drop('Music',axis=1)


# In[253]:


for i in range(47):
  df_norm[i]=gen_idf[i]
df_norm.head()


# In[ ]:


cosine_sim = cosine_similarity(count_matrix)


# In[ ]:


def get_title_from_index(index):
    return df[df.index == index]["name"].values[0]
def get_index_from_title(title):
    return df[df.name == title].index[0]


# In[ ]:


user=dfr[dfr["user_id"]==53698]
required=user['rating'].max()
user_likes=user[user['rating']==required]


# In[ ]:


user_likes=pd.merge(user_likes,df[['teleplay_id','name','rating']],how='inner',left_on='teleplay_id',right_on='teleplay_id')
user_likes = user_likes.rename(columns = {'rating_x':'user rating','rating_y':'general rating'})
user_likes


# In[ ]:


movies={}
for n in df['name']:
  movies[n]=0

for n in user_likes['name']:
  movie_user_likes = n
  movie_index = get_index_from_title(movie_user_likes)
  similar_movies = list(enumerate(cosine_sim[movie_index]))
  sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
  i=0
  for element in sorted_similar_movies:
      movies[get_title_from_index(element[0])]+=1
      print(get_title_from_index(element[0]),end = " ")
      i=i+1
      if i>5:
          break


# In[ ]:


recommended = sorted(movies.items(), key=lambda x:x[1],reverse=True)[:100]
print("The top 5 movies recommended are : ")
i=0
for l,k in recommended:
  if l not in list(user_likes['name']):
      print(l)
      i+=1
  if i==5:
    break


# COLLABORATIVE FILTERING OF THE VARIABLES

# In[ ]:


dfr['rating'].replace(-1,0,inplace=True)
dfc=pd.merge(dfr,df[['teleplay_id','name']],how='inner',left_on='teleplay_id',right_on='teleplay_id')
dfc=dfc.drop('teleplay_id',axis=1)
dfc.head()


# In[ ]:


movie_matrix_UII = dfc.pivot_table(index='user_id', columns='name', values='rating')
movie_matrix_UII.head()


# # DIVIVDING THE VARIABLE TO DIFFERENT TRAINING AND TEST SETS FOR THE MODEL LEARNING

# In[254]:


y=df['rating']
X=df_norm
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# TEST FOR THROUGH THE LINEAR REGRESSION MODEL

# In[255]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)


# In[256]:


from sklearn.metrics import mean_squared_error

rms = mean_squared_error(y_test, y_pred, squared=False)
rms


# In[257]:


regressor=LinearRegression()
regressor.fit(X,y)


# In[217]:


test_df=pd.read_csv("New_Teleplay.csv")
test_df


# In[218]:


from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
gen_test=test_df['genre'].fillna("")
test_df=test_df.drop('genre',axis=1)
test_df['genre']=gen_test
x = v.fit_transform(test_df['genre'])
gen_idf_test=pd.DataFrame(x.toarray())
gen_idf_test.head()


# In[219]:


print(test_df['type'].value_counts())
typ=test_df['type'].fillna(test_df['type'].value_counts().idxmax())
test_df=test_df.drop('type',axis=1)
test_df['type']=typ
test_df.head()


# In[220]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
typ_label=le.fit_transform(test_df['type'])
test_df=test_df.drop('type',axis=1)
test_df['type']=typ_label
test_df.head()

# One Hot Encoding

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
r=pd.DataFrame(test_df['type'])
#data = np.array(columnTransformer.fit_transform(r), dtype = np.str)
rated_dummies = pd.get_dummies(r.type)
type_names={}
for i in range(6):
  type_names[i]=le.inverse_transform([i])[0]
rated_dummies=rated_dummies.rename(columns=type_names)
rated_dummies.head()


# In[226]:


test_df.loc[test_df['episodes']=='Unknown','episodes']= None
test_df=test_df.drop('rating',axis=1)


# In[239]:


test_df=test_df.fillna(test_df.mean())
from sklearn.preprocessing import MinMaxScaler

# create a scaler object
scaler = MinMaxScaler()
# fit and transform the data
df_norm_test = pd.DataFrame(scaler.fit_transform(test_df[['members','episodes']]), columns=['members','episodes'])

for r in rated_dummies.columns:
  df_norm_test[r]=rated_dummies[r]
df_norm_test


# In[250]:


df_norm_test=df_norm_test.drop('Music',axis=1)


# In[258]:


for i in range(47):
  df_norm_test[i]=gen_idf_test[i]


# In[260]:


df_norm_test=df_norm_test.fillna(df_norm_test.mean())
y_ans=regressor.predict(df_norm_test)


# In[269]:


ans_df=pd.read_csv('New_Teleplay.csv')
ans_df.drop('rating',axis=1)
ans_df['rating']=y_ans
ans_df.to_csv("Final_rating.csv")


# Neural Network

# In[273]:



from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 


# In[302]:


NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(256, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# In[304]:



checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
NN_model.fit(X, y, epochs=300, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# In[305]:


wights_file = '/content/Weights-198--0.62641.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# In[306]:



def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'Id':pd.read_csv('test.csv').Id,'SalePrice':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

predictions = NN_model.predict(df_norm_test)
#make_submission(predictions[:,0],'submission(NN).csv')


# In[307]:


predictions


# In[308]:


ans_df=pd.read_csv('New_Teleplay.csv')
ans_df.drop('rating',axis=1)
ans_df['rating']=predictions
ans_df.to_csv("Final_rating_NN.csv")


# In[ ]:




