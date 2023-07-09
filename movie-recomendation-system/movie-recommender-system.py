#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv


# In[3]:


movies = pd.read_csv('tmdb_5000_movies.csv') #tmdb is a website of rating movies
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.head(1)


# In[5]:


credits.head(1)


# In[6]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head(1)


# In[8]:


# genres
# id
# keywords
# title
# overview
# cast
# crew

movies= movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.head()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','Adventure','Fantasy','SciFi']


# In[15]:


import ast


# In[16]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[17]:


movies.dropna(inplace=True)


# In[18]:


movies['genres'] = movies['genres'].apply(convert)


# In[19]:


movies.head()


# In[20]:


movies['keywords'].apply(convert)


# In[21]:


movies.head()


# In[22]:


movies['cast'][0]


# In[23]:


import ast


# In[24]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[25]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[26]:


movies['crew'][0]


# In[27]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[28]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[29]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[30]:


movies.head()


# In[31]:


movies['overview'][0]


# In[32]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[33]:


movies.head()


# In[34]:


movies['genres'] =movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] =movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] =movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] =movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[35]:


movies.head()


# In[36]:


movies['tags'] = movies['overview'] + movies['genres']+ movies['keywords']+ movies['cast']+ movies['crew']


# In[37]:


movies.head()


# In[38]:


new_df = movies[['movie_id','title','tags']]


# In[39]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[40]:


new_df.head()


# In[41]:


new_df['tags'][0]


# In[42]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[43]:


new_df.head()


# # Text vectorization

# In[44]:


new_df['tags'][1]


# In[45]:


from sklearn.feature_extraction.text import CountVectorizer      #by countvectorizer we can see that which 5000 words are they
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[46]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[47]:


vector


# In[48]:


vector[0]


# In[49]:


vector.shape


# ['loved','loving','love'] 
# 
# 
#  ['love','love','love']     steming 

# In[50]:


import nltk


# In[51]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[52]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[53]:


new_df['tags']= new_df['tags'].apply(stem)


# In[54]:


ps.stem('loving')


# In[55]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction [ { " i d " :  1 4 6 3 ,  " n a m e " :  " c u l t u r e  c l a s h " } ,  { " i d " :  2 9 6 4 ,  " n a m e " :  " f u t u r e " } ,  { " i d " :  3 3 8 6 ,  " n a m e " :  " s p a c e  w a r " } ,  { " i d " :  3 3 8 8 ,  " n a m e " :  " s p a c e  c o l o n y " } ,  { " i d " :  3 6 7 9 ,  " n a m e " :  " s o c i e t y " } ,  { " i d " :  3 8 0 1 ,  " n a m e " :  " s p a c e  t r a v e l " } ,  { " i d " :  9 6 8 5 ,  " n a m e " :  " f u t u r i s t i c " } ,  { " i d " :  9 8 4 0 ,  " n a m e " :  " r o m a n c e " } ,  { " i d " :  9 8 8 2 ,  " n a m e " :  " s p a c e " } ,  { " i d " :  9 9 5 1 ,  " n a m e " :  " a l i e n " } ,  { " i d " :  1 0 1 4 8 ,  " n a m e " :  " t r i b e " } ,  { " i d " :  1 0 1 5 8 ,  " n a m e " :  " a l i e n  p l a n e t " } ,  { " i d " :  1 0 9 8 7 ,  " n a m e " :  " c g i " } ,  { " i d " :  1 1 3 9 9 ,  " n a m e " :  " m a r i n e " } ,  { " i d " :  1 3 0 6 5 ,  " n a m e " :  " s o l d i e r " } ,  { " i d " :  1 4 6 4 3 ,  " n a m e " :  " b a t t l e " } ,  { " i d " :  1 4 7 2 0 ,  " n a m e " :  " l o v e  a f f a i r " } ,  { " i d " :  1 6 5 4 3 1 ,  " n a m e " :  " a n t i  w a r " } ,  { " i d " :  1 9 3 5 5 4 ,  " n a m e " :  " p o w e r  r e l a t i o n s " } ,  { " i d " :  2 0 6 6 9 0 ,  " n a m e " :  " m i n d  a n d  s o u l " } ,  { " i d " :  2 0 9 7 1 4 ,  " n a m e " :  " 3 d " } ] SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[56]:


# Euclidian dist is not for higher dimentional data
# Cosian dist is used for higher dimentional data


# In[57]:


from sklearn.metrics.pairwise import cosine_similarity


# In[58]:


similarity = cosine_similarity(vectors)   #dist calculated by each movie to other movies


# In[ ]:


similarity[0]


# In[ ]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        


# In[ ]:


recommend('Batman')      #recommending a movie 


# In[ ]:


import pickle


# In[ ]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[ ]:


new_df['title'].values


# In[ ]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[ ]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




