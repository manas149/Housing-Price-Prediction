#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real State - Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# for plotting histogram

# import matplotlib.pyplot as plt
# housing.hist(bins = 50, figsize = (20, 15))


# ## Train-Test Splitting

# In[9]:


#for learning purpose

# import numpy as np
# def split_train_test(data, test_ratio):
#     np.random.seed(42) #to fix shuffled value fix, train test fix
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


#train_set, test_set = split_train_test(housing, 0.2)

#print(len(train_set), len(test_set))


# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(len(train_set), len(test_set))


# ## Stratified Shuffling

# In[12]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[13]:


strat_test_set['CHAS'].value_counts()


# In[14]:


strat_train_set['CHAS'].value_counts()


# In[15]:


# 95/7


# In[16]:


# 376/28


# In[17]:


housing = strat_train_set


# In[18]:


housing.describe()


# ## Looking for Correlations

# In[19]:


corr_matrix = housing.corr()


# In[20]:


#corr_matrix


# In[21]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[22]:


# from pandas.plotting import scatter_matrix
# attributes = ["MEDV", "RM", "ZN", "LSTAT"]
# scatter_matrix(housing[attributes], figsize=(12, 8))


# In[23]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Trying Out Attribute Combinations

# In[24]:


#housing["TAXRM"] = housing["TAX"]/housing["RM"]


# In[25]:


#housing["TAXRM"]


# In[26]:


# corr_matrix = housing.corr()
# corr_matrix['MEDV'].sort_values(ascending=False)


# In[27]:


#housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[28]:


housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV"]


# In[29]:


housing.shape


# ## Missing Attributes

# In[30]:


# To take care of missing attributes, you have 3 options:
#     1. Get rid of the the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)


# In[31]:


a = housing.dropna(subset=["RM"]) #option1 *inplace= True
a.shape
# Note that the original housing dataframe will remain unchanged


# In[32]:


housing.drop("RM", axis=1) #option2
# Note that there is no RM column
# Note that the original housing dataframe will remain unchanged


# In[33]:


median = housing["RM"].median() # for option 3
median


# In[34]:


housing["RM"].fillna(median) #option3
# Note that the original housing dataframe will remain unchanged


# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[36]:


imputer.statistics_


# In[37]:


X = imputer.transform(housing)


# In[38]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[39]:


housing_tr.describe()


# ## Scikit-learn Design

# Primarily, three types of objects
# 
# 1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters
# 2. Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit transform() which fits and then transforms.
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.
#   

# ## Feature Scaling

# Primarily, two types of feature scaling methods: 
# 
#     1. Min-max scaling (Normalization) (value min)/(max - min)
#     Sklearn provides a class called MinMaxScaler for this
#     
#     2. Standardization
#     (value - mean)/std 
#     Sklearn provides a class called StandardScaler for this

# ## Creating a Pipeline

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy = "median")),
    #     .......... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[41]:


housing_num_tr = my_pipeline.fit_transform(housing) # its a numpy array because predictors take input numpy array in sklearn


# In[42]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon Real Estates

# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[44]:


some_data = housing.iloc[:5]


# In[45]:


some_labels = housing_labels[:5]


# In[46]:


prepared_data = my_pipeline.transform(some_data)


# In[47]:


model.predict(prepared_data)


# In[48]:


list(some_labels)


# ## Evaluating the model

# In[49]:


import numpy as np
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[50]:


rmse


# ## Using better evaluation technique - Cross Validation

# In[51]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)


# In[52]:


rmse_scores


# In[53]:


def print_scores(scores):
    print("Scores :", scores)
    print("Mean :", scores.mean())
    print("Standard deviation :", scores.std())


# In[54]:


print_scores(rmse_scores)


# Quiz: Convert this notebook into a python file and run the pipeline using Visual Studio Code

# ## Saving the model

# In[55]:


from joblib import dump, load
dump(model, 'Dragon.joblib')


# ## Testing the model on test data

# In[59]:


X_test = strat_test_set.drop("MEDV", axis = 1)
Y_test = strat_test_set["MEDV"]
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


# In[60]:


final_rmse


# In[61]:


prepared_data[0]


# ## Using the model

# In[62]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24050736, -19.31238772,  2.61111401, -6.0016859 , -0.5778192 ,
       -0.97491834,  5.41164221, -12.86091034]])
model.predict(features)


# In[ ]:




