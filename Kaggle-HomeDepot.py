
# coding: utf-8

# In[58]:

get_ipython().magic(u'matplotlib inline')
import matplotlib


# In[3]:

import numpy as np
import pandas as pd


# In[4]:

# check files in directory
from subprocess import check_output
print(check_output(["ls"]).decode("utf8"))


# In[5]:

# load files
training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")
attribute_data = pd.read_csv("attributes.csv")
descriptions = pd.read_csv("product_descriptions.csv")


# In[6]:

# look at the top 10 rows
print(training_data.tail(10))
print(attribute_data.head(10))
print(descriptions.head(10))


# In[7]:

# merge descriptions
training_data = pd.merge(training_data, descriptions, on="product_uid", how="left")


# In[12]:

training_data.head(10)


# In[14]:

# merge product counts
product_counts = pd.DataFrame(pd.Series(training_data.groupby(["product_uid"]).size(), name="product_count"))
training_data = pd.merge(training_data, product_counts, left_on="product_uid", right_index=True, how="left")


# In[21]:

# merge brand names
# ...and change column name by .rename()
brand_names = attribute_data[attribute_data.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand_name"})
training_data = pd.merge(training_data, brand_names, on="product_uid", how="left")


# In[40]:

# Check if any column has NaN
print(training_data.isnull().sum())
# Which rows are NaN
rowsnan = np.where(training_data['brand_name'].isnull())[0]
training_data.ix[rowsnan].head(10)


# In[41]:

# fill in NaN
training_data.brand_name.fillna("Unknown", inplace=True)


# In[44]:

# structure of the dataframe
print(str(training_data.info()))
# basic statistics of each column
print(str(training_data.describe()))
training_data[:50]


# In[45]:

# Looking into the attribute_data
print(attribute_data.name.value_counts())
print(attribute_data.value[attribute_data.name == "Indoor/Outdoor"].value_counts())


# In[53]:

# cut data into categories
# add new column existing data frame
training_data["id_bins"] = pd.cut(training_data.id, 20, labels=False)
# correlation analysis "spearman"
print(training_data.corr(method="spearman"))
training_data.describe()


# In[68]:

# histogram and table 
# .value_counts() is table for R
import matplotlib
training_data.relevance.hist(bins=10)
training_data.relevance.value_counts()


# In[67]:

(descriptions.product_description.str.len() / 5).hist(bins=30)


# In[69]:

(training_data.product_title.str.len() / 5).hist(bins=30)


# In[73]:

(training_data.search_term.str.len() / 5.).hist(bins=30)
(training_data.search_term.str.count("\\s+") + 1).hist(bins=30)


# In[74]:

testing_data.product_uid.value_counts()


# In[75]:

# Distribution Cosine
training_products = training_data.product_uid.value_counts()
testing_products = testing_data.product_uid.value_counts()
training_norm =np.sqrt((training_products ** 2).sum())
testing_norm =np.sqrt((testing_products ** 2).sum())
product_uid_cos = (training_products * testing_products).sum() / (training_norm * testing_norm)
print("Product distribution cosine", product_uid_cos)


# In[79]:

import collections

chars = collections.Counter()
for title in training_data.product_title:
    chars.update(title.lower())
total = sum(chars.values())

print("Title char counts")
for c, count in chars.most_common(30):
    print("{}: {:.1f}%".format(c, 100. * count / total))
    
words = collections.Counter()
for title in training_data.search_term:
    words.update(title.lower().split())

total = sum(words.values())
print("Search word counts")
for word, count in words.most_common(200):
    print("{}: {:.1f}% ({:,})".format(word, 100. * count / total, count))


# In[86]:

print("Indoor/outdoor", training_data.search_term.str.contains("indoor|outdoor|interior|exterior", case=False).value_counts())
print("Contains numbers", training_data.search_term.str.contains("\\d", case=False).value_counts())


# In[95]:

for title in training_data.head(10).product_title:
    print title


# In[99]:

# "\n" means print on newline
# what terms are in attributes
print("\n" + "Color Family")
def summarize_values(name, values):
    values.fillna("", inplace=True)
    counts = collections.Counter()
    for value in values:
        counts[value.lower()] += 1
        
    total = sum(counts.values())
    print("{} counts ({:,} values)".format(name, total))
    for word, count in counts.most_common(20):
        print("{}: {:.1f}% ({:,})".format(word, 100. * count / total, count))

for attribute_name in ["Color Family", "Color/Finish", "Material", "MFG Brand Name", "Indoor/Outdoor", "Commercial / Residential"]:
    summarize_values("\n" + attribute_name, attribute_data[attribute_data.name == attribute_name].value)
    


# In[ ]:



