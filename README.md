# Market-Basket-Insights
Step 1 – Importing required libraries
We will be using the mlxtend Apriori library in Python, as shown below:
//Import required libraries

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
Step 2 – Loading the data
//Load the data
data = pd.read_csv('data.csv', encoding= 'unicode_escape')
Step 3 – Cleaning the data
//Remove spaces from the description column
data['Description'] = data['Description'].str.strip()
 
#Drop rows without invoice number
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
 
//Remove the credit transaction with invoice numbers containing 'C'
data = data[~data['InvoiceNo'].str.contains('C')]
data.head()
 
 
Step 4 – Creating basket
We are going to create a basket matrix by grouping multiple items within the same order and then unstack our DataFrame:




basket = (data[data['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum()
          .unstack()
          .reset_index()
          .fillna(0)
          .set_index('InvoiceNo'))
basket.head(10)
 
Step 5 – Encoding
We now need to encode the values in our matrix to 1’s and 0’s. We will do this in the following manner:
•	Set the value to 0 if it is less than or equal to 0.
•	Set the value to 1 if it is greater than or equal to 1.
def encode_units(x):
    if x <= 0:
        return 0    
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
basket_sets.head()
 

Step 6 – Generating frequent item sets
We will generate frequent item sets that have a support of at least 7%: //Generate frequent itemsets
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
Step 7 – Generating association rules
We will generate association rules with their corresponding support, confidence, and lift:
//Generating the rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
 
rules.head()

Step 8 – Filtering rules with high Confidence and Lift
As we already know, the greater the lift ratio, the more significant the association between the items. Also, the higher the confidence, the more reliable are the rules.
So, we are looking for rules with high confidence (>=0.8) and high lift (>=6). For this we filter the records as shown:
//Filtering out the values with lift > = 6 and confidence > = 0.8
rules[ (rules['lift'] >= 6) & (rules['confidence'] >= 0.8) ]
 
 
So, we have our antecedents and consequent items along with their parameter values.
The result is giving us a lot of information about item grouping such as customers are 84% likely to buy the green alarm clock if they have already bought the red alarm clock!
