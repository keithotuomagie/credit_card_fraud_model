
# Business Problem

A financial services company wants to expand its business portfolio by entering the credit card business.  The company executives also recognize that fraud is a paramount issue.  As a result, the firm tasks KBO Analytics with the following:

- Create a model prototype to detect credit card fraud
- Identify characteristics that signal whether or not credit card fraud will take place

# Data Understanding

The data for examing the aforementioned problem comes from the following source: [Credit Card Fraud Data](https://www.kaggle.com/datasets/neharoychoudhury/credit-card-fraud-data/data)

Before beginning to identify any trends with customers that churn, I want to examine and become familiar with the dataset. I will conduct exploratory data analysis in order to understand the dataset attributes, which includes, but not limited to the following:

1. Number of Columns
2. Number of Rows
3. Column Names
4. Format of the data in each column

I created a Pandas Dataframe.  The Dataframe contains 14,446 rows of data.  The Dataframe contains 15 columns, which are the following:
    
1. Transaction Date and Time - (*fraud_df['trans_date_trans_time']*)
2. Merchant Name - (*fraud_df['merchant']*)
3. Category of Merchant - (*fraud_df['category']*)
4. Amount of Transaction - (*fraud_df['amt']*)
5. City of Credit Card Holder - (*fraud_df['city']*)
6. State of Credit Card Holder - (*fraud_df['state']*)
7. Latitute Location of Purchase - (*fraud_df['lat']*)
8. Longitude Location of Purchase - (*fraud_df['long']*)
9. Credit Card Holder's City Population - (*fraud_df['city_pop']*)
10. Job of Credit Card Holder - (*fraud_df['job']*)
11. Date of Birth of Credit Card Holder - (*fraud_df['dob']*)
12. Transaction Number - (*fraud_df['trans_num']*)
13. Latitude of Location of Merchant - (*fraud_df['merch_lat']*)
14. Longitude Location of Merchant - (*fraud_df['merch_long']*)
15. Whether Transaction is Fraud or Not - (*fraud_df['is_fraud']*)   

## Missing Data

I utilized the following code - *fraud_df.isna().sum()* - to check for missing values in each column.  There are no missing values in any of the columns.

## Duplicate Data

I utilized the following code - *fraud_df.duplicated().sum()* - to understand how many duplicated rows are in the dataframe.  There is a total of 63 duplicate rows.

## Examining Columns

I am going to conduct additional exploratory analysis for the following columns:
    
- Whether Transaction is Fraud or Not
- Merchant Name
- Category of Merchant
- City of Credit Card Holder
- State of Credit Card Holder
- Job of Credit Card Holder
- Date of Birth of Credit Card Holder
- Transaction Number

The following column - Whether Transaction is Fraud or Not - is the target.  This is the column that captures the non-fradulent and fradulent transactions.

The remaining columns are categorical.  I want to examine the distribution of the data within each column.

### Whether Transaction is Fraud or Not

**Observations | Whether Transaction is Fraud or Not**  

I utilized the following code - *fraud_df['is_fraud'].value_counts()* - to understand how many cases of fraud exist within the dataframe.

There 12,600 cases of no fraud.  There are 1,844 cases of fraud.  As expected, there is a class imbalance of fraud within the dataset.  In regards to modeling, accuracy will not be an appropriate metric to determine model performance.

I want to note there is a row with the following entry - *0"2019-01-01 00:00:44"*.  I am assuming this is a non-fradulent case in which the transaction date and time was incorporated.  I will remove the timestamp during the data cleaning phase.

I also want to note there is a row with the following entry - *1"2020-12-24 16:56:24"*.  I am assuming this is a fradulent case in which the transaction date and time was incorporated.  I will remove the timestamp during the data cleaning phase.

A bar chart that breaks down the non-fradulent and fradulent cases is below.

![Breakdown of Fraud Cases](image_1.png)

### Merchant Name

**Observations | Merchant Name** 

I utilized the following code - *fraud_df['merchant'].nunique()* - to identify the number of unique values in the Merchant column.  There are 693 different values in the Merchant column.  

I utilized the following code - fraud_df['merchant'].value_counts() - to understand the distribution of the categories within the Merchant column.  The top 5 values with their respective counts are the following:
    
- Kilback LLC - 58
- Cormier LLC - 48
- Kutch and Sons - 46
- Rau and Sons - 44
- Kiehn-Emmerich - 42
    
Based on the observations, there is a high degree of cardinality, or many unique values, within the Merchant column.
    
I utilized the following code - *fraud_df['merchant'].value_counts().plot()* - to provide a visualization of the high cardinality, which is below.

![Merchant - High Cardinality](image_2.png)

### Category (of Merchant)

**Observations | Category (of Merchant)**

I utilized the following code - *fraud_df['category'].nunique()* - to identify the number of unique values in the Category column.  There are 14 different values in the Category column.  

I utilized the following code - fraud_df['category'].value_counts() - to understand the distribution of the values within the Category column.  The top 5 values with their respective counts are the following:
    
- *grocery_pos* - 1602
- *gas_transport* - 1430
- *shopping_net* - 1408
- *shopping_pos* - 1354
- *home* - 1304
    
Based on the observations, there is a low degree of cardinality, or few unique values, within the Category column.
    
I utilized the following code - *fraud_df['category'].value_counts().plot()* - to provide a visualization of the low cardinality, which is below.

![Category - High Cardinality](image_3.png)

I created a dataframe that only has the fradulent credit card transactions.  The top five merchant values that have fradulent transactionss are the following:

- *grocery_pos* - 444
- *shopping_net* - 396
- *misc_net* - 223
- *shopping_pos* - 194
- *gas_transport* - 159

Upon observing the data, I see there are opportunities to group some of the merchant values together.  For example, *food_dining* category can become part of the *entertainment* category.

I also created a bar chart to represent which merchant values have the most fradulent cases.  The bar chart is below.

![Breakdown of Fraud Cases for each Merchant](image_4.png)

### City of Credit Card Holder

**Observations | City of Credit Card Holder**

I utilized the following code - *fraud_df['city'].nunique()* - to identify the number of unique values in the City column.  There are 176 different values in the City column.  

I utilized the following code - fraud_df['city'].value_counts() - to understand the distribution of the values within the City column.  The top 5 values with their respective counts are the following:
    
- Phoenix - 297
- Centerview - 197
- Orient - 192
- Sutherland - 187
- Fort Washakie - 187
    
Based on the observations, there is a high degree of cardinality, or many unique values, within the City column.
    
I utilized the following code - *fraud_df['city'].value_counts().plot()* - to provide a visualization of the high cardinality, which is below.

![City - High Cardinality](image_5.png)

### State of Credit Card Holder

**Observations | State of Credit Card Holder**

I utilized the following code - *fraud_df['state'].nunique()* - to identify the number of unique values in the State column.  There are 13 different values in the State column.  

I utilized the following code - fraud_df['state'].value_counts() - to understand the distribution of the values within the State column.  The top 5 values with their respective counts are the following:
    
- CA - 3375
- MO - 2329
- NE - 1460
- OR - 1211
- WA - 1150
    
Based on the observations, there is a low degree of cardinality, or few unique values, within the State column.
    
I utilized the following code - *fraud_df['state'].value_counts().plot()* - to provide a visualization of the low cardinality, which is below.

![State - High Cardinality](image_6.png)

I leveraged the dataframe that only has the fradulent credit card transactions.  The top five states that have fradulent transactionss are the following:

- CA - 410
- MO - 267
- NE - 238
- OR - 197
- WA - 126

Upon observing the data, I see there are opportunities to group some of the states together.  For example, I can group the states into the following categories: West, Southwest, and Midwest.  However, I am inclined to maintain California (CA), Alaska (AK), and HI (Hawaii) as standalone states.

I also created a bar chart to represent which states have the most fradulent cases.  The bar chart is below.

![Breakdown of Fraud Cases by State](image_7.png)

### Job of Credit Card Holder

**Observations | Job of Credit Card Holder**

I utilized the following code - *fraud_df['job'].nunique()* - to identify the number of unique values in the Job column.  There are 163 different categories in the Job column.  

I utilized the following code - fraud_df['job'].value_counts() - to understand the distribution of the values within the Job column.  The top 5 values with their respective counts are the following:
    
- Surveyor, minerals - 262
- Surveyor, land/geomatics - 240
- Land/geomatics surveyor - 225
- Insurance broker - 209
- Electronics engineer - 197
    
Based on the observations, there is a high degree of cardinality, or many unique values, within the Job column.
    
I utilized the following code - *fraud_df['job'].value_counts().plot()* - to provide a visualization of the high cardinality, which is below.

![Job - High Cardinality](image_8.png)

### Date of Birth of Credit Card Holder

I utilized the following code - *fraud_df['dob'].nunique()* - to identify the number of unique values in the Date of Birth column.  There are 187 different values in the Date of Birth column.  

I utilized the following code - fraud_df['dob'].value_counts() - to understand the distribution of the categories within the Date of Birth column.  The top 5 values with their respective counts are the following:
    
- July 17th, 1989 - 197
- June 21st, 1978 - 192
- October 24th, 1981 - 190
- February 11th, 1982 - 187
- October 28th, 1987 - 183
    
Based on the observations, there is a high degree of cardinality, or many unique values, within the Date of Birth column.
    
I utilized the following code - *fraud_df['dob'].value_counts().plot()* - to provide a visualization of the high cardinality, which is below.

![Date of Birth - High Cardinality](image_9.png)

### Transaction Number

**Observations | Transaction Number**

I utilized the following code - *fraud_df['trans_num'].nunique()* - to identify the number of unique values in the Transaction Number column.  There are 14383 different values in the Transaction Number column.  

I utilized the following code - *fraud_df['trans_num'].value_counts()* - to understand the distribution of the values within the Job column.
    
Based on the observations, this column has the highest degree of cardinality in comparison to all of the other columns in the dataset.  This is expected.  A transaction number is a unique identifier that is specific for a singular transaction.  
    
I utilized the following code - *fraud_df['trans_num'].value_counts().plot()* - to provide a visualization of the high cardinality, which is below.

![Transaction Number - High Cardinality](image_10.png)

# Data Preparation

# Modeling

# Overall Conclusion and Recommendations

## Overall Conclusion

## Recommendations
