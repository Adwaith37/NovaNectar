#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error


# In[14]:


campaign_desc = pd.read_csv('campaign_desc.csv')
campaign_table = pd.read_csv('campaign_table.csv')
causal_data = pd.read_csv('causal_data.csv')
coupon = pd.read_csv('coupon.csv')
coupon_redempt = pd.read_csv('coupon_redempt.csv')
hh_demo = pd.read_csv('hh_demographic.csv')
product = pd.read_csv('product.csv')
transaction_data = pd.read_csv('transaction_data.csv')


# In[3]:


datasets = {
    'Campaign Desc': campaign_desc,
    'Campaign Table': campaign_table,
    'Causal Data': causal_data,
    'Coupon': coupon,
    'Coupon Redempt': coupon_redempt,
    'Household Demographic': hh_demo,
    'Product': product,
    'Transaction Data': transaction_data
}


# In[4]:


print("Campaign Desc Columns:\n", campaign_desc.columns)
print("\nCampaign Table Columns:\n", campaign_table.columns)
print("\nCausal Data Columns:\n", causal_data.columns)
print("\nCoupon Columns:\n", coupon.columns)
print("\nCoupon Redempt Columns:\n", coupon_redempt.columns)
print("\nHousehold Demographic Columns:\n", hh_demo.columns)
print("\nProduct Columns:\n", product.columns)
print("\nTransaction Data Columns:\n", transaction_data.columns)


# In[5]:


for name, df in [('Campaign Desc', campaign_desc), ('Campaign Table', campaign_table),
                 ('Causal Data', causal_data), ('Coupon', coupon), 
                 ('Coupon Redempt', coupon_redempt), ('Household Demographic', hh_demo),
                 ('Product', product), ('Transaction Data', transaction_data)]:
    print(f"\n{name}:\n")
    print(df.isnull().sum(), "\n")


# In[6]:


def remove_high_null_columns(df, threshold=0.4):
    return df.dropna(axis=1, thresh=int(threshold * len(df)))

for name in datasets:
    datasets[name] = remove_high_null_columns(datasets[name])


# In[7]:


# Step 1: Calculate the engagement rise for each product
product_engagement = transaction_data.groupby(['PRODUCT_ID', 'WEEK_NO'])['QUANTITY'].sum().reset_index()

# Step 2: Calculate the week-over-week change in quantity sold
product_engagement['QUANTITY_CHANGE'] = product_engagement.groupby('PRODUCT_ID')['QUANTITY'].diff()

# Step 3: Find cumulative rise for each product
cumulative_rise = product_engagement.groupby('PRODUCT_ID')['QUANTITY_CHANGE'].sum().sort_values(ascending=False)

# Step 4: Line Plot → Top 5 products with highest engagement rise over time
top_products = cumulative_rise.head(5).index
plt.figure(figsize=(14, 8))

for product in top_products:
    product_data = product_engagement[product_engagement['PRODUCT_ID'] == product]
    plt.plot(product_data['WEEK_NO'], product_data['QUANTITY'], label=f'Product {product}', marker='o')

plt.title('Top 5 Products with Highest Rise in Engagement Over Time')
plt.xlabel('Week Number')
plt.ylabel('Quantity Sold')
plt.legend()
plt.grid(True)
plt.show()


# In[8]:


# Grouping sales by product
product_sales = transaction_data.groupby('PRODUCT_ID')['SALES_VALUE'].sum().reset_index()

# Sorting products based on total sales (highest first)
product_sales = product_sales.sort_values(by='SALES_VALUE', ascending=False)

# Display top 10 products by sales
print(product_sales.head(10))

# Plotting the top 10 products by sales
plt.figure(figsize=(12, 6))
sns.barplot(x='PRODUCT_ID', y='SALES_VALUE', data=product_sales.head(10), palette='viridis')
plt.title('Top 10 Products by Total Sales')
plt.xticks(rotation=45)
plt.show()


# In[9]:


print(coupon_redempt['DAY'].dtype)
print(campaign_desc['START_DAY'].dtype)
print(campaign_desc['END_DAY'].dtype)



# In[10]:


start_date = pd.Timestamp('2016-01-01')

# Convert DAY to integer by using .view() and then create timedelta
coupon_redempt['DATE'] = start_date + pd.to_timedelta(coupon_redempt['DAY'].view('int64'), unit='D')


start_date = pd.Timestamp('2016-01-01')

# Convert numeric days to actual dates
campaign_desc['START_DATE'] = start_date + pd.to_timedelta(campaign_desc['START_DAY'].fillna(0), unit='D')
campaign_desc['END_DATE'] = start_date + pd.to_timedelta(campaign_desc['END_DAY'].fillna(0), unit='D')

coupon_redempt['YEAR'] = coupon_redempt['DATE'].dt.year
coupon_redempt[['household_key', 'COUPON_UPC', 'CAMPAIGN', 'YEAR', 'DATE']].head()


# In[11]:


coupon_usage = coupon_redempt.groupby(['YEAR', 'COUPON_UPC']).size().reset_index(name='count')

# Find the highest used coupon for each year
highest_used_coupons = coupon_usage.loc[coupon_usage.groupby('YEAR')['count'].idxmax()]

print(highest_used_coupons)


# In[12]:


print(type(coupon_redempt))
print(type(product))


# In[15]:


print(coupon_redempt.columns)
print(product.columns)


# In[16]:


coupon_merged = pd.merge(coupon_redempt, coupon, on='COUPON_UPC', how='inner')
merged_data = pd.merge(coupon_merged, product, on='PRODUCT_ID', how='inner')
print(merged_data.head())


# In[17]:


coupon_classification = merged_data.groupby('COMMODITY_DESC').size().reset_index(name='COUPON_COUNT')
coupon_classification = coupon_classification.sort_values(by='COUPON_COUNT', ascending=False)
top_categories = coupon_classification.sort_values(by='COUPON_COUNT', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='COUPON_COUNT', y='COMMODITY_DESC', data=top_categories, palette='magma')

# Improving readability
plt.title('Top 20 Coupon Classifications by Product Category', fontsize=16)
plt.xlabel('Number of Coupons Used', fontsize=12)
plt.ylabel('Product Category', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Show the plot
plt.show()


# In[18]:


transaction_data['NET_SALES'] = (transaction_data['SALES_VALUE'] + 
                                 transaction_data['RETAIL_DISC'].fillna(0) + 
                                 transaction_data['COUPON_DISC'].fillna(0) + 
                                 transaction_data['COUPON_MATCH_DISC'].fillna(0))
#Replace zero values in NET_SALES before log transformation
transaction_data['NET_SALES_LOG'] = np.log1p(transaction_data['NET_SALES'].replace(0, np.nan))

# Feature Engineering: Create alternative features
transaction_data['NET_SALES_RATIO'] = transaction_data['NET_SALES'] / (transaction_data['QUANTITY'] + 1)

transaction_data['DISCOUNT_RATE'] = ((transaction_data['RETAIL_DISC'].fillna(0) + 
                                      transaction_data['COUPON_DISC'].fillna(0) + 
                                      transaction_data['COUPON_MATCH_DISC'].fillna(0)) /
                                      transaction_data['SALES_VALUE'].replace(0, np.nan))
coupon_usage = coupon_redempt.groupby('household_key')['COUPON_UPC'].count().reset_index()
coupon_usage.columns = ['household_key', 'COUPON_USAGE_FREQUENCY']
transaction_data = transaction_data.merge(coupon_usage, on='household_key', how='left')
transaction_data['COUPON_USAGE_FREQUENCY'].fillna(0, inplace=True)

# Step 4: Create Weekend Flag
transaction_data['IS_WEEKEND'] = transaction_data['DAY'] % 7 >= 5  # Saturday (5) or Sunday (6)

# Display the updated dataframe to check the new features
transaction_data[['NET_SALES', 'DISCOUNT_RATE', 'COUPON_USAGE_FREQUENCY', 'IS_WEEKEND']].head()



# In[19]:


start_date = pd.Timestamp('2016-01-01')

# Convert 'DAY' column to actual date in transaction_data
transaction_data['DATE'] = start_date + pd.to_timedelta(transaction_data['DAY'].fillna(0).astype(int), unit='D')

# Extract Year and Month
transaction_data['Year'] = transaction_data['DATE'].dt.year
transaction_data['Month'] = transaction_data['DATE'].dt.month

# Verify the year values
print("Unique Years in Data:", transaction_data['Year'].unique())

# Aggregate total sales by Year and Month
monthly_sales = transaction_data.groupby(['Year', 'Month'])['SALES_VALUE'].sum().unstack()

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(monthly_sales, annot=True, fmt=".0f", cmap="RdBu_r", linewidths=0.5)
plt.title("Month-wise Sales Frequency Heatmap")
plt.xlabel("Month")
plt.ylabel("Year")
plt.show()


# In[22]:


# Drop NET_SALES (to prevent leakage)
X = transaction_data.drop(columns=['SALES_VALUE', 'NET_SALES'])
y = transaction_data['SALES_VALUE']

# Keep Only Numeric Columns & Drop NA or Inf values
X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]  # Ensure y matches X after dropping rows

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Residual Analysis
plt.figure(figsize=(8, 5))
sns.histplot(y_test - y_pred, bins=50, kde=True, color='red')
plt.title("Residual Analysis (Error Distribution)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# In[24]:


absolute_error = abs(y_test - y_pred)

# Sort values for visualization
sorted_indices = np.argsort(y_test.values)
sorted_absolute_error = absolute_error.iloc[sorted_indices]

# Plot the absolute error
plt.figure(figsize=(12, 5))
plt.plot(sorted_absolute_error.values, marker='o', linestyle='-', color='red', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Reference line at 0
plt.xlabel("Sorted Index (Based on Actual Sales)")
plt.ylabel("Absolute Error")
plt.title("Absolute Error Across Predictions")
plt.grid(True)
plt.show()


# In[ ]:




