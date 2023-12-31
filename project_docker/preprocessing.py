import pandas as pd 
df = pd.read_csv("Bank_Deposite.csv")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


print(df.info())

######################################################################################333
# Get statistical analysis
print(df['age'].describe())

# Define figure size
plt.figure(figsize=(10, 7))

# Plot the histogram
ax = sns.histplot(df['age'], bins=50, kde=True)
# Add labels and title
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xticks([i for i in range(15, 105, 5)])

# Show the plot
plt.show()

###########################################################################################

print(df['job'].value_counts())


df['job'] = df['job'].replace('unknown', 'others')
print(df['job'].value_counts())


# Define counts
job_counts = df['job'].value_counts()
print(job_counts)
# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=job_counts.index, y=job_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Job Category')
plt.xlabel('\nJob')
plt.ylabel('Count')
plt.xticks(rotation=90, ha='right')
# Show the plot
plt.show()


########################################################################################



# Count the occurrences of each unique value in the 'marital' column
marital_counts = df['marital'].value_counts()
# Print the result
print(marital_counts)

# Define figure size
plt.figure(figsize=(12, 6))
# Plot bar chart
sns.barplot(x=marital_counts.index, y=marital_counts.values, palette='viridis')
# Add labels and title
plt.title('Count of Each Marital-Status Category')
plt.xlabel('\nMarital-Status')
plt.ylabel('Count')
# Show the plot
plt.show()

#############################################################################################

# Count the occurrences of each unique value in the 'education' column
education_counts = df['education'].value_counts()

# Print the result
print(education_counts)

# Replace 'unknown' with 'others' in the 'education' column
df['education'] = df['education'].replace('unknown', 'others')

# Count the occurrences of each unique value in the updated 'education' column
education_counts = df['education'].value_counts()

# Print the result
print(education_counts)

# Define Counts
education_counts = df['education'].value_counts()

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=education_counts.index, y=education_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Educational-Status Category')
plt.xlabel('\nEducational-Status')
plt.ylabel('Count')

# Show the plot
plt.show()

#############################################################################################


#Whether the customer has credit in default or not.
# Count the occurrences of each unique value in the 'default' column
default_counts = df['default'].value_counts()

# Print the result
print(default_counts)

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart with a different color palette (e.g., 'Set2')
sns.barplot(x=default_counts.index, y=default_counts.values, palette='Set2')

# Add labels and title
plt.title('Count of Each Credit in Default Category')
plt.xlabel('\nDefault')
plt.ylabel('Count')
# Show the plot
plt.show()

###########################################################################################

# Drop the 'default' column from the DataFrame in-place
# the "no" values is so poor
df.drop(columns=['default'], inplace=True)

# Assuming train_df is your DataFrame
balance_stats = df['balance'].describe()

# Print the descriptive statistics
print(balance_stats)

# Define figure size
plt.figure(figsize=(10, 6))

# Plot the histogram
sns.kdeplot(df['balance'], palette='viridis')

# Add labels and title
plt.title('Kernel Density Estimate of Account Balances')
plt.xlabel('Balance')
plt.ylabel('Density')
# Show the plot
plt.show()

####################################################################################################

# Count the number of records where the 'balance' is less than or equal to zero
balance_less_than_or_equal_to_zero_count = df[df['balance'] <= 0]['balance'].count()

# Print the result
print(f"Number of records with balance less than or equal to zero: {balance_less_than_or_equal_to_zero_count}")

# Define the percentile threshold
percentile_threshold = 95

# Calculate the specified percentile
percentile_value = int(np.percentile(df['balance'], percentile_threshold))

# Identify potential outliers
outliers = df[df['balance'] > percentile_value]

# Print the results
print(f'{percentile_threshold}th Percentile Value: {percentile_value}')
print(f'Number of Potential Outliers: {len(outliers)}')


# Filter rows in the 'df' DataFrame where 'balance' is less than or equal to 5768
df = df[df['balance'] <= 5768]

# Define figure size
plt.figure(figsize=(10, 6))

# Plot the histogram
sns.kdeplot(df['balance'], palette='viridis')

# Add labels and title
plt.title('Kernel Density Estimate of Account Balances')
plt.xlabel('Balance')
plt.ylabel('Density')
# Show the plot
plt.show()

################################################################################

# Define figure size
plt.figure(figsize=(8, 5))

# Plot the boxplot
sns.boxplot(df['balance'], palette='viridis')

# Add labels and title
plt.title('Box Plot of Account Balance')
plt.xlabel('Balance')
# Show the plot
plt.show()
################################################################################

# Define the percentile threshold
percentile_threshold = 5

# Calculate the specified percentile
percentile_value = int(np.percentile(df['balance'], percentile_threshold))

# Identify potential outliers
outliers = df[df['balance'] < percentile_value]

print(f'{percentile_threshold}th Percentile Value: {percentile_value}')
print(f'Number of Potential Outliers: {len(outliers)}')

df = df[df['balance'] > -191]

# Define figure size
plt.figure(figsize=(10, 6))

# Plot the histogram
sns.kdeplot(df['balance'], palette='viridis')

# Add labels and title
plt.title('Kernel Density Estimate of Account Balances')
plt.xlabel('Balance')
plt.ylabel('Density')
# Show the plot
plt.show()

#####################################################################################

# Define figure size
plt.figure(figsize=(8, 5))

# Plot the boxplot
sns.boxplot(df['balance'], palette='viridis')
# Add labels and title
plt.title('Box Plot of Account Balance')
plt.xlabel('Balance')
# Show the plot
plt.show()

#######################################################################################################3

# Count occurrences of unique values in the 'housing' column
housing_counts = df['housing'].value_counts()

# Display the result
print(housing_counts)

#Define counts
#Whether the customer has a housing loan or not.
housing_counts = df['housing'].value_counts()
print(housing_counts)

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=housing_counts.index, y=housing_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Housing Loan Category')
plt.xlabel('\nHousing Loan')
plt.ylabel('Count')
# Show the plot
plt.show()

##################################################################################################

# Count occurrences of unique values in the 'loan' column
#Whether the customer has a loan or not.
loan_counts = df['loan'].value_counts()

# Display the result
print(loan_counts)

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=loan_counts.index, y=loan_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Loan Category')
plt.xlabel('\nLoan')
plt.ylabel('Count')
# Show the plot
plt.show()

#########################################################################################3

#Type of communication used to contact customers

# Count occurrences of unique values in the 'contact' column
contact_counts = df['contact'].value_counts()

# Display the result
print(contact_counts)

# Replace 'unknown' with 'others' in the 'contact' column
df['contact'] = df['contact'].replace('unknown', 'others')

# Count occurrences of unique values in the 'contact' column
contact_counts = df['contact'].value_counts()

# Display the result
print(contact_counts)


# Define counts
contact_counts = df['contact'].value_counts()

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=contact_counts.index, y=contact_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Contact Category')
plt.xlabel('\nContact')
plt.ylabel('Count')
# Show the plot
plt.show()

###########################################################################################3

#Day of the month when customers were last contacted

# Count occurrences of unique values in the 'day' column
day_counts = df['day'].value_counts()

# Display the result
print(day_counts)

# Define counts
day_counts = df['day'].value_counts()

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=day_counts.index, y=day_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Day Category')
plt.xlabel('\nDay')
plt.ylabel('Count')
# Show the plot
plt.show()

#######################################################################################33

#last contact month of year.
# Count occurrences of unique values in the 'month' column
month_counts = df['month'].value_counts()

# Display the result
print(month_counts)

# Define counts
month_counts = df['month'].value_counts()

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=month_counts.index, y=month_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Month Category')
plt.xlabel('\nMonth')
plt.ylabel('Count')
# Show the plot
plt.show()

################################################################################################
#last contact duration, in seconds

# Get descriptive statistics for the 'duration' column
duration_stats = df['duration'].describe()

# Display the result
print(duration_stats)

# Define figure size
plt.figure(figsize=(10, 6))

# Plot the histogram
ax = sns.histplot(df['duration'], bins=50, kde=True, palette='viridis')

# Add labels and title
plt.title('Distribution of Duration')
plt.xlabel('Duration')
plt.ylabel('Frequency')
# plt.xticks([i for i in range(15, 100, 5)])
# Show the plot
plt.show()

############################################################################################

#number of contacts performed during this campaign and for this client
df['campaign'].describe()

# Define figure size
plt.figure(figsize=(10, 6))

# Plot the histogram
ax = sns.histplot(df['campaign'], bins=30, kde=True, palette='viridis')

# Add labels and title
plt.title('Distribution of Campaign')
plt.xlabel('Ccampaign')
plt.ylabel('Frequency')
# plt.xticks([i for i in range(15, 100, 5)])

# Show the plot
plt.show()

###############################################################################################3

#number of contacts performed before this campaign and for this client.
df['previous'].value_counts()


df.drop(columns=['previous'], inplace=True)

#outcome of the previous marketing campaign
df['poutcome'].value_counts()

# Define counts
pout_counts = df['poutcome'].value_counts()

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=pout_counts.index, y=pout_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Previous Outcome Category')
plt.xlabel('\nPOut')
plt.ylabel('Count')
# Show the plot
plt.show()


df.drop(columns=['poutcome'], inplace=True)

############################################################################################3

#has the client subscribed a term deposit
df['y'].value_counts()

# Define counts
target_counts = df['y'].value_counts()

# Define figure size
plt.figure(figsize=(12, 6))

# Plot bar chart
sns.barplot(x=target_counts.index, y=target_counts.values, palette='viridis')

# Add labels and title
plt.title('Count of Each Target Class')
plt.xlabel('\nSubscribes')
plt.ylabel('Count')
# Show the plot
plt.show()

################################################################################################

#Create a Label Encoder model to convert the categorical values into numeric
df = df.apply(LabelEncoder().fit_transform)

# Calculate the correlation matrix
correlation_matrix = df.corr()
# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()