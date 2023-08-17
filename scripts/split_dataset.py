import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset from the CSV file into a DataFrame
df = pd.read_csv('/home/wiseyak/suraj/Bengali_ASR/bengali_dataset.csv')

# Define the features (X) and the target (y)
df = df.loc[df['Transcript'].str.len() < 180]
# Split the dataset into train and test sets with a fixed random seed (e.g., 42)
# Set the test_size to the desired proportion of the test set (e.g., 0.2 for 20%)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_csv('bengali_dataset_train.csv',index=False)
test_df.to_csv('bengali_dataset_test.csv', index=False)
