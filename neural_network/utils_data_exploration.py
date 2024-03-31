import pandas as pd
from sklearn.model_selection import train_test_split

def get_total_reviews_for_all_dataset(file_paths):
    """Iterate over a list of file paths and print the file path together with the total rows in it.
    Print the total reviews across all files.

    Parameter:
    -file_paths: a Python list containing file paths in string format.
    """
    total_reviews = 0
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        rows_in_file = df.shape[0]
        total_reviews += rows_in_file
        print(f"File: {file_path}, Reviews: {rows_in_file}")

    print("\nTotal reviews across all files:", total_reviews)


def process_csv_file(file_path):
    """Take the path to a csv file as an argument as a string, 
    read it with pandas and keep the following columns that are valuable for the task:
    -'review_text', rename to 'text'
    -'is_recommended', rename to 'label'
    -'rating', name stays unchanged.
    Return the DataFrame containing these three columns.
    
    Parameter:
    -file_path: path to a csv file."""
    
    #reading the csv file into a DataFrame with pandas
    df = pd.read_csv(file_path)

    #selecting the columns that are valuable for the task
    df = df[['review_text', 'is_recommended', 'rating']]

    #renaming the columns
    df.rename(columns={'is_recommended': 'label', 'review_text': 'text'}, inplace=True)

    return df


def print_label_percentages(df):
    """Take a DataFrame (df) as an argument and
    prints the percentage of positive and negative labels in it."""
    
    positive_percentage = round(df['label'].value_counts()[1] / len(df) * 100, 2)
    negative_percentage = round(df['label'].value_counts()[0] / len(df) * 100, 2)

    print("Positive labels percentage:", positive_percentage, "%")
    print("Negative labels percentage:", negative_percentage, "%")


def calculate_percentage_and_count_for_values(df, target_column, condition_column, values):
    """Calculate and print the percentages and counts for a list of specified values in a DataFrame.

    Parameters:
    - df: a pandas DataFrame
    - target_column: the name of the column for which to calculate the percentage in the df, provided as Python string
    - condition_column: the name of the column containing the condition for filtering, provided as Python string
    - values: a list of values to filter and calculate the percentages and counts in the df

    Returns None.
    Print the percentage and counts of the specified values."""
    
    for value in values:
        # filtering the df based on the condition
        filtered_df = df[df[condition_column] == value]

        # calculate positive and negative percentages
        positive_percentage = round(filtered_df[target_column].value_counts(normalize=True)[1] * 100, 2)
        negative_percentage = round(filtered_df[target_column].value_counts(normalize=True)[0] * 100, 2)

        # getting the counts
        positive_count = filtered_df[target_column].value_counts()[1]
        negative_count = filtered_df[target_column].value_counts()[0]

        print(f"""For {condition_column} value {value}: \n Positive {target_column}: {positive_percentage}% - count: {positive_count} \n Negative {target_column}: {negative_percentage}% - count: {negative_count} \n""")


def write_df_to_file(df, file_path, index=False):
    """Write a pandas DataFrame to a csv file.

    Parameters:
    - dataframe: pandas DataFrame
    - file_path: path to the output file in string format
    - index: whether to include the index in the output file as Boolean (default is False)
    """
    df.to_csv(file_path, index=index)
    print(f"df successfully written to {file_path}")

def split_and_write_data(df, input_dir, test_size=0.20, dev_size=0.15, random_state=None):
    """
    Split the input csv into training, development, and test sets, and write them to CSV files in the same directory.

    Parameters:
    - df (pandas DataFrame): the DataFrame to be split and written to files.
    - input_dir (str): path to the csv input file the DataFrame was created from.
    - test_size (float): proportion of the dataset to include in the test split, default=0.20
    - dev_size (float): proportion of the dataset to include in the development split, default=0.15
    - random_state (int or None): random seed for reproducibility, default=None 
    """
    
    # first split: data into training and test sets
    df_training, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    # second split: training data into training and dev sets
    df_train, df_dev = train_test_split(df_training, test_size=dev_size, random_state=random_state)

    # defining output file paths
    output_file_path_train = input_dir + "_training.csv"
    output_file_path_dev = input_dir + "_dev.csv"
    output_file_path_test = input_dir + "_test.csv"

    # writing DataFrames to files
    write_df_to_file(df_train, output_file_path_train)
    write_df_to_file(df_dev, output_file_path_dev)
    write_df_to_file(df_test, output_file_path_test)
