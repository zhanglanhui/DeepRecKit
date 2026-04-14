import os
import pandas as pd

def read_all_csv_files_in_directory(directory):
    file_list = os.listdir(directory)
    csv_files = [file for file in file_list if file.endswith('.csv')]

    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)
    return result_df

def read_all_files_in_directory(directory):
    file_list = os.listdir(directory)
    text_files = [file for file in file_list if file.startswith('part')]
    data = []
    for text_file in text_files:
        file_path = os.path.join(directory, text_file)
        with open(file_path, 'r') as file:
            file_data = file.readlines()
            data.extend(file_data)

    return data

# directory_path = "/path/to/your/csv/files"
# combined_df = read_all_csv_files_in_directory(directory_path)
