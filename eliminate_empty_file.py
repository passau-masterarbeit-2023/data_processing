import os
import csv
import argparse

def remove_empty_csv_files(folder_path: str):
    i = 0
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # Check if the file has a .csv extension
        if file.endswith('.csv'):
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                first_data_row = next(reader, None)

                # If there's no data after the header, delete the file
                if not first_data_row:
                    print(f'Deleting file: {file_path}')
                    os.remove(file_path)
                    i+=1
    print(f'Deleted {i} files.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove empty CSV files from the specified folder')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing CSV files')
    args = parser.parse_args()

    folder_path = args.folder_path
    remove_empty_csv_files(folder_path)