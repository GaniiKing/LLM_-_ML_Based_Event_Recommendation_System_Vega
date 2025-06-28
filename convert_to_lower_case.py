import pandas as pd
import sys

def convert_csv_to_lowercase(input_file, output_file=None):
    """
    Convert all text in a CSV file to lowercase.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the output CSV file. 
                                   If not provided, will overwrite the input file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Convert all string columns to lowercase
        for column in df.columns:
            if df[column].dtype == 'object':  # Only convert string columns
                df[column] = df[column].str.lower()
        
        # Save the modified dataframe
        if output_file is None:
            output_file = input_file
            
        df.to_csv(output_file, index=False)
        print(f"Successfully converted {input_file} to lowercase")
        print(f"Saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    
    input_file = 'vega_detailed_csv_files/generated_events.csv'
    output_file = 'vega_detailed_csv_files/generated_events.csv'
    
    convert_csv_to_lowercase(input_file, output_file) 
