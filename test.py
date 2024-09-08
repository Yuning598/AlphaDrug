import pandas as pd
from easydict import EasyDict
import selfies as sf

def splitSmi(smiles):
    # Example function to split SMILES, replace with your actual implementation
    return list(smiles)

def to_selfies(smiles):  # Convert SMILES to SELFIES
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        print("EncoderError in to_selfies()")
        return None

def main():
    args = {}  # Populate your args as needed

    # Get a sample of the first 1000 rows
    data = pd.read_csv('./data/small_train-val-data.tsv', sep='\t')  # Load only the first 1000 rows
    # Save the sample to a file
    data.to_csv('./data/small_train-val-data.tsv', sep='\t', index=False)


def clean_file(input_file, output_file):
    # Read the original file
    data = pd.read_csv(input_file, sep='\t')

    # Print the number of rows before cleaning
    print(f"Number of rows before cleaning: {len(data)}")

    # Convert SMILES to SELFIES
    data['selfies'] = data['smiles'].apply(to_selfies)

    # Remove rows where SELFIES conversion failed
    cleaned_data = data.dropna(subset=['selfies'])
    cleaned_data = cleaned_data.drop(columns=['smiles'])

    # Print the number of rows after cleaning
    print(f"Number of rows after cleaning: {len(cleaned_data)}")

    # Save the cleaned data to a new file
    cleaned_data.to_csv(output_file, sep='\t', index=False)
    print(f"Cleaned data saved to {output_file}")
def clean():
    clean_file('./data/small_with_selfies.tsv','./data/small_with_selfies.tsv')
def transformSMI():
    args = {}

    sample_data = pd.read_csv('./data/train-val-data.tsv', sep='\t')
    # Convert SMILES to SELFIES
    sample_data['selfies'] = sample_data['smiles'].apply(to_selfies)

    # Save the new data with SELFIES to a file
    sample_data.to_csv('./data/smile_train-val-data.tsv', sep='\t', index=False)











if __name__ == "__main__":
    import torch

    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        # Print the IDs and names of each GPU
        for i in range(num_gpus):
            print(f"Device ID: {i}, Device Name: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPU found.")