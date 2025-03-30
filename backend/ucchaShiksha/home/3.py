import os
import markdown
import pandas as pd
from datasets import Dataset, load_from_disk
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import re
from bs4 import BeautifulSoup

def parse_markdown_file(file_path):
    """Read and parse a markdown file to extract plain text and split into data points."""
    with open(file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
        # Split the text into sections based on top-level headings
        sections = re.split(r'\n# ', md_text)
        
        data_points = []
        for section in sections:
            if section.strip():
                # Re-add the removed heading symbol for processing
                if not section.startswith('# '):
                    section = '# ' + section
                # Convert Markdown to HTML
                html = markdown.markdown(section)
                # Use BeautifulSoup to remove HTML tags and get plain text
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                data_points.append({"text": text})
    
    return data_points

def create_dataset_from_markdown(file_path, output_file):
    """Create a pretraining dataset from a markdown file."""
    # Parse the markdown file to extract data points
    data_points = parse_markdown_file(file_path)
    
    # Convert to a Dataset object
    dataset = Dataset.from_pandas(pd.DataFrame(data_points))
    
    # Save the dataset directly in the pretrainingdataset folder
    dataset.save_to_disk(output_file)
    print(f"Dataset saved to {output_file}")

def load_dataset(output_file):
    """Load the dataset from the specified file."""
    dataset = load_from_disk(output_file)
    return dataset

def print_dataset_summary(dataset):
    """Print general information about the dataset."""
    print("Dataset loaded successfully.")
    print(f"Number of rows: {len(dataset)}")
    print(f"Features: {dataset.features}")

def print_column_summary(dataset):
    """Print the summary of each column in the dataset."""
    print("\nColumn Summary:")
    for column in dataset.column_names:
        data_type = type(dataset[column][0])
        # Get a sample length or size if it's a string or list
        sample_length = None
        if isinstance(dataset[column][0], str):
            sample_length = len(dataset[column][0])
        elif isinstance(dataset[column][0], list):
            sample_length = len(dataset[column][0])
        elif isinstance(dataset[column][0], dict):
            sample_length = len(dataset[column][0])
        
        # Print the column summary
        print(f"{column}\n{data_type.__name__}\nSample Length: {sample_length}")

def print_first_n_entries(dataset, n=5):
    """Print the first n entries in the dataset."""
    print(f"First {n} entries in the dataset:")
    for i in range(n):
        print(f"Entry {i+1}: {dataset[i]}")

def convert_to_dataframe(dataset):
    """Convert the dataset to a Pandas DataFrame and return it."""
    df = dataset.to_pandas()
    return df

def access_data_example(dataset):
    """Example function to show how to access and manipulate dataset entries."""
    # Accessing a single entry
    entry = dataset[0]
    print("\nSingle entry:")
    print(entry)
    
    # Accessing a specific column
    texts = dataset['text']
    print("\nFirst 5 texts:")
    for text in texts[:5]:
        print(text)
    
    # Adding a new column
    new_column_data = ["Sample Data"] * len(dataset)
    dataset = dataset.add_column("new_column", new_column_data)
    print("\nDataset with new column:")
    print(dataset[0])

def main():
    # Create a Tkinter root window (hidden)
    root = Tk()
    root.withdraw()  # Hide the root window
    
    # Open a file dialog to select a Markdown file
    file_path = askopenfilename(title="Select a Markdown File", filetypes=[("Markdown files", "*.md")])
    if not file_path:
        print("No file selected.")
        return
    
    # Extract the directory name of the input file
    dir_name = os.path.basename(os.path.dirname(file_path))
    
    # Define the base output directory
    base_output_directory = 'pretrainingdataset'
    
    # Define the output file path using the directory name of the markdown file
    output_file = os.path.join(base_output_directory, f"{dir_name}_dataset")
    
    # Create the dataset from the selected Markdown file
    create_dataset_from_markdown(file_path, output_file)
    
    # Load the dataset
    dataset = load_dataset(output_file)
    
    # Print dataset summary
    print_dataset_summary(dataset)
    
    # Print column summary
    print_column_summary(dataset)
    
    # Print the first 5 entries
    print_first_n_entries(dataset)
    
    # Convert to DataFrame and print
    df = convert_to_dataframe(dataset)
    print("\nConverted to DataFrame:")
    print(df.head())
    
    # Example of accessing and manipulating data
    access_data_example(dataset)

if __name__ == "__main__":
    main()
