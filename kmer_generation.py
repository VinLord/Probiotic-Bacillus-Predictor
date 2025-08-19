Python 3.11.5 (tags/v3.11.5:cce6ba9, Aug 24 2023, 14:38:34) [MSC v.1936 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import os
... import csv
... import glob
... from collections import Counter
... import pandas as pd  # Ensure pandas is imported
... 
... def read_fasta_file(file_path):
...     with open(file_path, 'r') as file:
...         lines = file.readlines()
...     header = lines[0].strip()
...     sequence = ''.join(lines[1:]).replace('\n', '')
...     return header, sequence
... 
... def generate_kmers(sequence, min_k, max_k):
...     kmers_counter = Counter()
...     valid_nucleotides = {'A', 'T', 'C', 'G'}
...     for k in range(min_k, max_k + 1):
...         kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
...         # Filter out kmers that contain characters other than A, T, C, G
...         valid_kmers = [kmer for kmer in kmers if set(kmer).issubset(valid_nucleotides)]
...         kmers_counter.update(valid_kmers)
...     return kmers_counter
... 
... def process_files(input_folder, output_file, min_k, max_k):
...     # Find all .fna files in the input folder
...     fna_files = glob.glob(os.path.join(input_folder, '*.fna'))
...     all_kmers_counter = Counter()
...     data = []
... 
...     for fna_file in fna_files:
...         print(f"Processing {fna_file}...")
...         label = 0 if 'probiotic_' in os.path.basename(fna_file) else 1
...         _, sequence = read_fasta_file(fna_file)
...         kmers_counter = generate_kmers(sequence, min_k, max_k)
...         all_kmers_counter.update(kmers_counter.keys())
...         data.append({'Label': label, **kmers_counter})

    # Ensure all rows have the same columns
    for row in data:
        for kmer in all_kmers_counter:
            if kmer not in row:
                row[kmer] = 0

    # Create DataFrame
    df = pd.DataFrame(data)
    # Reorder DataFrame columns
    df = df[['Label'] + [col for col in df.columns if col != 'Label']]
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print("Process completed successfully. K-mer data has been saved to the CSV file.")

# Parameters
input_folder = "/content/files"
output_file = "/content/output/data_processed.csv"
min_k = 5
max_k = 10

# Process files and generate CSV
