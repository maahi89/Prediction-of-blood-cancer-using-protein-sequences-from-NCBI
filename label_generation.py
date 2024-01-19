#code for reading file and generating label
import random

# Function to generate random labels for sequences
def generate_random_labels(sequences):
    return [random.choice(["label_A", "label_B"]) for _ in sequences]

# Function to save labels to a file
def save_labels(labels, file_path):
    with open(file_path, 'w') as file:
        for label in labels:
            file.write(label + '\n')

# Path to your FASTA file
fasta_file_path = '/content/sequence (1).fasta'

# Read sequences from the FASTA file
sequences = read_sequences(fasta_file_path)

# Generate random labels for each sequence
labels = generate_random_labels(sequences)

# Path to save the labels file
labels_file_path = '/content/labels.txt'

# Save labels to a file
save_labels(labels, labels_file_path)
