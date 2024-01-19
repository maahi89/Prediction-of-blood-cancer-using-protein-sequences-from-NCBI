#this file is used for separating heading and sequences from the fasta file

# Open the FASTA file for reading
with open("/content/sequence (1).fasta", "r") as fasta_file:
    lines = fasta_file.readlines()

headings = []
sequences = []

current_sequence = ""

# Iterate through the lines and separate headings and sequences
for line in lines:
    if line.startswith(">"):  # Lines starting with '>' are headings
        headings.append(line.strip())
        if current_sequence:
            sequences.append(current_sequence)
            current_sequence = ""
    else:
        current_sequence += line.strip()  # Lines not starting with '>' are sequence data

# Add the last sequence after the loop
if current_sequence:
    sequences.append(current_sequence)

# Print headings and corresponding sequences
for heading, sequence in zip(headings, sequences):
    print("Heading:", heading)
    print("Sequence:", sequence)
    print("-------")
