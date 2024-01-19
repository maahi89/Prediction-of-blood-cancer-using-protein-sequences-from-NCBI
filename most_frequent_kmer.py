# this code is used for finding most_frequent_kmers for the project.

from Bio import SeqIO
from collections import Counter
#function for finding the most frequent k-mer
def find_most_frequent_kmer(sequence, k):
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)
    most_common_kmers = kmer_counts.most_common()
    max_count = most_common_kmers[0][1]
    most_frequent_kmers = [kmer for kmer, count in most_common_kmers if count == max_count]
    return most_frequent_kmers, max_count

#function for finding the most frequent k-mer from the fasta file
def find_most_frequent_kmer_from_fasta(file_path, k):
    sequences = SeqIO.parse("/content/sequence (1).fasta", "fasta")
    all_most_frequent_kmers = []
    max_count = 0
    for record in sequences:
        sequence = str(record.seq)
        most_frequent_kmers, current_max_count = find_most_frequent_kmer(sequence, k)
        if current_max_count > max_count:
            all_most_frequent_kmers = most_frequent_kmers
            max_count = current_max_count
    return all_most_frequent_kmers, max_count

fasta_file_path = "/content/sequence (1).fasta"
k = 7 # Change this to the desired k-mer length of your choice
most_frequent_kmers, max_count = find_most_frequent_kmer_from_fasta(fasta_file_path, k)
print(f"The most frequent {k}-mer(s) is/are: {most_frequent_kmers}")
print(f"Frequency: {max_count}")
