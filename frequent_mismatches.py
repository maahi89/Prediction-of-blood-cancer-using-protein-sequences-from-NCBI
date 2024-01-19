#code for finding most frequent mismatches

from Bio import SeqIO
def approx(kmer, substring, k, n):
	count = 0
	for i in range(k):
		if substring[i] != kmer[i]:
			count += 1
		if count > n:
			return False
	return True

fasta_file_path = input("Enter the path to the uploaded FASTA file: ")
k = int(input("What is the length of k-mer? "))
n = int(input("What is the maximum number of mismatches? "))
# Read the genomic sequence from the FASTA file
sequence = ""
for record in SeqIO.parse(fasta_file_path, "fasta"):
    sequence += str(record.seq)


DNA = str(sequence.upper())

counts = {}

for i in range(0, len(DNA) - k + 1):
	kmer = DNA[i:i + k]
	if kmer in counts:
		counts[kmer] += 1
	else:
		counts[kmer] = 1

updateCounts = {}

for a in counts:
	c = 0
	for b in counts:
		if approx(a, b, k, n):
			c += counts.get(b)
	updateCounts[a] = c

print(updateCounts)

frequent = max(updateCounts.values())

ans = []
for k in updateCounts:
	if updateCounts[k] == frequent:
		ans.append(k)

output = ' '.join(ans)
print(" ")
print("most frequent k-mers are: ",output)
