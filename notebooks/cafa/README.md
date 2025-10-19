## Competition
Predict gene ontology (GO) terms in each of the three subontolgies:
- Molecular Function (MF)
- Biological Process (BP)
- Cellular Component (CC)

In total there are three different testsets, one for each subontoloy.

## Metric
Weighted f1 where rare terms are weighted more highly.

## Submission file
Protein id, GO terms and probability for term tab-seperated (!), e.g.
```
P9WHI7   GO:0009274   0.931
P9WHI7   GO:0071944   0.540
P9WHI7   GO:0005575   0.324
P04637   GO:1990837   0.23
P04637   GO:0031625   0.989
P04637   GO:0043565   0.64
P04637   GO:0001091   0.49
```

### Optional Textual predictions
Predict functionality of proteins in textual form. Will be evaluated seperately.

## Background

### Gene ontology
Protein ontology is directed acyclic graph with three roots:
- MF: What it does on a molecular level?
- BP: Which biological processes does participates in?
- CC: Where in the cell is it located (CC)?

Codes are identified experimentally or computationally.
See [GO evidence codes](https://geneontology.org/docs/guide-go-evidence-codes/) for details.
Here, only experimentally proven codes are used (computationally derived ones are ignored!).

### Training data
Annotations from UniProtKB release of 18 June 2025 are used made up of proteins from eukaryotes and 17 bacteria-of-interest.
The participants are not required to use these data. Any additional data may be used available!

#### Gene ontology
Data is in file `go-basic.obo`. This structure is the 2025-06-01 release of the GO graph.
Use [obonet](https://github.com/dhimmel/obonet/blob/main/examples/go-obonet.ipynb) to handle OBO file type.

#### Training sequences
Training protein sequences are in file `train_sequences.fasta` taken from UniProt database, specifically from Swiss-Prot database (2025_03 release, 18 June 2025).

File contains `>DBNAME|UNITPROID|GENENAME` following lines `protein sequence`, e.g.
```
>sp|A0A0C5B5G6|MOTSC_HUMAN Mitochondrial-derived peptide MOTS-c OS=Homo sapiens OX=9606 GN=MT-RNR1 PE=1 SV=1
MRWQEMGYIFYPRKLR
>sp|A0JNW5|BLT3B_HUMAN Bridge-like lipid transfer protein family member 3B OS=Homo sapiens OX=9606 GN=BLTP3B PE=1 SV=2
MAGIIKKQILKHLSRFTKNLSPDKINLSTLKGEGELKNLELDEEVLQNMLDLPTWLAINK
VFCNKASIRIPWTKLKTHPICLSLDKVIMEMSTCEEPRSPNGPSPIATASGQSEYGFAEK
```

#### Labels
Labels for training sequences are contained in `train_terms.tsv`.
