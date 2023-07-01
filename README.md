# worder

This project is an attempt to implement a GPU-augmented word frequency analyser using the CUDA 
programming language. As depicted in the following sections, compared to CPU, GPU has 
had significant performance and speedup.

## Input

### - Keywords
We used 10K most used English words from Google to use as our subject histogram and keywords.
The number and offset of keywords to read from the dataset file is arbitrary but subject to the upperbound of **1489**.

### - Data
We gathered text datasets of varying sizes from 5MB to 59MB for testing purposes. The dataset consists of texts from articles and books. Proportional to the size of the dataset, the application loads a specific number of words into memory.

| Data   | Number of Words to Read | Size (MB) |
|:------:|:-----------------------:|:---------:|
| Small | 131072 | 4 |
| Medium | 393216  | 12 |
| Large | 786432  | 24 |
| Huge | 1572864 | 48 |
