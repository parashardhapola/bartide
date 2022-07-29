[![PyPI version shields.io](https://img.shields.io/pypi/v/bartide.svg)](https://pypi.python.org/pypi/bartide/)
[![PyPI license](https://img.shields.io/pypi/l/bartide.svg)](https://pypi.python.org/pypi/bartide/)

# Bartide
## Extract, correct and analyze barcodes from sequenced reads


### INSTALLATION

To install Bartide you need to have Python version 3.9 or upwards. The suggest wasy to install Python is to use Miniconda:
https://docs.conda.io/en/latest/miniconda.html

Use the following command to install Bartide
```
pip install bartide
```

#### Installation of NMSlib:
Easiet way to install NMSlib is to use the precompiled version from conda-forge repo. This works across Linux, MacOS and Windows machines.
```
conda install -c conda-forge nmslib
```

### USAGE


### 1. Extraction

Barcodes are extracted from the sequencing reads using the Bartide’s `BarcodeExtractor` class. This class is provided two input files, one for each end of paired-end sequencing, in FASTQ format. This class is designed to automatically extract the barcodes assuming that the barcodes are of the same lengths, they span the same position in the reads and the flanking sequence is constant. This is achieved by first summarizing the nucleotide composition at each position for all the reads (or a sample of reads). The flanking sequence will be dominated by a single nucleotide while the barcode should have variable base composition. This pattern is used to identify the position of barcodes and the flanking primer sequence. This behaviour can be overridden by providing the barcode length and the flanking primer sequence to `BarcodeExtractor`. Below we show an example of how to call the BarcodeExtractor class, and perform automatic flank identification.

```
import bartide

extractor = bartide.BarcodeExtractor(
	'sample1_read1.fastq.gz',
	'sample1_read2.fastq.gz'
)

extractor.identify_flanks()
```

Alternatively, users can provide the flanking sequence can the barcode length manually like below:

```
extractor = bartide.BarcodeExtractor(
	'sample1_read1.fastq.gz',
	'sample1_read2.fastq.gz',
	left_flank='GTAGCC',
	right_flank='AGATCG',
	barcode_length=27
)
```


Once the flank sequences and barcode length are determined, they are stored as `extractor.leftFlank`, `extractor.rightFlank` and `extractor.barcodeLength`. Now the barcodes can be extracted, and their frequency counted. The `BarcodeExtractor` class will compare the barcode sequence and its reverse complementary sequence from the other pair of the sequenced read. By default, there should not be more than 3 mismatches between the two sequences otherwise the extraction fails for that read. Users can change the maximum allowed mismatch value by using the `max_dist` parameter when calling `BarcodeExtractor`.

The actual extraction of barcodes is triggered by the following command:
```
extractor.count_barcodes()
```

Users can access these uncorrected barcodes with the following command:
```
print (extractor.rawCounts)
```
			
This prints the barcodes and their frequencies as shown below:

```
TTGTAGGGGTGTGTTCTACCGGTAATT    2843
GTGCTGGTAATGTGGGCGACGGTGGGG     913
TTGGTGAAGCATAGTTCCGTGATTGAA     909
TTCCATGACGTTAAATACCTCCTTATA     723
ATCTGGCGTCCAGCAGATATTAGTTTT     717
                               ... 
AAGTTACATGCCGCAAAGGGTTCTTTG       1
AAGGATGAATGACAAGGTGCTAGCCAT       1
GGTACAAGGCGGGATTACCATGCATTG       1
GTGCTGGAAATGTGGGCGACGGTGGGG       1
AAGTCACATGCCGCAAAGTGTCCATTG       1
```

### 2. Error correction

Next, the list obtained above may still contain barcodes that harbour sequencing errors. We assume that a barcode contains error(s) if it has less than three nucleotide difference with a higher abundance barcode. Since the pairwise comparison of all the barcodes can be computationally prohibitive, we use approximate nearest neighbour detection library ‘nmslib’ to efficiently identify similar barcodes. If an erroneous barcode is found, its frequency is added to the barcode with the nearest match. This functionality is implemented in the `SeqCorrect` class. Users can obtain the corrected barcode list, by running the following command:

```
corrector = bartide.SeqCorrect()
corrector.run(extractor.rawCounts)
```

The corrected list of barcodes is stored under `corrector.correctedCounts`. These barcodes can then be saved in CSV format table as shown below:

```
corrector.save_to_csv('barcodes_freq_sample1.csv')
```

The `SeqCorrect` class will by default, remove any barcode with a frequency of less than 20, as suggested previously (Naik et al., 2014). This behaviour can be overridden by changing the value of the `min_counts` parameter when calling `SeqCorrect`.

### 3. Sample comparison, analysis and visualization

Once the corrected barcode frequencies are saved for all the samples, they can be compared using the `BarcodeAnalyzer` class. This class is initialized by providing the name, with full path, of the directory wherein all the CSV files were saved. The following command illustrates this:

```
analyzer = bartide.BarcodeAnalyzer(‘barcodes_dir’)
```
	
This will lead to aggregation of all the barcodes across all the samples that can be accessed from a single table stored under `analyzer.barcodes`. Users can perform all the custom downstream analysis using this table as the starting point.

Bartide provides four essential plots that allow users to easily identify how the barcodes are shared across the sample.

The first plot is the ‘Upset’ that shows all the combinations of samples and the number of barcodes that overlap between them. This plot, compared to Venn diagrams, allow easy visualization of the overlaps and non-overlaps of barcodes. Please note that when using upset plots, we only look at the unique number of barcodes found in each of the samples and not their frequencies.

```
analyzer.plot_upset()
```
<img src="./notebooks/images/upset_plot.png" alt="upset_plot" width="300"/>

Sometimes due large difference in the number of barcodes captured, it might be difficult to easily identify the similarity or differences between the samples. To solve this, rather than using the absolute number of barcodes in a sample, the percentage overlap of barcodes from a sample with all other samples are used. This allows the barcodes from a sample to be defined in proportions and may allow insights into sample similarity that is otherwise to identify with absolute frequencies. The following command shows the proportions in form of a stacked barplot:

```
analyzer.plot_stacked()
```
<img src="./notebooks/images/stacked_plot.png" alt="stacked_plot" width="300"/>

An alternative way to deal with the situation wherein the absolute number of unique barcodes are quite different across the samples is to perform normalization by dividing the overlap value by the sum of the total barcodes from the two samples. The resulting normalized values can be visualized in form of a heatmap.

```
analyzer.plot_overlap_heatmap()
```
<img src="./notebooks/images/overlap_heatmap.png" alt="overlap_heatmap" width="300"/>

In all the above three plotting functions, we do not the frequencies of the barcodes, which are indicative of how dominant a particular barcode is in the samples. A weighted overlap of barcodes is calculated between two samples as following:

$$\sum_{b}^{B}\left(S_b^i-S_b^j\right)^2$$

Wherein, $S$ is a column-sum normalized matrix of samples (columns) and barcodes (rows) containing barcode frequencies, $i$ and $j$ are two samples, $b$ is a barcode in a set of barcodes $B$ that are present in either of the two samples or both. These overlap values are then plotted in form of a heatmap using the following function:

```
analyzer.plot_weighted_heatmap()
```
<img src="./notebooks/images/weighted_heatmap.png" alt="weighted_heatmap" width="300"/> 

To save the images generated by the functions above, users can pass a value (path and name of the file where to save) to the `save_name` parameter.
