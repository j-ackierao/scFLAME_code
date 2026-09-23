# scFLAME/paper

Code to reproduce some figures and results from the paper.

`data_preprocessing' contains scripts to preprocess 8/9 of the real scRNA-seq data used in experiments in the paper: see Methods in the manuscript for a summary of how all data was downloaded and processed. The LUAD dataset uses raw count data from the sc_mixology repository (https://github.com/LuyiTian/sc_mixology; Tian et al.), preprocessed following https://github.com/songfd2018/BUSseq-1.1_implementation/tree/master/HumanLUAD. After the raw count data is obtained, dispersions and library sizes can be found in the same way as in other scripts.

The compute cluster all experiments were originally run on required a single self-contained file per job.