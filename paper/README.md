# scFLAME/paper

Code to reproduce some figures and results from the paper.

`data_preprocessing' contains scripts to preprocess 8/9 of the real scRNA-seq data used in experiments in the paper: see Methods in the manuscript for a summary of how all data was downloaded and processed. 

The LUAD dataset uses raw count data from the sc_mixology repository (https://github.com/LuyiTian/sc_mixology; Tian et al.), preprocessed following https://github.com/songfd2018/BUSseq-1.1_implementation/tree/master/HumanLUAD. After the raw count data is obtained, dispersions and library sizes can be found in the same way as in other scripts.

While we provide our pre-processing script for the Pancreas dataset, this was largely adapted from the MNN2017 repository (https://github.com/MarioniLab/MNN2017/tree/master; Haghverdi et al.) where changes were made to replace outdated function calls and changing cell-type assignment as described in the manuscript. Notably, some data must be donwloaded by running https://github.com/MarioniLab/MNN2017/blob/master/DownloadData.sh which downloads a zipped file containing some raw count matrices for one of the 4 batches.

The compute cluster all experiments were originally run on required a single self-contained file per job.