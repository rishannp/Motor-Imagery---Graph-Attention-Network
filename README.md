# Graph Attention Networks against Deep Learning Benchmark pipelines for 2 Class Motor Imagery Decoding in ALS and BCI Comp IV 2a. 
## Overview
Accurately classifying EEG signals, especially for individuals with neurodegenerative conditions like Amyotrophic Lateral Sclerosis (ALS), poses a significant challenge due to high inter-subject and inter-session changes in signal. This study introduces a novel three-layer Graph Attention Network (GAT) model for motor imagery (MI) classification, utilizing Phase Locking Value (PLV) as the graph input. The improvement underscores the effectiveness of graph-based representations in enhancing classification performance for neurodegenerative conditions. Evaluated across two datasets—the BCI Competition IV 2a dataset and a longitudinal ALS dataset with approximately 320 trials collected over 1-2 months—our GAT model not only achieves competitive accuracy but also shows robust performance across sessions and subjects. This stability is highlighted by statistically significant reductions in variance compared to state-of-the-art methods. These results support the hypothesis that phase-locking value-based graph representations can enhance neural representations in BCIs, offering promising avenues for more personalized approaches in MI classification. Our work emphasizes the potential for further optimizing GAT architectures and feature sets, pointing to future research directions that could improve performance and efficiency in MI classification tasks.

## Models Implemented
- EEGNet (Implementation from Scratch)
- EEGNet (Originally from V. Lawhern https://github.com/vlawhern/arl-eegmodels)
- DeepConvNet (Originally from Schirrmeister)
- ShallowConvNet (Originally from Schirrmeister)
- WaveletCNN (D.Zhang - 10.1109/ACCESS.2018.2889093)
- CSPCNN (J.Bang - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9325918)
- GCN 1
- GCN 2
- GAT 

## Results


## To Be Done:
- Publish ALS Dataset for open access and reproducibility
- CSP GAT : CSP Filter before converting into a graph. 
