# CSCI 548 Group 6: Information Extraction - Named Entity Recognition (ditk.ner)

## Deliverables
1. DITKModel.py - abstract class with abstract methods
2. DITKModel_Impl.py - implmentation of DITKModel with methods structures ditk.ner.<methodname>

## Benchmarks
- CoNLL 2003
- OntoNotes 5.0
- CHEMDNER

Try to run model on all 3 datasets. If the dataset does not fit the format of your model, you will have to preprocess it so it does (Sentence vs Paragraph in CoNLL). If you are unable to run on a dataset, provide an explaination why so.  

## Evaluation Metrics
- F1
- Precision
- Recall

Try to train and evaluate on each of the above benchmark datasets. Do not worry about the score. If you are unable to run on a specific benchmark, provide explaination why.

## FAQ
1. My python version is different
 - Separate packages for python version on gitbub (how?)
 
2. My dependency versions are different (e.g. tensorflow)
- For tensorflow there is package to make them compatible (prof will send more details)

3. Do we need to use pretrained weights
- No, just evaluation is fine

4. What is the pointer to parent class
- Just add url link to the parent class aka this repo
