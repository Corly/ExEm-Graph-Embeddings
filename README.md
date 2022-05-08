# ExEm-Graph-Embeddings

Hi there! This is a custom implementation of the paper "ExEm: Expert Embedding using dominating set theory with deep learning approaches" (see citation below).
The purpose of this project is to create graph embeddings from texts.

In order to use it, you have to clone/download the repo and make minimal modifications.
To see an example, please check out main.py.

## Steps in the algorithm

1. Get a dataset. Right now the only dataset available in the datasets folder is Inspec.
From this dataset, the code requires a List[str] (aka a list of strings / texts).

2. Preprocess the text. The preprocess.preprocess.Preprocesser class loads a SpaCy model that will be used for lemmatization and transforming each word in its lemma id.
The "preprocess" function takes as arguments the texts (List[str] obtained at the previous step) and two boolean arguments specificing if you want the stop words and punctuations to be ignored.
It returns a List[List[int]], each text being replace by a list of the words' lemma ids from SpaCy.

3. Build a Coappearence graph. The coappearence graph is built based on a word window given as parameter and the List[List[int]] previously obtained.

4. Generate random walks starting from the graph. Previous to generation of the walks, a domination set is built from the graph.
The class RandomWalker receives as arguments the graph, the number of walks from node and the walk length.

5. The embeddings of the nodes are computed using either a Word2Vec model or a FastText model. See the trainer package for more details.
The models are saved to a specified path. This repo comes with 2 already compiled models that use the texts from the Inspec dataset ignoring the stop words and punctuation.

## Citations and Resources

ExEm paper:
```
@article{DBLP:journals/eswa/Nikzad-Khasmakhi21,
  author    = {Narjes Nikzad{-}Khasmakhi and
               Mohammad Ali Balafar and
               Mohammad{-}Reza Feizi{-}Derakhshi and
               Cina Motamed},
  title     = {ExEm: Expert embedding using dominating set theory with deep learning
               approaches},
  journal   = {Expert Syst. Appl.},
  volume    = {177},
  pages     = {114913},
  year      = {2021},
  url       = {https://doi.org/10.1016/j.eswa.2021.114913},
  doi       = {10.1016/j.eswa.2021.114913},
  timestamp = {Thu, 14 Oct 2021 08:46:38 +0200},
  biburl    = {https://dblp.org/rec/journals/eswa/Nikzad-Khasmakhi21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Inspec Dataset Citation:
```
@inproceedings{DBLP:conf/emnlp/Hulth03,
  author    = {Anette Hulth},
  title     = {Improved Automatic Keyword Extraction Given More Linguistic Knowledge},
  booktitle = {Proceedings of the Conference on Empirical Methods in Natural Language
               Processing, {EMNLP} 2003, Sapporo, Japan, July 11-12, 2003},
  year      = {2003},
  url       = {https://aclanthology.org/W03-1028/},
  timestamp = {Fri, 06 Aug 2021 00:40:21 +0200},
  biburl    = {https://dblp.org/rec/conf/emnlp/Hulth03.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Inspec Dataset Download Source:
```
https://github.com/LIAAD/KeywordExtractor-Datasets
```
