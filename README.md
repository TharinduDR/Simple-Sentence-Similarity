# Sentence Similarity
Sentence SImilarity can be used for many domains in natural language processing, including information retrieval, translation memories, question answering etc. There are many methods to calculate the similarity between two strings. In this project I have evaluated those methods using SICK and STS data sets. 

*Traditional approaches* can be found in [Traditional Sentence Similarity.ipynb](https://github.com/TharinduDR/Simple-Sentence-Similarity/blob/master/Sentence%20Similarity%20-%20Traditional.ipynb). Methods like edit distance and levenshtein distance have been explored here. 

*Word Vector approaches* can be found in [Sentence Similarity - Word Vectors.ipynb](https://github.com/TharinduDR/Simple-Sentence-Similarity/blob/master/Sentence%20Similarity%20-%20Word%20Vectors.ipynb). Glove, word2vec and fasttext word embedding models were used for experiements. Distance measures like cosine similarity, word moving distance, smooth inverse frequency were considered.

*Context Vector approaches* can be found in [Sentence Similarity - Elmo Vectors.ipynb](https://github.com/TharinduDR/Simple-Sentence-Similarity/blob/master/Sentence%20Similarity%20-%20Context%20Vectors%20-%20ELMO.ipynb) and [Sentence Similarity - Flair Vectors.ipynb](https://github.com/TharinduDR/Simple-Sentence-Similarity/blob/master/Sentence%20Similarity%20-%20Context%20Vectors%20-%20FLAIR.ipynb). The same benchmarks in the *Word Vector approaches* were considered using ELMo, BERT and FLAIR embeddings and compared the results with word2vec embeddings. Results are shown in following tables. 

#### Word Average benchmark

| Model         | RMSE          |
| ------------- | ------------- |
| AVG-W2V         | 0.258  |
| AVG-ELMO        | 0.273  |
| AVG-FLAIR       | 0.366  |
| AVG-BERT        | 0.361  |
| AVG-BERT+ELMO   | 0.334  |
| AVG-W2V-STOP    | 0.240  |
| AVG-ELMO-STOP   | 0.263  |
| AVG-FLAIR-STOP  | 0.299  |
| AVG-BERT-STOP   | 0.344  |
| AVG-BERT+ELMO-STOP | 0.316  |
| AVG-W2V-TFIDF      | 0.229  |
| AVG-ELMO-TFIDF     | 0.253  |
| AVG-FLAIR-TFIDF    | 0.288  |
| AVG-BERT-TFIDF      | 0.340  |
| AVG-BERT+ELMO-TFIDF | 0.309  |
| AVG-W2V-TFIDF-STOP | 0.228  |
| AVG-ELMO-TFIDF-STOP| 0.249  |
| AVG-FLAIR-TFIDF-STOP | 0.269  |
| AVG-BERT-TFIDF-STOP | 0.331  |
| AVG-BERT-TFIDF-STOP | 0.300  |

#### Smooth Inverse Frequency benchmark

| Model  | RMSE |
| ------------- | ------------- |
| SIF - W2V <sup>*</sup>    | 0.204 <sup>*</sup> |
| SIF - ELMO    | 0.193  |
| SIF-FLAIR     | 0.201  |
| SIF-BERT      | 0.184  |
| SIF-BERT+ELMO<sup>✞</sup>  |0.181 <sup>✞</sup> |

#### Word Moving distance benchmark

| Model | RMSE |
| ------------- | ------------- |
| WMD-W2V       | 0.205  |
| WMD-ELMO      | 0.220  |
| WMD-FLAIR     | 0.216  |
| WMD-BERT      | 0.214  |
| WMD-BERT+ELMO  | 0.218  |
| WMD-W2V-STOP  | 0.215  |
| WMD-ELM0-STOP | 0.238  |
| WMD-FLAIR-STOP| 0.224  |
| WMD-BERT-STOP | 0.217  |
| WMD-BERT+ELMO-STOP  | 0.228  |

### Conclusions
Even though the contextual embeddings didn't improve  word average and moving distance benchmarks, it improved the smooth inverse frequency benchmark significantly. Best results were provided when BERT and ELMO were stacked together.<sup>✞</sup> denotes the best result and <sup>*</sup> denotes the current best benchmark.

### References
If you find this code useful in your research, please consider citing:

>
> @inproceedings{ranashinghe2019enhancing,    
    title={Enhancing unsupervised sentence similarity methods with deep contextualised word representations},    
    author={Ranashinghe, Tharindu and Orasan, Constantin and Mitkov, Ruslan},    
    booktitle={Proceedings of the Recent Advances in Natural Language Processing (RANLP)},    
    year={2019} 
  }\
  