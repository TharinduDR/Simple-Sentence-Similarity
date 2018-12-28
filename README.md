# Sentence Similarity
Sentence SImilarity can be used for many domains in natural language processing, including information retrieval, translation memories, question answering etc. There are many methods to calculate the similarity between two strings. In this project I have evaluated those methods using SICK and STS data sets. 

*Traditional approaches* can be found in [Traditional Sentence Similarity.ipynb](https://github.com/TharinduDR/Simple-Sentence-Similarity/blob/master/Sentence%20Similarity%20-%20Traditional.ipynb). Methods like edit distance and levenshtein distance have been explored here. 

*Word Vector approaches* can be found in [Sentence Similarity - Word Vectors.ipynb](https://github.com/TharinduDR/Simple-Sentence-Similarity/blob/master/Sentence%20Similarity%20-%20Word%20Vectors.ipynb). Glove, word2vec and fasttext word embedding models were used for experiements. Distance measures like cosine similarity, word moving distance, smooth inverse frequency were considered.

*Context Vector approaches* can be found in [Sentence Similarity - Context Vectors.ipynb](https://github.com/TharinduDR/Simple-Sentence-Similarity/blob/master/Sentence%20Similarity%20-%20Context%20Vectors.ipynb). The same benchmarks in the *Word Vector approaches* were considered using ELMo embeddings and compared the results with wordvec embeddings. 
