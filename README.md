# Self-Attentive Sentence Embedding

This is a PyTorch implementation of [A structured self-attentive sentence embedding](https://arxiv.org/pdf/1703.03130.pdf) by Lin et al 2017. This approach has been applied to Author profiling [PAN 2015](https://pan.webis.de/clef15/pan15-web/author-profiling.html) and [2016](https://pan.webis.de/clef16/pan16-web/author-profiling.html) tasks. The data can be obtained from the above links. This implementations handles gender and age group classification.

The approach uses [100-dimensional Glove word embeddings](https://nlp.stanford.edu/projects/glove/) to initialize the word embedding layer. 

The program can be executed by

python main.py --input ./data --expt self-attn-gender --attr gender


Parameters:

--input     - Input path with 

--results   - Directory to store models and results

--expt      - Experiment name

--wordemb   - Word embeddings (100-dim Glove embeddings)

--batchsz   - Batch size

--nepoch    - Number of epochs

--embedsz   - Word embedding size

--hiddensz  - Hidden layer size

--nlayers   - Number of hidden layers

--attnsz    - Number of attention units (d_a)

--attnhops  - Number of attention hops (r)

--fcsize    - Fully connected layer size

--attr      - Attribute to profile (gender or age group)

--lr        - Learning rate

## Salient features
Features that were found salient by the attention layer for different social groups

### Female
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Female")

### Male
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Male")

### Ages 18-24
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Ages 18-24")

### Ages 50+
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Ages 50+")



Reference:
1) Lin, Z., Feng, M., Santos, C. N. D., Yu, M., Xiang, B., Zhou, B., & Bengio, Y. (2017). A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130.
