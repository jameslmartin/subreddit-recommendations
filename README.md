### Goal
Train a model that classifies very unstructured text like Reddit comments to see if a recommendation engine could be built  
If subreddits cluster together, nearest clusters may be other subreddits that are "related" or "interesting" to a user.

### Steps:
- Train many different models with varying parameters on as large of a set of reddit comments as possible, and answer:
    - Why am I tuning parameters the way I am? How does each model differ? What are the limitations of using skip-grams / word2vec?
    - Would doc2vec work better? Would GloVe work better?
- Evaluate each by calculating the misclassification rate of subreddit label
    - Visualize best model through dimensionality reduction
    - Use K-means clustering to see “related” subreddits, even if clusters are wrong based on labeling
- Train a model using BookCorpus and see how well that model classifies comments
    - Perform same visualization and compare
- Use LDA for topic discovery of acquired clusters
    - See what sort of topics each cluster contains and compare results
    - See what is “most likely” subreddit of each cluster and compare to ground truth
    
#### Contact
jamesml@cs.unc.edu

#### Acknowledgements
Kaggle for uploading the data set and great tutorials on Python's gensim and sklearn libraries
