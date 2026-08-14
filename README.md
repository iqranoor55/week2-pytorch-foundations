# Week 2 — PyTorch Foundations & ML Mathematics

Transitioning from NumPy into PyTorch while building the 
mathematical foundations that power modern machine learning.

## What is in this repository

**Day 1 — NumPy to PyTorch Transition**
Rebuilt Week 1 statistical operations (mean, std, correlation, 
outlier detection) as a PyTorch class. Benchmarked NumPy vs 
PyTorch on 1 million rows to understand performance differences.

**Day 3 — Naive Bayes Spam Classifier from Scratch**
Built a complete Naive Bayes classifier in pure Python and NumPy.
Implemented vocabulary building, word count matrix, Laplace 
smoothing, log priors and log likelihoods from first principles.
Tested on real SMS spam dataset. Accuracy: 98%

**Day 4 — Information Theory from Scratch**
Implemented entropy, cross-entropy and KL divergence manually.
Verified the fundamental theorem: H(P,Q) = H(P) + D_KL(P||Q)
This is the exact mathematics behind every neural network 
loss function.

## Concepts Covered
- PyTorch tensor operations and benchmarking
- Autograd: requires_grad, backward(), computation graphs
- Bayes theorem applied to spam classification
- Information theory: entropy, cross-entropy, KL divergence
- Probability and posterior computation

## DSA This Week
- Sliding Window pattern
- Solved 2 LeetCode problems on Sliding Window

## Tech Stack
Python • NumPy • PyTorch • Pandas • sickit learn

## Key Insight
Cross-entropy is not just a loss function name. It is the 
exact mathematical distance between what your model predicts 
and what reality actually is. Understanding this changes how 
you debug models.