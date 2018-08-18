# to see a word indicates postive/negative sentiment
from collections import Counter
import numpy as np
# 1. count words
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
for review, label in zip(reviews, labels):
    target_counts = positive_counts if label == 'POSITIVE' else negative_counts
    for w in review.split(' '):
        target_counts[w] += 1
        total_counts[w] += 1
# 2. compute positive/negative ratio
pos_neg_ratios = Counter()
for w in total_counts:
    pos_neg_ratios[w] = positive_counts[w] / (negative_counts[w] + 1)
# 3. now we have positive words with large value, negative words with small value
#    but it's hard to directly compare two numbers and see if one word conveys the
#    same magnitude of positive sentiment as another word conveys negative sentiment.
#    to fix it, we convert the ratios to logarithms
for w in pos_neg_ratios:
    pos_neg_ratios[w] = np.log(pos_neg_ratios[w])
