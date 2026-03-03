from collections import Counter, defaultdict
import numpy as np

with open('train.txt') as f:
    train_sentences = [line.strip().split() for line in f if line.strip()]
with open('allowed_tokens.txt') as f:
    allowed_tokens = set(line.strip() for line in f if line.strip())
with open('test_masked.txt') as f:
    test_masked = [line.strip().split() for line in f if line.strip()]
with open('test_labels.txt') as f:
    test_labels = [line.strip() for line in f if line.strip()]

# Bigram coverage on all test data
bigrams = defaultdict(Counter)
for s in train_sentences:
    for i in range(len(s) - 1):
        bigrams[s[i]][s[i+1]] += 1
        bigrams[s[i+1]][s[i]] += 1

bigram_hits = 0
for i, s in enumerate(test_masked):
    mask_idx = s.index('MASK')
    label = test_labels[i]
    found = False
    if mask_idx > 0 and s[mask_idx-1] in bigrams:
        if label in bigrams[s[mask_idx-1]]:
            found = True
    if mask_idx < len(s)-1 and s[mask_idx+1] in bigrams:
        if label in bigrams[s[mask_idx+1]]:
            found = True
    if found:
        bigram_hits += 1
print(f'Bigram coverage (ALL test): {bigram_hits}/2000 = {bigram_hits/2000:.3f}')

# Trigram coverage
trigrams_fwd = defaultdict(Counter)
trigrams_bwd = defaultdict(Counter)
for s in train_sentences:
    for i in range(2, len(s)):
        trigrams_fwd[(s[i-2], s[i-1])][s[i]] += 1
    for i in range(len(s)-2):
        trigrams_bwd[(s[i+1], s[i+2])][s[i]] += 1

tri_hits = 0
for i, s in enumerate(test_masked):
    mask_idx = s.index('MASK')
    label = test_labels[i]
    found = False
    if mask_idx >= 2:
        key = (s[mask_idx-2], s[mask_idx-1])
        if key in trigrams_fwd and label in trigrams_fwd[key]:
            found = True
    if mask_idx < len(s)-2:
        key = (s[mask_idx+1], s[mask_idx+2])
        if key in trigrams_bwd and label in trigrams_bwd[key]:
            found = True
    if found:
        tri_hits += 1
print(f'Trigram coverage (ALL test): {tri_hits}/2000 = {tri_hits/2000:.3f}')

# Max seq length
all_lens = [len(s) for s in train_sentences] + [len(s) for s in test_masked]
print(f'Max seq len: {max(all_lens)}, P99: {np.percentile(all_lens, 99):.0f}')
