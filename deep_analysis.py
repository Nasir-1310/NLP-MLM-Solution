import numpy as np
from collections import Counter, defaultdict

with open('train.txt') as f:
    train_sentences = [line.strip().split() for line in f if line.strip()]
with open('allowed_tokens.txt') as f:
    allowed_tokens = set(line.strip() for line in f if line.strip())
with open('test_masked.txt') as f:
    test_masked = [line.strip().split() for line in f if line.strip()]
with open('test_labels.txt') as f:
    test_labels = [line.strip() for line in f if line.strip()]

# 1. Token frequency analysis
all_tokens = []
for s in train_sentences:
    all_tokens.extend(s)
freq = Counter(all_tokens)

allowed_freqs = [freq.get(t, 0) for t in allowed_tokens]
print(f"Allowed token freq - min: {min(allowed_freqs)}, max: {max(allowed_freqs)}, mean: {np.mean(allowed_freqs):.1f}, median: {np.median(allowed_freqs):.1f}")

# 2. Position analysis of MASK in test
mask_positions = []
mask_relative_positions = []
for s in test_masked:
    pos = s.index('MASK')
    mask_positions.append(pos)
    mask_relative_positions.append(pos / len(s))
print(f"\nMask position - min: {min(mask_positions)}, max: {max(mask_positions)}, mean: {np.mean(mask_positions):.1f}")
print(f"Mask relative pos - mean: {np.mean(mask_relative_positions):.3f}")

# 3. Co-occurrence strength
# For each test sentence, count how many context tokens appear together in training
context_match_counts = []
for s in test_masked:
    mask_idx = s.index('MASK')
    context = set(s) - {'MASK'}
    matches = 0
    for ts in train_sentences:
        ts_set = set(ts)
        if len(context & ts_set) >= 3:
            matches += 1
    context_match_counts.append(matches)
    if len(context_match_counts) <= 5:
        print(f"  Test {len(context_match_counts)}: {matches} matching train sentences (3+ shared tokens)")

print(f"\nContext match counts - mean: {np.mean(context_match_counts[:100]):.1f} (first 100 samples)")

# 4. Bigram coverage
bigrams = defaultdict(Counter)
for s in train_sentences:
    for i in range(len(s) - 1):
        bigrams[s[i]][s[i+1]] += 1
        bigrams[s[i+1]][s[i]] += 1

bigram_hits = 0
for i, s in enumerate(test_masked[:100]):
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

print(f"\nBigram coverage (first 100 test): {bigram_hits}/100 labels found via adjacent bigrams")

# 5. Label distribution
label_freq_in_train = Counter()
for l in test_labels:
    label_freq_in_train[l] = freq.get(l, 0)
print(f"\nTest label freq in train - min: {min(label_freq_in_train.values())}, max: {max(label_freq_in_train.values())}, mean: {np.mean(list(label_freq_in_train.values())):.1f}")

# 6. Seq length distribution
train_lens = [len(s) for s in train_sentences]
test_lens = [len(s) for s in test_masked]
print(f"\nTrain seq lengths - P25:{np.percentile(train_lens,25):.0f} P50:{np.percentile(train_lens,50):.0f} P75:{np.percentile(train_lens,75):.0f} P95:{np.percentile(train_lens,95):.0f}")
print(f"Test seq lengths  - P25:{np.percentile(test_lens,25):.0f} P50:{np.percentile(test_lens,50):.0f} P75:{np.percentile(test_lens,75):.0f} P95:{np.percentile(test_lens,95):.0f}")
