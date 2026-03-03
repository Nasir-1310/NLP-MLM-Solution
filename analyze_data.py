# Analyze dataset statistics
with open('train.txt') as f:
    train_lines = f.readlines()
with open('allowed_tokens.txt') as f:
    allowed = set(line.strip() for line in f.readlines())
with open('test_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]

print(f'Training sentences: {len(train_lines)}')
print(f'Allowed tokens: {len(allowed)}')
print(f'Sample token length: {len(labels[0])} hex chars')

# Get vocabulary from training
all_tokens = set()
seq_lengths = []
for line in train_lines:
    tokens = line.strip().split()
    seq_lengths.append(len(tokens))
    all_tokens.update(tokens)

print(f'Unique tokens in train: {len(all_tokens)}')
print(f'Avg sequence length: {sum(seq_lengths)/len(seq_lengths):.1f}')
print(f'Min/Max seq length: {min(seq_lengths)}/{max(seq_lengths)}')
print(f'Labels in allowed: {sum(1 for l in labels if l in allowed)}/{len(labels)}')

# Check token overlap
train_allowed_overlap = len(all_tokens & allowed)
print(f'Allowed tokens in train data: {train_allowed_overlap}/{len(allowed)}')
