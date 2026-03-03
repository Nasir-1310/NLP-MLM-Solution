# এটা run করো locally বা Colab এ
# শুধু file paths দাও

with open('allowed_tokens.txt') as f:
    allowed = [line.strip() for line in f]
print(f"Total allowed tokens: {len(allowed)}")

with open('train.txt') as f:
    lines = f.readlines()
print(f"Total train lines: {len(lines)}")

# Sentence length stats
lengths = [len(line.split()) for line in lines]
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Avg length: {sum(lengths)/len(lengths):.1f}")

# Unique tokens in train
all_tokens = set()
for line in lines:
    all_tokens.update(line.split())
print(f"Unique tokens in train: {len(all_tokens)}")