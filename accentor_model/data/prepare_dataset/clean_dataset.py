import re
import pandas as pd
from Levenshtein import ratio


def clean_text(sentence):
    words = re.split(r'[ ,.?!;:«»()[\]-]+', sentence.lower())
    pattern = r'^[^абвгґдеєжзиіїйклмнопрстуфхцчшщьюя]+|[^абвгґдеєжзиіїйклмнопрстуфхцчшщьюя]+$'
    return " ".join(re.sub(pattern, '', word) for word in words if word)


df = pd.read_csv(f'voa_transcribed.csv')
print("Before dropping duplicates: ", df.shape)
df = df.drop_duplicates(subset='file', keep='first')
print("After dropping duplicates: ", df.shape)

# Apply text cleaning
df['cleaned_text'] = df['text'].astype(str).apply(clean_text)
df['cleaned_transcription'] = df['transcription_ukrainian'].astype(str).apply(clean_text)

# Calculate the percentage of matching rows
matching_rows = (df['cleaned_text'] == df['cleaned_transcription']).sum()
total_rows = len(df)
percentage_matching = (matching_rows / total_rows) * 100

# Calculate similarity using Levenshtein ratio
df['similarity'] = df.apply(lambda row: ratio(row['cleaned_text'], row['cleaned_transcription']), axis=1)
avg_similarity = df['similarity'].mean() * 100

print(f"Number of rows: {df.shape[0]}")
print(f'Percentage of exact matches: {percentage_matching:.2f}%')
print(f'Average text similarity: {avg_similarity:.2f}%\n')

thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 0.92, 0.95, 0.96, 0.97, 0.98, 0.99]
for threshold in thresholds:
    count_above_threshold = (df['similarity'] >= threshold).sum()
    percentage_above_threshold = (count_above_threshold / total_rows) * 100
    print(f'Rows with similarity >= {threshold * 100:.0f}%: {count_above_threshold} ({percentage_above_threshold:.2f}%)')

# Select random 20 rows with 90% <= similarity < 100% and print cleaned versions

lower_bound = 0.0
higher_bound = 0.9
similar_rows = df[(df['similarity'] >= lower_bound) & (df['similarity'] < higher_bound)]
similar_rows = similar_rows.sample(n=min(20, len(similar_rows)))

print(f"Random 20 rows with similarity between {lower_bound*100}% and {higher_bound*100}%:")
for index, row in similar_rows.iterrows():
    print(f"Similarity: {row['similarity']}  --- Filename: {row['file']}")
    print(f"Cleaned Text: {row['cleaned_text']}")
    print(f"Cleaned Transcription: {row['cleaned_transcription']}")
    print("-")


# Get unique symbols in cleaned_transcription
all_chars = set("".join(df['cleaned_transcription']))
ukrainian_chars = set("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя")

# Print symbols not in Ukrainian alphabet
non_ukrainian_chars = all_chars - ukrainian_chars
print("Symbols in cleaned_transcription that are not in the Ukrainian alphabet:", "".join(non_ukrainian_chars))

# Print missing Ukrainian symbols
missing_ukrainian_chars = ukrainian_chars - all_chars
print("Ukrainian alphabet symbols missing in cleaned_transcription:", "".join(missing_ukrainian_chars))


df = df[(df['similarity'] > 0.9)]
print("> 90% similarity : ", df.shape)
df = df.drop(columns=['cleaned_text', 'cleaned_transcription'])

n_rows = df.shape[0] / 1000
df.to_csv(f"voa_transcribed_{n_rows:.0f}_rows.csv", index=False)