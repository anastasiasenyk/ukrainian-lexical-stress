import json
import pandas as pd
import os

parent_dir = "."

jsonl_file = os.path.join(parent_dir, "voa_clean.jsonl")
csv_file = os.path.join(parent_dir, "voa_clean.csv")

# Read JSONL and convert to DataFrame
data = []
with open(jsonl_file, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df = df[df['text'].apply(lambda x: len(x.split(" ")) > 2)]
df.to_csv(csv_file, index=False, encoding="utf-8")




