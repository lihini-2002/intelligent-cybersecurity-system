import pandas as pd

# Load your full dataset
df = pd.read_csv("phishing_urls_combined_cleaned.csv")

# Set number of parts
parts = 3
chunk_size = len(df) // parts

# Save each chunk
for i in range(parts):
    start = i * chunk_size
    end = None if i == parts - 1 else (i + 1) * chunk_size
    chunk = df.iloc[start:end]
    chunk.to_csv(f"phishing_urls_part{i+1}.csv", index=False)

print("Done splitting CSV into", parts, "parts.")
