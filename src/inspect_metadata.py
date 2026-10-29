import pandas as pd

df = pd.read_csv("data/all_data_info.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(3).to_string())

counts = df["artist"].value_counts()
big = counts[counts >= 450]
print("\nArtists with >=450 paintings:", len(big))
print(big.to_string())

keys = ["Durer", "Dürer", "Kustodiev", "Corot", "Pissarro", "Hassam",
        "Monet", "Degas", "Boudin", "Piranesi", "Dore", "Doré",
        "Matisse", "Repin", "Aivazovsky", "Shishkin", "Sargent"]
print("\nPaper artists search:")
for k in keys:
    m = counts[counts.index.str.contains(k, case=False, na=False)]
    print(k, "->", m.to_dict())