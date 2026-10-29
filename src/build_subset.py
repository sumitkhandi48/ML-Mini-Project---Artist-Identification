import pandas as pd

paper = ["Albrecht Durer", "Boris Kustodiev", "Camille Corot", "Camille Pissarro",
         "Childe Hassam", "Claude Monet", "Edgar Degas", "Eugene Boudin",
         "Giovanni Battista Piranesi", "Gustave Dore", "Henri Matisse",
         "Ilya Repin", "Ivan Aivazovsky", "Ivan Shishkin", "John Singer Sargent"]

df = pd.read_csv("data/all_data_info.csv")
sub = df[df["artist"].isin(paper)][["artist", "new_filename", "in_train", "style", "title"]].copy()

print("Rows:", len(sub))
print("Missing filenames:", sub["new_filename"].isna().sum())
print("Duplicate filenames:", sub["new_filename"].duplicated().sum())
print("\nin_train counts:")
print(sub["in_train"].value_counts())

ti = pd.read_csv("data/train_info.csv")
sub["in_train_info"] = sub["new_filename"].isin(ti["filename"])
print("\nFound in your train_info.csv:")
print(sub["in_train_info"].value_counts())

sub.to_csv("data/paper_subset.csv", index=False)
print("\nSaved data/paper_subset.csv")