#this code was used to compress the dataset to 3gb CIC_IoT2023 dataset.
import pandas as pd
import glob
import os
import shutil
import time

source_path = "/Users/hussain/code/archive/"
save_path = "/Users/hussain/code/"

files = sorted(glob.glob(os.path.join(source_path, "part-*.csv")))
total_files = len(files)

print(f"\nFound {total_files} dataset parts.\n")
if total_files == 0:
    raise FileNotFoundError("No CSV parts found.")

samples = []
start_time = time.time()

for i, f in enumerate(files, start=1):
    percent = (i / total_files) * 100
    print(f"[{i}/{total_files}] ({percent:.1f}%) Processing {os.path.basename(f)}")

    df = pd.read_csv(f)
    sample = df.sample(frac=0.25, random_state=42)
    samples.append(sample)

data_small = pd.concat(samples, ignore_index=True)
print("\nCombined dataset shape:", data_small.shape)

data_small = data_small.drop_duplicates()
data_small = data_small.replace([float("inf"), float("-inf")], pd.NA)
data_small = data_small.dropna()

out_file = os.path.join(save_path, "CIC_IoT2023_3GB.csv")
data_small.to_csv(out_file, index=False)

print(f"\nSaved compressed dataset to: {out_file}")
print(f"\nTotal runtime: {(time.time() - start_time) / 60:.2f} minutes")

delete_original = input("\nDelete original archive folder? (y/n): ").lower()
if delete_original == "y":
    shutil.rmtree(source_path)
    print("Original archive folder deleted.")
else:
    print("Original archive folder kept.")
