import os
import pandas as pd

def find_csv_files(data_dir):
    """Find all relevant CSV files in the data directory."""
    csv_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def merge_csv_files(data_dir, output_file):
    """Merge all CSV files into one dataframe and save."""
    csv_files = find_csv_files(data_dir)

    if not csv_files:
        print(f"❌ Error: No CSV files found in {data_dir}!")
        return

    print(f"✅ Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(" -", f)

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")

    if not dfs:
        print("❌ Error: No valid CSV files to merge.")
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Merged dataset saved to {output_file} with {len(merged_df)} rows.")


if __name__ == "__main__":
    data_dir = "data"
    output_file = "data/merged_dataset.csv"
    merge_csv_files(data_dir, output_file)
