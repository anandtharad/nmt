import pandas as pd

splits = {
    'hin_Deva': ['bpcc-seed-v1/hin_Deva.tsv', 'bpcc-seed-latest/hin_Deva.tsv'],
}

def load_and_merge_splits(dataset_name: str, paths: list, base_path: str = "hf://datasets/ai4bharat/BPCC/") -> pd.DataFrame:
    dfs = []
    for path in paths:
        full_path = base_path + path
        df = pd.read_csv(full_path, sep="\t", usecols=['src', 'tgt'])
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return combined_df

for lang, paths in splits.items():
    df = load_and_merge_splits(lang, paths)
print(df)

print('writing')
with open('data/en_files/source','w') as source:
    for line in df['src'].tolist():
        source.write(line.strip()+"\n")

with open('data/hi_files/target','w') as target:
    for line in df['tgt'].tolist():
        target.write(line.strip()+"\n")