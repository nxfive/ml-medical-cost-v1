import pandas as pd
import kagglehub
import shutil
from pathlib import Path


def main():
    cache_path = kagglehub.dataset_download('mirichoi0218/insurance')
    print(cache_path)
    path = shutil.copytree(cache_path, 'data/raw', dirs_exist_ok=True)
    print(path)

    csv_file = next(Path(path).rglob('*.csv'), None)
    df = pd.read_csv(csv_file)

    parquet_file_path = path + '/insurance.parquet'
    df.to_parquet(parquet_file_path, engine='pyarrow')

    csv_file.unlink()


if __name__ == '__main__':
    main()