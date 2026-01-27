import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
print(f"Total records: {len(df)}")
print(f"Unique records: {len(df.drop_duplicates())}")