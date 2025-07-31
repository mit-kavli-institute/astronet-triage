import pandas as pd
import re
sectors = [85, 86, 87]
properties_dfs = []
vetter_labels = []


def to_snake_case(name):
    # Insert underscore before capital letters, except the first one
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Handle acronyms and multiple caps (e.g., 'SMass' → 's_mass')
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()

for sector in sectors:
    properties_path = f"/pdo/astronet-data/data/tfrecords/sector-{sector}/tces-sector{sector}_with_labels.csv"
    properties = pd.read_csv(properties_path)
    if 'Unnamed: 0' in properties.columns:
        properties = properties.drop(columns=['Unnamed: 0'])
    properties.columns = [to_snake_case(col) for col in properties.columns]
    properties = properties.drop_duplicates()
    properties_dfs.append(properties)
    is_planet = properties['true_label'] == 'p'
    filtered = properties[is_planet]
    print(len(filtered))
    # #filtered = filtered.drop_duplicates()
    # print(len(filtered))

properties_df = pd.concat(properties_dfs)
properties_df = properties_df.reset_index(drop=True)
properties_df.to_csv("/pdo/users/dimond/sectors_85_to_87_properties.csv", index=False)

# is_planet = properties_df['true_label'] == 'p'
# filtered = properties_df[is_planet][['tic_id', 'planetno']]

# print(len(filtered))
# 1/0