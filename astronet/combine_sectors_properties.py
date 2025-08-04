import pandas as pd
import re
sectors = [85, 86, 87]
properties_dfs = []
vetter_labels = []

def to_snake_case(name):
    # Replace non-alphanumeric characters (space, dash, etc.) with underscore
    name = re.sub(r'[^0-9a-zA-Z]+', '_', name)
    # Insert underscore before camelCase transitions
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower().strip('_')

for sector in sectors:
    properties_path = f"/pdo/astronet-data/data/properties/tces-sector{sector}_with_labels.csv"
    properties = pd.read_csv(properties_path)
    if 'Unnamed: 0' in properties.columns:
        properties = properties.drop(columns=['Unnamed: 0'])
    properties.columns = [to_snake_case(col) for col in properties.columns]
    properties["sector"] = sector
    properties = properties.drop_duplicates()
    properties_dfs.append(properties)

properties_df = pd.concat(properties_dfs)
properties_df['tic_id'] = properties_df['tic_id'].astype(int)
properties_df = properties_df.rename(columns={
    'per': 'period',
    'epoc': 'epoch',
    'dur': 'duration',
    'centroid_distance_arc_sec': 'centroid_distance_arcsec'
})
properties_df = properties_df.reset_index(drop=True)
properties_df.to_csv("/pdo/users/dimond/sectors_85_to_87_properties.csv", index=False)

is_planet = properties_df['true_label'] == 'p'
filtered = properties_df[is_planet][['tic_id', 'planetno']]
filtered.to_csv('/pdo/users/dimond/sectors_85_to_87_candidates.ls', index=False, header=False, sep=' ')
