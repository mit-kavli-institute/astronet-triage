from astroquery import mast
import pandas as pd

def update_ext_mast_data(tces):
    ext_data_file = '../mnt/tess/labels/ext_mast_data.csv'
    ext_table = pd.read_csv(ext_data_file, header=0).set_index('tic_id')
    
    addl_data = []
    for tic in tces.index.values:
        print(tic, end='')
        if tic in ext_table.index.values:
            print(' skiped')
            continue
        catalog_data = mast.Catalogs.query_object(f'TIC {tic}', catalog="TIC", radius='0.1s')
        print(' loaded')
        row_i = None
        for i, entry in enumerate(catalog_data['ID']):
            if entry == str(tic):
                row_i = i
                break
        assert row_i is not None, (tic, catalog_data)
        addl_data.append({'tic_id': tic, 'objType': catalog_data['objType'][row_i].item()})
        if len(addl_data) % 20 == 0:
            print(f'Updated {len(addl_data)} records')
            ext_table = ext_table.append(pd.DataFrame(addl_data).set_index('tic_id'))
            ext_table.to_csv(ext_data_file)
            addl_data = []

tcenorth = pd.read_csv('../mnt/tess/labels/north_tce_instar.csv', header=0).set_index('tic_id')
update_ext_mast_data(tcenorth)