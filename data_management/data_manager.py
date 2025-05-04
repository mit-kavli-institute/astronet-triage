import re
from pathlib import Path
from config_parser import DatasetConfig
from dataclasses import dataclass
from data_management.astro_data import AstroData, Split
from data_management.google_sheets_reader import GoogleSheetsReader
from data_management.data_storage import DataStorage
import pandas as pd
import traceback

from data_management.type_mapping import HUMAN_LABEL_MAP, TRUE_MAPPING

dataset_config = DatasetConfig.from_yaml()

dtype_dict = {
    # Integer IDs (Nullable to handle missing values)
    "astro_id": "Int64",
    "tic_id": "Int64",

    # File Paths / Text Columns
    "fits_path": "string",
    "report_paths": "string",
    "filename": "string",
    "comment": "string",

    # Categorical/String Columns
    "label": "string",
    "label_simplified": "string",
    "split": "string",
    "final": "string",
    "decision": "string",
    "distinct": "string",
    "mk": "string",
    "ch": "string",
    "et": "string",
    "md": "string",
    "as": "string",
    "dm": "string",
    "tansu": "string",
    "shishir": "string",
    "jh": "string",
    "astronet_note": "string",
    "sectors": "string",

    # Numerical Values (Floats)
    "ra": "float64",
    "dec": "float64",
    "tmag": "float64",
    "epoc": "float64",
    "period": "float64",
    "duration": "float64",
    "transit_depth": "float64",
    "star_rad": "float64",
    "star_mass": "float64",
    "teff": "float64",
    "logg": "float64",
    "sn": "float64",
    "qingress": "float64",
    "star_rad_est": "float64",

    # Random Seed (Nullable Integer)
    "seed_randbetween(1,_100)": "string",
}

@dataclass
class AstroDataReport:
    num_successful_loads: int = 0
    num_fits_failed_to_load: int = 0
    num_labels_failed_to_load: int = 0
    num_properties_failed_to_load: int = 0

    def __str__(self):
        return (
            "AstroDataReport:\n"
            f"  - # Successful Load: {self.num_successful_loads}\n"
            f"  - FITS Files Failed to Load: {self.num_fits_failed_to_load}\n"
            f"  - Labels Failed to Load: {self.num_labels_failed_to_load}\n"
            f"  - Properties Failed to Load: {self.num_properties_failed_to_load}"
        )
    
class DataManagerConstants:
    TIC_ID_COLUMN = "tic_id"
    SPLIT_COLUMN = "split" # Comes from dataset split sheet
    LABEL_COLUMN = "final" # Comes from labels sheet
    ASTRO_ID_COLUMN = "astro_id" # Comes from all sheets (primary key)
    DISTINCT_COLUMN = "distinct" # Comes from labels sheet
    ASTRONET_NOTE_COLUMN = "astronet_note" # Comes from labels sheet (human notes about the labels)

def to_camel_case(s: str):
    return s.lower().replace(' ', '_')

def is_url(string):
    url_pattern = re.compile(
        r'^(https?://)?'              # optional http:// or https://
        r'([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  # domain
        r'(:\d+)?'                    # optional port
        r'(/[^\s]*)?$'                # optional path
    )
    return bool(url_pattern.match(string))

class DataManager:
    """
    Links all data sources together into AstroData objects.

    Utilizes the Google sheets + storage directory. Rows are loaded based on
    the test/train/validation split sheet, so that no other data is loaded if it
    is not needed.
    """

    def __init__(self):
        print(f'Loading dataset from dataset_config:\n{dataset_config}\n')
        self.data_dir = dataset_config.raw_data_dir
        self.data_storage = DataStorage(data_dir=self.data_dir, images_dir=dataset_config.images_dir, reports_dir=dataset_config.reports_dir)

        # Read sheets
        sheets_reader = GoogleSheetsReader()
        self.sheets_reader = sheets_reader
        properties_sheet = dataset_config.properties_sheet
        if is_url(properties_sheet):
            self.properties_df = sheets_reader.from_url(properties_sheet)
        else:
            self.properties_df = pd.read_csv(properties_sheet)
        self.properties_df = self.convert_columns_to_snake_case(self.properties_df)

        if 'label' not in self.properties_df.columns:
            self.properties_df['label'] = self.properties_df['final'].str.lower().map(TRUE_MAPPING).map(HUMAN_LABEL_MAP)

        self.labels_df = pd.DataFrame()
        if dataset_config.labels_sheet:
            labels_sheet = dataset_config.labels_sheet
            self.labels_df = sheets_reader.from_url(labels_sheet)
            self.labels_df = self.convert_columns_to_snake_case(self.labels_df)
        self.dataset_split_df = pd.DataFrame()
        if dataset_config.dataset_split_sheet:
            dataset_split_sheet = dataset_config.dataset_split_sheet
            self.dataset_split_df = sheets_reader.from_url(dataset_split_sheet)
            self.dataset_split_df = self.convert_columns_to_snake_case(self.dataset_split_df)

        # Init data
        (self.astro_data, report) = self._init_astro_data()
        self.tic_id_to_data = {}
        for data in self.astro_data:
            self.tic_id_to_data[data.tic_id] = data
        print(str(report) + '\n')

    def _init_astro_data(self) -> tuple[list[AstroData], AstroDataReport]:
        astro_data = []
        astro_data_report = AstroDataReport()
        # Load data based on the test/train/validation sheet
        for index, row in self.properties_df.iterrows():
            try:
                tic_id = int(row[DataManagerConstants.TIC_ID_COLUMN])
                astro_id = int(row[DataManagerConstants.ASTRO_ID_COLUMN])

                split = Split.UNALLOCATED
                try:
                    # join the split from the dataset_split_df
                    split_row = self.dataset_split_df[self.dataset_split_df[DataManagerConstants.TIC_ID_COLUMN] == tic_id]
                    split = Split.from_str(split_row[DataManagerConstants.SPLIT_COLUMN])
                except Exception:
                    pass

                label = None
                if DataManagerConstants.LABEL_COLUMN not in self.properties_df.columns: # final contains human label
                    # load label from labels_df
                    labels_row = self.labels_df[self.labels_df[DataManagerConstants.TIC_ID_COLUMN] == tic_id]
                    label = labels_row.to_dict(orient='records')[0] if not labels_row.empty else {}
                else:
                    label = row[DataManagerConstants.LABEL_COLUMN]
                if label is None:
                    astro_data_report.num_labels_failed_to_load += 1

                properties_row = self.properties_df[self.properties_df[DataManagerConstants.TIC_ID_COLUMN] == tic_id]
                properties_dict = (
                    {to_camel_case(k): v for k, v in properties_row.to_dict(orient='records')[0].items()}
                    if not properties_row.empty else {}
                )

                try:
                    labels_row = self.labels_df[self.labels_df[DataManagerConstants.TIC_ID_COLUMN] == tic_id]
                    labels_dict = labels_row.to_dict(orient='records')[0] if not labels_row.empty else {}
                    properties_dict.update(
                        {
                            'distinct': labels_dict.get(DataManagerConstants.DISTINCT_COLUMN),
                            'astronet_note': labels_dict.get(DataManagerConstants.ASTRONET_NOTE_COLUMN)
                        }
                    )
                except Exception:
                    pass

                fits_path = self.data_storage.get_path(tic_id=tic_id)
                if not fits_path:
                    astro_data_report.num_fits_failed_to_load += 1
                    #continue
                
                images_path = self.data_storage.get_images_path(tic_id=tic_id)
                report_paths = self.data_storage.get_reports_path(tic_id=tic_id)



                astro_data.append(AstroData(
                    astro_id=astro_id,
                    tic_id=tic_id,
                    fits_path=fits_path,
                    images_path=images_path,
                    report_paths=report_paths,
                    properties=properties_dict,
                    label=label,
                    split=split.value,
                ))
                astro_data_report.num_successful_loads += 1
            except Exception as e:
                print(f'Failed to load index={index} with exception ' + str(e) + ', skipping...')
                traceback.print_exc()

        report = AstroDataReport(
            num_successful_loads=astro_data_report.num_successful_loads,
            num_fits_failed_to_load=astro_data_report.num_fits_failed_to_load,
            num_labels_failed_to_load=astro_data_report.num_labels_failed_to_load,
            num_properties_failed_to_load=astro_data_report.num_properties_failed_to_load,
        )
        return (astro_data, report)

    def to_snake_case(self, s):
        s = re.sub(r'\s+', '_', s)  # Replace spaces with underscores
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)  # Insert underscore before uppercase letters preceded by lowercase or number
        s = re.sub(r'[^a-zA-Z0-9_]', '', s)  # Remove non-alphanumeric characters except underscores
        return s.lower().strip('_')  # Convert to lowercase and remove leading/trailing underscores
    
    def convert_columns_to_snake_case(self, df):
        df.columns = [self.to_snake_case(col) for col in df.columns]
        return df

    def get_data_from_tic_id(self, tic_id: int) -> AstroData:
        data = self.tic_id_to_data.get(tic_id)
        if data is not None:
            raise Exception(f"Data not found for tic id {tic_id}")
        return data

    def get_data_frame(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all AstroData objects with properties unpacked.
        """
        TRUE_MAPPING = {
            "eb": "Eclipsing Binary", "ebs": "Eclipsing Binary", "et": "Eclipsing Binary", "eu": "Eclipsing Binary",
            "ets": "Eclipsing Binary", "eus": "Eclipsing Binary", "pt": "Planet", "pb": "Planet", "pu": "Planet",
            "pts": "Planet", "pus": "Planet", "nt": "Noise", "nb": "Noise", "nu": "Noise", "jj": "Junk",
            "ub": "Unknown", "i": "Indeterminate"
        }
        data_list = []
        for data in self.astro_data:
            label_simplified = TRUE_MAPPING.get(data.label)
            data_dict = {
                'astro_id': data.astro_id,
                'tic_id': data.tic_id,
                'fits_path': data.fits_path,
                'report_paths': data.report_paths,
                'label': data.label,
                'label_simplified': label_simplified,
                'split': data.split,
            }
            data_dict.update(data.properties)
            data_list.append(data_dict)
        
        df = pd.DataFrame(data_list)
        filtered_dtype_dict = {col: dtype for col, dtype in dtype_dict.items() if col in df.columns}
        df = df.astype(filtered_dtype_dict)
        return df

data_manager = DataManager() # singleton

# Example usage:
if __name__ == "__main__":
    tic_id_example = 100100823
    print(data_manager.tic_id_to_data[tic_id_example])
