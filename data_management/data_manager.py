from pathlib import Path
from config_parser import DatasetConfig
from dataclasses import dataclass
from data_management.astro_data import AstroData
from data_management.google_sheets_reader import GoogleSheetsReader
from data_management.data_storage import DataStorage

dataset_config = DatasetConfig.from_yaml()

@dataclass
class AstroDataReport:
    num_successful_loads: int = 0
    num_fits_failed_to_load: int = 0
    num_labels_failed_to_load: int = 0
    num_properties_failed_to_load: int = 0

class DataManagerConstants:
    TIC_ID_COLUMN = "TIC ID"
    SPLIT_COLUMN = "Split" # Comes from dataset split sheet
    LABEL_COLUMN = "Final" # Comes from labels sheet
    ASTRO_ID_COLUMN = "Astro ID" # Comes from all sheets (primary key)
    DISTINCT_COLUMN = "Distinct" # Comes from labels sheet
    ASTRONET_NOTE_COLUMN = "Astronet note" # Comes from labels sheet (human notes about the labels)

@dataclass
class AstroDataReport:
    num_successful_loads: int
    num_fits_failed_to_load: int
    num_labels_failed_to_load: int
    num_properties_failed_to_load: int

    def __str__(self):
        return (
            "AstroDataReport:\n"
            f"  - # Successful Load: {self.num_successful_loads}\n"
            f"  - FITS Files Failed to Load: {self.num_fits_failed_to_load}\n"
            f"  - Labels Failed to Load: {self.num_labels_failed_to_load}\n"
            f"  - Properties Failed to Load: {self.num_properties_failed_to_load}"
        )

def to_camel_case(s: str):
    return s.lower().replace(' ', '_')

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
        self.data_storage = DataStorage(data_dir=self.data_dir)

        # Read sheets
        labels_sheet = dataset_config.labels_sheet
        sheets_reader = GoogleSheetsReader()
        self.labels_df = sheets_reader.from_url(labels_sheet)
        properties_sheet = dataset_config.properties_sheet
        self.properties_df = sheets_reader.from_url(properties_sheet)
        dataset_split_sheet = dataset_config.dataset_split_sheet
        self.dataset_split_df = sheets_reader.from_url(dataset_split_sheet)

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
        for index, row in self.dataset_split_df.iterrows():
            try:
                tic_id = int(row[DataManagerConstants.TIC_ID_COLUMN])
                split = row[DataManagerConstants.SPLIT_COLUMN]

                labels_row = self.labels_df[self.labels_df[DataManagerConstants.TIC_ID_COLUMN] == tic_id]
                labels_dict = labels_row.to_dict(orient='records')[0] if not labels_row.empty else {}
                if not labels_dict:
                    astro_data_report.num_labels_failed_to_load += 1
                    continue

                label = labels_dict[DataManagerConstants.LABEL_COLUMN]
                astro_id = labels_dict[DataManagerConstants.ASTRO_ID_COLUMN]

                properties_row = self.properties_df[self.properties_df[DataManagerConstants.TIC_ID_COLUMN] == tic_id]
                properties_dict = (
                    {to_camel_case(k): v for k, v in properties_row.to_dict(orient='records')[0].items()}
                    if not properties_row.empty else {}
                )

                properties_dict.update(
                    {
                        'distinct': labels_dict.get(DataManagerConstants.DISTINCT_COLUMN),
                        'astronet_note': labels_dict.get(DataManagerConstants.ASTRONET_NOTE_COLUMN)
                    }
                )

                fits_path = self.data_storage.get_path(tic_id=tic_id)
                if not fits_path:
                    astro_data_report.num_fits_failed_to_load += 1
                    continue


                astro_data.append(AstroData(
                    astro_id=astro_id,
                    tic_id=tic_id,
                    fits_path=fits_path,
                    report_path=None,
                    properties=properties_dict,
                    label=label,
                ))
                astro_data_report.num_successful_loads += 1
            except Exception as e:
                print(f'Failed to load tic_id={tic_id} with exception ' + str(e) + ', skipping...')

        report = AstroDataReport(
            num_successful_loads=astro_data_report.num_successful_loads,
            num_fits_failed_to_load=astro_data_report.num_fits_failed_to_load,
            num_labels_failed_to_load=astro_data_report.num_labels_failed_to_load,
            num_properties_failed_to_load=astro_data_report.num_properties_failed_to_load,
        )
        return (astro_data, report)

    def get_data_from_tic_id(self, tic_id: int) -> AstroData:
        data = self.tic_id_to_data.get(tic_id)
        if data is not None:
            raise Exception(f"Data not found for tic id {tic_id}")
        return data

data_manager = DataManager() # singleton

# Example usage:
if __name__ == "__main__":
    tic_id_example = 100100823
    print(data_manager.tic_id_to_data[tic_id_example])
