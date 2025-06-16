import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc
import streamlit as st


from data_management.type_mapping import PREDICTION_LABELS, PREDICTION_MAPPING, TRUE_MAPPING
REQUIRED_MODEL_COLUMNS = {"astro_id", "model_no", "disp_p", "disp_e", "disp_n", "disp_j"}

data_manager = st.session_state.data_manager

class EvalUtils:
    def __init__(self, model_results: pd.DataFrame):

        self.properties = data_manager.get_data_frame()

        self.model_results = model_results # (astro_id, model_no, disp_p, disp_e, disp_n, disp_j)
        # drop rows where the true label does not exist
        self.model_results = self.model_results.dropna(subset=["true_label"])

        self.aggregated_model_results = self.aggregate_results()

    def _compute_ensemble_results(self, thresholds=None):
        """
        Computes ensemble results by averaging model predictions per `astro_id`.
        
        Parameters:
        - thresholds (dict, optional): A dictionary mapping each class probability column
        to a threshold value. Example: {"disp_p": 0.6, "disp_e": 0.5, "disp_n": 0.7, "disp_j": 0.5}
        
        Returns:
        - DataFrame with `astro_id`, averaged probabilities, and predicted labels.
        """
        probability_cols = ["disp_p", "disp_e", "disp_n", "disp_j"]
        ensemble = self.model_results.groupby(["astro_id", "true_label"])[probability_cols].mean().reset_index()
        if thresholds:
            # Apply thresholds: Assign the first class that meets its threshold
            def apply_thresholds(row):
                for class_col, threshold in thresholds.items():
                    if row[class_col] >= threshold:
                        return class_col
                return "disp_j"  # Default if no threshold is met

            ensemble["predicted_label"] = ensemble.apply(apply_thresholds, axis=1)
        else:
            # Default behavior: Use max probability
            ensemble["predicted_label"] = ensemble[probability_cols].idxmax(axis=1)
        return ensemble

    def get_model_results(self, include_labels=False, include_properties=False, dropna=True):
        """Returns model results with optional true labels and properties."""
        df = self.model_results.copy()
        if include_labels:
            df = df.merge(self.properties[['astro_id', 'label']], on="astro_id", how="left", suffixes=("", "_true"))
            df.rename(columns={"label": "true_label"}, inplace=True)
        if include_properties:
            df = df.merge(self.properties, on="astro_id", how="left", suffixes=("", "_true"))
        if dropna:
            # print unique true label values
            df.dropna(subset=["true_label"], inplace=True)
        
        desired_start = ["astro_id", "predicted_label", "true_label"]
        remaining_columns = [col for col in df.columns if col not in desired_start]
        df = df.reindex(columns=desired_start + remaining_columns)
        return df

    def get_ensemble_results(self, thresholds, include_labels=False, include_properties=False, dropna=True, human_readable_names=True):
        """Returns ensemble results with optional true labels and properties."""
        self.ensemble_results = self._compute_ensemble_results(thresholds)
        df = self.ensemble_results.copy()
        if include_labels and 'true_label' not in df.columns:
            df = df.merge(self.properties[['astro_id', 'label']], on="astro_id", how="left", suffixes=("", "_true"))
            df.rename(columns={"label": "true_label"}, inplace=True)
        if include_properties:
            df = df.merge(self.properties, on="astro_id", how="left", suffixes=("", "_true"))
        if dropna:
            df.dropna(subset=["true_label"], inplace=True)
        
        desired_start = ["astro_id", "predicted_label", "true_label"]
        remaining_columns = [col for col in df.columns if col not in desired_start]
        df = df.reindex(columns=desired_start + remaining_columns)

        if human_readable_names:
            df["predicted_label"] = df["predicted_label"].map(PREDICTION_MAPPING)
            df['true_label'] = df['true_label'].map({"p": "Planet", "j": "Junk"})
        return df

    def aggregate_results(self):
        required_columns = {"astro_id", "disp_p", "disp_e", "disp_n", "disp_j"}
        if not required_columns.issubset(self.model_results.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(self.model_results.columns)}")

        def get_mean_pred_label(sub_df):
            argmax_labels = sub_df[PREDICTION_LABELS].idxmax(axis=1)
            mean_pred = argmax_labels.mode()[0]  
            return PREDICTION_MAPPING[mean_pred]  

        result_df = self.model_results.groupby(["astro_id", "true_label"]).apply(lambda x: pd.Series({
            "pred_label": get_mean_pred_label(x)
        })).reset_index()
        
        return result_df

    def compute_performance(self):
        """
        Computes the performance metrics for each model in the evaluation results.

        :return: DataFrame with model performance metrics.
        """
        model_results = self.model_results.copy()
        model_results["predicted_label"] = model_results[PREDICTION_LABELS].idxmax(axis=1)
        model_results["predicted_label"] = model_results["predicted_label"].map(PREDICTION_MAPPING)
        model_results['true_label'] = model_results['true_label'].map(TRUE_MAPPING)
        
        filtered_model_results = model_results.dropna(subset=["true_label", "predicted_label"], how="all")
        
        model_performance = []
        for model_no in filtered_model_results["model_no"].unique():
            model_df = filtered_model_results[filtered_model_results["model_no"] == model_no]
            valid_model_df = model_df.dropna(subset=["true_label", "predicted_label"])
            
            if not valid_model_df.empty:
                acc = accuracy_score(valid_model_df["true_label"], valid_model_df["predicted_label"])
                prec = precision_score(valid_model_df["true_label"], valid_model_df["predicted_label"], average="weighted", zero_division=0)
                rec = recall_score(valid_model_df["true_label"], valid_model_df["predicted_label"], average="weighted", zero_division=0)
                f1 = f1_score(valid_model_df["true_label"], valid_model_df["predicted_label"], average="weighted", zero_division=0)
                
                model_performance.append({
                    "model_no": model_no, 
                    "accuracy": acc, 
                    "precision": prec, 
                    "recall": rec, 
                    "f1_score": f1
                })
        return pd.DataFrame(model_performance)