import pandas as pd

#def main():
    # files = [
    #     '/pdo/astronet-data/data/labels/labels_Group_Jun18.csv',
    #     '/pdo/astronet-data/data/labels/labels_Group_Jun25.csv',
    #     '/pdo/astronet-data/data/labels/labels_labels_Group_Jun24.csv'
    # ]

    # properties_file = '/pdo/astronet-data/data/labels/sector_86_qlp_properties_with_centroid_CORRECTED.csv'

    # dfs = [pd.read_csv(f) for f in files]
    # labels_df = pd.concat(dfs, ignore_index=True)
    # num_duplicates = labels_df.duplicated(subset="astro_id").sum()
    # print(f"Number of duplicate astro_id entries: {num_duplicates}")
    # duplicates = labels_df[labels_df.duplicated(subset="astro_id", keep=False)]
    # print(duplicates.sort_values("astro_id"))
    # labels_df = labels_df.drop_duplicates(subset="astro_id", keep="first")

    # # Load properties file
    # props_df = pd.read_csv(properties_file)

    # # Merge on shared identifier — likely 'astro_id' or 'TIC ID'
    # # Adjust the key below depending on your actual column names
    # # Merge on astro_id (left) and Astro ID (right)
    # merged_df = labels_df.merge(props_df, left_on="astro_id", right_on="Astro ID", how="left")

    # # Drop redundant 'astro_id' column if needed
    # merged_df = merged_df.drop(columns=["astro_id"])
    # merged_df = merged_df.rename(columns={"label_1": "true_label"})
    
    # print(merged_df)

    # merged_df.to_csv("/pdo/astronet-data/data/labels/sector_86_reannotated_only_jun_26.csv", index=False)


def main():
    sector_86_props_file = '/pdo/astronet-data/data/labels/sector_86_qlp_properties_with_centroid_CORRECTED.csv'
    model_pred = '/pdo/astronet-data/exodash/cached_model_results/sector_86_predictions.csv'

    props_df = pd.read_csv(sector_86_props_file)
    pred_df = pd.read_csv(model_pred)
    pred_df = pred_df.drop_duplicates(subset="astro_id", keep="first")


    print(props_df.head())
    print(pred_df.head())

    props_df["true_label"] = props_df["Astro ID"].map(
        pred_df.set_index("astro_id")["true_label"]
    )
    props_df["Astro ID"] = props_df["Astro ID"].astype("Int64")
    props_df.to_csv("/pdo/astronet-data/data/labels/sector_86_qlp_properties_with_centroid_CORRECTED_and_labels.csv", index=False)

if __name__ == "__main__":
    main()