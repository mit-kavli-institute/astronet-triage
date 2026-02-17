import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the npz file
z_dim = 64
data = np.load(f'embeddings_zdim{z_dim}.npz')
n_neighbors_global_views = 20 # 11 neighbors plus the target itself
# target_idx = 1448

# 2. Extract the array (replace 'my_array' with your actual key)
# If you don't know the key, use data.files to see them
embeddings = data['embeddings']
astro_ids = data['astro_ids']
print(embeddings.shape)


# 3. Convert to DataFrame
df = pd.DataFrame(embeddings)
df['astro_id'] = astro_ids
print('Length of dataframe: ', len(df))

# print(df.head())

for index, astro_id in df['astro_id'].items():
    # print(type(astro_id))
    # print(astro_id)
    # print(str(astro_id)[:-2])

    if str(astro_id)[:-2] == '466376085':
        print('Found it!')
        print(index)
        print(f"Full astro_id: {astro_id}")
        print(f"Embedding shape: {df.iloc[index].drop('astro_id').shape}")
        print(f"Embedding values (first 10): {df.iloc[index].drop('astro_id').values[:10]}")
        print(f"Full row data:")
        print(df.iloc[index])
        target_idx = index
        break

print(f"Target Index: {target_idx}")


import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# 1. Assume X is your dataset of shape (N, 512)
# Using random data for demonstration
X = embeddings

# 2. PRE-PROCESSING (Crucial for Embeddings)
# L2-normalize the vectors so Euclidean distance = Cosine Similarity
X_normalized = normalize(X, norm='l2', axis=1)
# X_normalized = X


# 3. Initialize and Fit the Model
# 'brute' is often faster for high dimensions (512) than tree-based methods
# n_neighbors=6 because the closest neighbor will be the point itself (distance 0)
nbrs = NearestNeighbors(n_neighbors=(n_neighbors_global_views+1), algorithm='brute', metric='euclidean')
nbrs.fit(X_normalized)

# 4. Select your target vector (Target Index)
target_vector = X_normalized[target_idx].reshape(1, -1)

# 5. Find Neighbors
distances, indices = nbrs.kneighbors(target_vector)

# Output results and collect astro_ids for plotting
print(f"Target Index: {target_idx}")
print("-" * 30)
neighbor_astro_ids = []
neighbor_info = []

for i in range(len(indices[0])):
    idx = indices[0][i]
    dist = distances[0][i]
    astro_id = astro_ids[idx]

    # Convert Euclidean distance back to Cosine Similarity score if desired
    # Similarity = 1 - (dist^2) / 2
    cosine_sim = 1 - (dist**2) / 2

    if idx == target_idx:
        print(f"Match {i}: Index {idx} (The query item itself) | Astro ID: {astro_id}")
    else:
        print(f"Match {i}: Index {idx} | Euclidean Dist: {dist:.4f} | Cosine Sim: {cosine_sim:.4f} | Astro ID: {astro_id}")

    neighbor_astro_ids.append(int(astro_id))
    neighbor_info.append({
        'index': idx,
        'astro_id': int(astro_id),
        'distance': dist,
        'cosine_sim': cosine_sim,
        'is_query': (idx == target_idx)
    })

# Now retrieve and plot global views for all neighbors
print("\n" + "=" * 50)
print("Retrieving global views from TFRecord files...")
print("=" * 50)

# TFRecord path - adjust this to match your actual path
# tfrecord_path = os.path.expanduser("/pdo/astronet-data/data/tfrecords/sector-82-scatter/000*-of-00050")
# tfrecord_files = sorted(glob.glob(tfrecord_path))
tfrecord_files = []
for s in range(73, 84):
    path_pattern = f"/pdo/astronet-data/data/tfrecords/sector-{s}-scatter/*"
    # Find files for this specific sector and add them to our master list
    tfrecord_files.extend(glob.glob(path_pattern))

tfrecord_files = sorted(tfrecord_files)


if not tfrecord_files:
    print("Warning: No TFRecord files found at the specified path.")
    print("Please update the tfrecord_files path in the script.")
else:
    print(f"Found {len(tfrecord_files)} TFRecord shards")

    feature_spec = {
        "astro_id": tf.io.FixedLenFeature([1], tf.int64),
        "global_view": tf.io.FixedLenFeature([201], tf.float32),
    }

    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)

    # Create a dictionary to store global views by astro_id
    global_views = {}

    # Search through all records to find the ones we need
    print("Searching for global views...")
    for raw in ds:
        ex = tf.io.parse_single_example(raw, feature_spec)
        astro_id = int(ex["astro_id"][0].numpy())
        # print(f"Astro ID from TFRecord: {astro_id} | Neighbor Astro IDs: {neighbor_astro_ids}")
        if astro_id in neighbor_astro_ids:
            gv = ex["global_view"].numpy()
            global_views[astro_id] = gv
            print(f"Found global view for astro_id: {astro_id}")
            if len(global_views) == len(neighbor_astro_ids):
                break

    # Plot all global views
    if global_views:
        n_neighbors = len(neighbor_info)
        n_cols = 3
        n_rows = (n_neighbors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, info in enumerate(neighbor_info):
            astro_id = info['astro_id']
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            if astro_id in global_views:
                gv = global_views[astro_id]
                ax.plot(gv)
                title = f"Astro ID: {astro_id}"
                if info['is_query']:
                    title += " (Query)"
                else:
                    title += f"\nDist: {info['distance']:.4f}, Sim: {info['cosine_sim']:.4f}"
                ax.set_title(title)
                ax.set_xlabel("Bins (phase)")
                ax.set_ylabel("Flux")
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f"Not found:\n{astro_id}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Astro ID: {astro_id} (Not Found)")

        # Hide unused subplots
        for i in range(n_neighbors, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(f'nearest_neighbors_global_views_zdim{z_dim}.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: nearest_neighbors_global_views_zdim{z_dim}.png")
        plt.show()
    else:
        print("No global views found for any of the neighbors.")



# 1. Set the number of neighbors to 501
# (500 neighbors + 1 for the target item itself)
print('Now making the plot for the top 500 neighbors')
n_neighbors_to_find = 501
nbrs_500 = NearestNeighbors(n_neighbors=n_neighbors_to_find, algorithm='brute', metric='euclidean')
nbrs_500.fit(X_normalized)

# 2. Find neighbors for your target index
distances, indices = nbrs_500.kneighbors(target_vector)

# 3. Extract distances, excluding the first one (which is the query point itself, dist=0)
# We flatten distances[0] which is of shape (1, 501) to (501,)
neighbor_distances = distances[0][1:]

# 4. Create the visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Histogram ---
ax1.hist(neighbor_distances, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_title(f'Distribution of Distances (Top 500 NNs)\nTarget Index: {target_idx}')
ax1.set_xlabel('Euclidean Distance')
ax1.set_ylabel('Frequency')
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# --- Plot 2: Distance vs Rank ---
ax2.plot(range(1, 501), neighbor_distances, marker='o', markersize=2, linestyle='-', color='teal')
ax2.set_title('Distance to Neighbor by Rank')
ax2.set_xlabel('Neighbor Rank (1st to 500th)')
ax2.set_ylabel('Euclidean Distance')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f'distance_distribution_zdim{z_dim}_idx{target_idx}.png', dpi=150)
print(f"Saved plot to: distance_distribution_zdim{z_dim}_idx{target_idx}.png")
# plt.show()

print(f"Mean distance to top 500: {np.mean(neighbor_distances):.4f}")
print(f"Closest (rank 1): {neighbor_distances[0]:.4f}")
print(f"Furthest (rank 500): {neighbor_distances[-1]:.4f}")
