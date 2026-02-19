import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. LOAD DATA (Global Views)
# ==========================================

print("=" * 50)
print("Loading Global Views from TFRecords...")
print("=" * 50)

# Define path to TFRecords (Sectors 73-83 as per your logic)
tfrecord_files = []
for s in range(73, 84):
    path_pattern = f"/pdo/astronet-data/data/tfrecords/sector-{s}-scatter/*"
    found = glob.glob(path_pattern)
    tfrecord_files.extend(found)

tfrecord_files = sorted(tfrecord_files)

if not tfrecord_files:
    raise FileNotFoundError("No TFRecord files found. Check your paths.")

print(f"Found {len(tfrecord_files)} TFRecord shards.")

# Define the feature spec strictly for what we need
feature_spec = {
    "astro_id": tf.io.FixedLenFeature([1], tf.int64),
    "global_view": tf.io.FixedLenFeature([201], tf.float32),
}

# Create dataset
ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)

# iterate and collect data
# NOTE: This loads the data into RAM. If you have >1M stars,
# you might want to limit this or save to .npz for future reuse.
all_global_views = []
all_astro_ids = []

count = 0
print("Parsing records (this may take a moment)...")

# Parse in batches for slight speedup, or single example
# Using single example parsing here for simplicity and safety
for raw in ds:
    ex = tf.io.parse_single_example(raw, feature_spec)

    # Extract values
    gv = ex['global_view'].numpy()
    aid = ex['astro_id'].numpy()[0]

    all_global_views.append(gv)
    all_astro_ids.append(aid)

    count += 1
    if count % 10000 == 0:
        print(f"Loaded {count} light curves...", end='\r')

print(f"\nTotal loaded: {count}")

# Convert to numpy arrays for Scikit-Learn
X = np.array(all_global_views)  # The "Global View" Space
astro_ids = np.array(all_astro_ids)

print(f"Data Shape: {X.shape}")
print("-" * 30)

# ==========================================
# 2. FIND TARGET INDEX
# ==========================================

target_astro_id_str = '466376085'
target_idx = None

# Create a quick lookup or just iterate (iterating is fast enough for <1M)
# Using pandas for easy matching as per original script style
df = pd.DataFrame({'astro_id': astro_ids})

# Optimized search
matches = df[df['astro_id'].astype(str).str.startswith(target_astro_id_str)]

if not matches.empty:
    target_idx = matches.index[0]
    real_id = matches.iloc[0]['astro_id']
    print('Found Target!')
    print(f"Index: {target_idx}")
    print(f"Full astro_id: {real_id}")
    print(f"Global View shape: {X[target_idx].shape}")
else:
    print(f"Target ID {target_astro_id_str} not found in loaded records.")
    # Exit or handle error
    exit()

# ==========================================
# 3. NEAREST NEIGHBORS (Global View Space)
# ==========================================

n_neighbors_global_views = 20

# Normalize?
# Global views are usually flux values. Euclidean distance works best
# if they are on the same scale. Assuming they are pre-normalized by the pipeline.
# If not, you might want X_normalized = normalize(X, axis=1)
from sklearn.preprocessing import normalize

# X_normalized = normalize(X, norm='l2', axis=1)
X_used = X

print(f"X_used shape: {X_used.shape}")
#Take the last 100 points of the global views
X_used = X_used[:, 100:150]
print(f"X_used shape: {X_used.shape}")

print(f"\nFitting NearestNeighbors on Global View space (dim={X.shape[1]})...")
nbrs = NearestNeighbors(n_neighbors=(n_neighbors_global_views + 1), algorithm='brute', metric='euclidean')
nbrs.fit(X_used)

target_vector = X_used[target_idx].reshape(1, -1)
distances, indices = nbrs.kneighbors(target_vector)

# ==========================================
# 4. REPORT AND PLOT
# ==========================================

print(f"Target Index: {target_idx}")
print("-" * 30)

neighbor_info = []

for i in range(len(indices[0])):
    idx = indices[0][i]
    dist = distances[0][i]
    found_astro_id = astro_ids[idx]

    # Calculate similarity just for display
    # For global views (light curves), Pearson correlation is often a good metric too,
    # but here we stick to the Euclidean calc you used.
    similarity = 1 - (dist**2) / 2

    if idx == target_idx:
        print(f"Match {i}: Index {idx} (The query item itself) | Astro ID: {found_astro_id}")
    else:
        print(f"Match {i}: Index {idx} | Dist: {dist:.4f} | Sim: {similarity:.4f} | Astro ID: {found_astro_id}")

    neighbor_info.append({
        'index': idx,
        'astro_id': found_astro_id,
        'distance': dist,
        'cosine_sim': similarity,
        'is_query': (idx == target_idx),
        'data': X[idx] # We already have the data in memory!
    })

# Plotting
print("\n" + "=" * 50)
print("Plotting results...")
print("=" * 50)

n_neighbors = len(neighbor_info)
n_cols = 3
n_rows = (n_neighbors + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
if n_rows == 1: axes = axes.reshape(1, -1)
axes = axes.flatten() # Flatten for easy indexing

for i, info in enumerate(neighbor_info):
    ax = axes[i]
    gv = info['data']

    ax.plot(gv, color='tab:blue')

    # Highlight the query
    if info['is_query']:
        ax.plot(gv, color='tab:orange', linewidth=2, alpha=0.7)
        title = f"QUERY\nID: {info['astro_id']}"
        ax.set_facecolor('#fff8e1') # Light yellow background for query
    else:
        title = f"ID: {info['astro_id']}\nDist: {info['distance']:.4f}"

    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for i in range(n_neighbors, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(f'nn_global_view_space_idx{target_idx}.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to: nn_global_view_space_idx{target_idx}.png")
plt.show()

# ==========================================
# 5. DISTRIBUTION PLOTS (Top 500)
# ==========================================

print('\nCalculating stats for top 500 neighbors...')
n_neighbors_to_find = 501
nbrs_500 = NearestNeighbors(n_neighbors=n_neighbors_to_find, algorithm='brute', metric='euclidean')
nbrs_500.fit(X_used)

distances_500, indices_500 = nbrs_500.kneighbors(target_vector)
neighbor_distances = distances_500[0][1:] # Exclude self

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram
ax1.hist(neighbor_distances, bins=30, color='mediumpurple', edgecolor='black', alpha=0.7)
ax1.set_title(f'Dist. Distribution (Global View Space)\nTarget: {target_astro_id_str}')
ax1.set_xlabel('Euclidean Distance')
ax1.set_ylabel('Frequency')

# Plot 2: Rank vs Distance
ax2.plot(range(1, 501), neighbor_distances, color='indigo', linewidth=1)
ax2.set_title('Distance vs Rank')
ax2.set_xlabel('Rank')
ax2.set_ylabel('Distance')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'dist_distribution_gv_space_idx{target_idx}.png', dpi=150)
print(f"Saved stats plot.")
