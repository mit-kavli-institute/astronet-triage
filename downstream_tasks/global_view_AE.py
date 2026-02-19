import numpy as np
import pandas as pd
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras import layers, models, callbacks

# ==========================================
# 1. LOAD DATA
# ==========================================
print("=" * 50)
print("Loading Global Views...")
print("=" * 50)

# 1. Define paths (Adjust if your path differs)
tfrecord_files = []
for s in range(73, 84):
    path_pattern = f"/pdo/astronet-data/data/tfrecords/sector-{s}-scatter/*"
    found = glob.glob(path_pattern)
    tfrecord_files.extend(found)

tfrecord_files = sorted(tfrecord_files)
if not tfrecord_files:
    raise FileNotFoundError("No TFRecord files found. Check your paths.")

print(f"Found {len(tfrecord_files)} TFRecord shards.")

# 2. Parse TFRecords
feature_spec = {
    "astro_id": tf.io.FixedLenFeature([1], tf.int64),
    "global_view": tf.io.FixedLenFeature([201], tf.float32),
}

ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)

all_global_views = []
all_astro_ids = []
count = 0

for raw in ds:
    ex = tf.io.parse_single_example(raw, feature_spec)
    all_global_views.append(ex['global_view'].numpy())
    all_astro_ids.append(ex['astro_id'].numpy()[0])
    count += 1
    if count % 10000 == 0: print(f"Loaded {count}...", end='\r')

X_raw = np.array(all_global_views)
astro_ids = np.array(all_astro_ids)
print(f"\nTotal loaded: {len(X_raw)}")

# ==========================================
# 2. PREPROCESSING
# ==========================================
# Expand dims (N, 201, 1)
X_processed = np.expand_dims(X_raw, axis=-1)

# MinMax Normalize per sample (Important for MAE)
X_min = X_processed.min(axis=1, keepdims=True)
X_max = X_processed.max(axis=1, keepdims=True)
X_denom = X_max - X_min
X_denom[X_denom == 0] = 1
X_norm = (X_processed - X_min) / X_denom

# ==========================================
# 3. CUSTOM MASKING LAYER & MAE MODEL
# ==========================================

class ThreePatchMasker(layers.Layer):
    """
    For each sample in the batch, selects 3 random starting positions
    and masks the subsequent 10 points (setting them to 0).
    Patches can overlap.
    """
    def __init__(self, patch_len=10, num_patches=3, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.num_patches = num_patches

    def call(self, inputs, training=None):
        if not training:
            return inputs # Pass through untouched during inference

        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1] # 201

        # We need a mask of shape (Batch, Seq_Len) initialized to 0 (don't mask)
        # We will set spots to 1 (mask this)
        mask_accum = tf.zeros((batch_size, seq_len))

        # Create a range [0, 1, ..., 200] to compare against
        indices = tf.range(seq_len, dtype=tf.float32)
        indices = tf.expand_dims(indices, 0) # (1, 201)

        # Loop 3 times to create 3 patches
        # (This loop runs at graph construction time, not runtime, so it's efficient)
        for _ in range(self.num_patches):
            # Pick a random start index for each item in batch
            # Max index is seq_len - patch_len to avoid falling off the edge
            starts = tf.random.uniform((batch_size, 1), minval=0, maxval=seq_len - self.patch_len, dtype=tf.int32)
            starts = tf.cast(starts, tf.float32)

            # Create a boolean mask for this specific patch
            # (index >= start) AND (index < start + patch_len)
            lower_bound = tf.cast(indices >= starts, tf.float32)
            upper_bound = tf.cast(indices < (starts + self.patch_len), tf.float32)
            patch_mask = lower_bound * upper_bound

            # Add to accumulator
            mask_accum = mask_accum + patch_mask

        # Clip accumulator: If a point was hit by 1 or more patches, it is now > 0.
        # We want: 1 = drop, 0 = keep.
        drop_mask = tf.clip_by_value(mask_accum, 0, 1)

        # Invert: 1 = keep, 0 = drop
        keep_mask = 1.0 - drop_mask
        keep_mask = tf.expand_dims(keep_mask, -1) # (Batch, 201, 1)

        return inputs * keep_mask

# --- ARCHITECTURE ---
input_shape = (201, 1)
latent_dim = 32

inputs = layers.Input(shape=input_shape)

# 1. Apply Custom Masking
masked_inputs = ThreePatchMasker(patch_len=10, num_patches=3)(inputs)

# 2. Encoder
x = layers.Conv1D(32, 5, activation="relu", strides=1, padding="same")(masked_inputs)
x = layers.Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv1D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
encoded = layers.Dense(latent_dim, activation="relu", name="embedding")(x)

# 3. Decoder
x = layers.Dense(51 * 128, activation="relu")(encoded)
x = layers.Reshape((51, 128))(x)
x = layers.Conv1DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv1DTranspose(32, 5, activation="relu", strides=1, padding="same")(x)
x = layers.Conv1DTranspose(1, 3, activation="sigmoid", strides=1, padding="same")(x)
decoded = layers.Cropping1D((0, 3))(x) # Crop to 201

# --- COMPILE ---
mae = models.Model(inputs, decoded)
mae.compile(optimizer='adam', loss='mse')
mae.summary()

# ==========================================
# 4. TRAIN
# ==========================================
callbacks_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1)
]

print("\nStarting Training...")
history = mae.fit(
    X_norm, X_norm, # Input is X (will be masked internally), Target is full X
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks_list,
    verbose=1
)

# ==========================================
# 5. EXTRACT EMBEDDINGS
# ==========================================
# Isolate the encoder part
encoder_model = models.Model(inputs=mae.input, outputs=mae.get_layer("embedding").output)

print("\nExtracting embeddings...")
# Masking layer is inactive during prediction, so we embed the FULL light curve
embeddings = encoder_model.predict(X_norm)

# ==========================================
# 6. NEAREST NEIGHBORS SEARCH
# ==========================================
target_astro_id_str = '466376085'

# Find target index
matches = pd.DataFrame({'astro_id': astro_ids})
match_row = matches[matches['astro_id'].astype(str).str.startswith(target_astro_id_str)]

if match_row.empty:
    print(f"Target {target_astro_id_str} not found.")
    exit()

target_idx = match_row.index[0]
print(f"Target found at index: {target_idx}")

# Search
nbrs = NearestNeighbors(n_neighbors=21, metric='euclidean', algorithm='brute')
nbrs.fit(embeddings)
dists, indices = nbrs.kneighbors(embeddings[target_idx].reshape(1, -1))

# ==========================================
# 7. VISUALIZE
# ==========================================
indices = indices[0]
dists = dists[0]

print("Plotting results...")

fig, axes = plt.subplots(7, 3, figsize=(15, 15))
axes = axes.flatten()

for i, idx in enumerate(indices):
    ax = axes[i]

    # Plot original raw data
    ax.plot(X_raw[idx], color='tab:blue', alpha=0.9, linewidth=1)

    title = f"ID: {astro_ids[idx]}\nDist: {dists[i]:.4f}"

    if idx == target_idx:
        title = f"QUERY\n{title}"
        ax.set_facecolor('#fff8e1')
        ax.plot(X_raw[idx], color='tab:orange', linewidth=2, alpha=0.8) # Highlight query trace

    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
filename = f'mae_3patch_neighbors_{target_astro_id_str}.png'
plt.savefig(filename)
print(f"Done. Saved plot to {filename}")
