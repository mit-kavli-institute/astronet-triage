"""Script for outputting a report for a given TFRecords set."""

from absl import app, flags, logging
import tensorflow as tf
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count


flags.DEFINE_string("tfrecord_dir", None, "TFRecord dir location.", required=True)
flags.DEFINE_string("output_dir", None, "Output dir location.", required=True)

FLAGS = flags.FLAGS

def extract_features_by_prefix(example, prefix):
    return {
        k: np.array(example.features.feature[k].float_list.value)
        for k in example.features.feature
        if k.startswith(prefix)
    }

def extract_scalar_features(example, keys):
    data = {}
    for key in keys:
        val = example.features.feature.get(key)
        if val and val.HasField('float_list'):
            data[key] = val.float_list.value[0]
        elif val and val.HasField('int64_list'):
            data[key] = val.int64_list.value[0]
    return data

def plot_scalar_properties(example, output_path):
    scalar_keys = [
        'astro_id', 'Period', 'Duration', 'Transit_Depth', 'Tmag',
        'star_mass', 'star_rad', 'star_rad_est',
        'disp_p', 'disp_n', 'disp_e', 'disp_b', 'disp_u', 'disp_t', 'disp_j', 'n_folds', 'n_points'
    ]
    scalar_data = extract_scalar_features(example, scalar_keys)

    # Format into aligned table text (grouped per line)
    items = [f"{k:<14}: {v:<.5g}" if isinstance(v, float) else f"{k:<14}: {v}" for k, v in scalar_data.items()]
    lines = []
    per_line = 3
    for i in range(0, len(items), per_line):
        lines.append("    ".join(items[i:i + per_line]))
    body_text = "\n".join(lines)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")

    astro_id = scalar_data.get("astro_id", "Unknown")
    fig.suptitle(f"Astro ID {astro_id} – Properties", fontsize=14, y=0.95)

    ax.text(
        0.01, 0.85, body_text,
        fontsize=11, family="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="#f8f8f8", edgecolor="gray", pad=0.6)
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_views(example, prefix, output_path, title_suffix=""):
    views = extract_features_by_prefix(example, prefix)
    if not views:
        return

    n = len(views)
    n_cols = 2
    n_rows = (n + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 2.5 * n_rows))
    axes = axes.flatten()

    for i, (k, v) in enumerate(sorted(views.items())):
        axes[i].plot(v, linewidth=0.5)
        axes[i].set_title(k)
        axes[i].set_xlabel("Index")
        axes[i].set_ylabel("Flux")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    astro_id = example.features.feature.get('astro_id', None)
    astro_id_val = astro_id.int64_list.value[0] if astro_id else "Unknown"
    fig.suptitle(f"Astro ID {astro_id_val} – {title_suffix}", fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_split_report(example, output_dir):
    astro_id = example.features.feature['astro_id'].int64_list.value[0]
    prefix =  os.path.join(output_dir, str(astro_id)[:3])
    os.makedirs(prefix, exist_ok=True)
    prefix = os.path.join(prefix, str(astro_id))

    plot_scalar_properties(example, f"{prefix}_props.png")
    plot_views(example, "global_view", f"{prefix}_global.png", title_suffix="Global Views")
    plot_views(example, "local_view", f"{prefix}_local.png", title_suffix="Local Views")
    plot_views(example, "secondary_", f"{prefix}_secondary.png", title_suffix="Secondary Views")
    plot_views(example, "sample_segments", f"{prefix}_segments.png", title_suffix="Sample Segments Views")

def run_parallel_report_generation(examples, output_dir, num_processes=None):
    num_processes = num_processes or cpu_count()
    args = [(ex, output_dir) for ex in examples]

    with Pool(num_processes) as pool:
        pool.starmap(generate_split_report, args)

def main(_):
    tfrecord_files = glob(f"{FLAGS.tfrecord_dir}/*")
    print(f'\nStarting to generate reports from {FLAGS.tfrecord_dir} with {len(tfrecord_files)} files...\n')
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    examples = []
    for i, raw_record in enumerate(dataset):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        examples.append(example)
        
    run_parallel_report_generation(examples, output_dir=FLAGS.output_dir, num_processes=32)

if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)