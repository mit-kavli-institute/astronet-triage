'''

I need something that looks like this:
Astro ID,tic_id,planetno,model_no,disp_p,disp_e,disp_n,disp_j
26030031601,260300316,1,0,0.9066316,0.0073629273,0.0024443907,0.083561055
26036363001,260363630,1,0,0.8635499,0.040531892,0.0058301706,0.090088
26037647702,260376477,2,0,0.42106113,0.12508824,0.0031241935,0.45072642

'''


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from astronet import models, evaluation
from astronet.util import config_util

# Adjust these paths:
# model_dir = '/pdo/users/pablomer/mnt/tess/models/vetting/20250429/cshallue/AstroCNNModelVetting_cshallue_20250429_181612'
model_dir = '/pdo/users/pablomer/mnt/tess/models/vetting/20250723/pablomer/AstroCNNModelVetting_pablomer_20250723_224824'

# test_tfrecord_pattern = '../mnt/tess/astronet/tfrecords-vetting-v01-tois-triageJs-nocentroid-april2025-test/*'
test_tfrecord_pattern = '/pdo/astronet-data/data/tfrecords/sector-86/000**-of-00025'



train_flags = config_util.load_config(os.path.join(model_dir, 'train_flags.json'))
config      = config_util.load_config(os.path.join(model_dir, 'config.json'))

# load the model
model_name = train_flags['model']
# model = models.load_model(model_name, model_dir)
with tf.device('/CPU:0'):
    model = models.load_model(model_name, model_dir)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

#Get both preds and true labels + IDs

from astronet.astro_cnn_model import input_ds
import numpy as np

batch_size = config.hparams.batch_size

# 1) Build dataset for predictions (drops labels, keeps IDs)
ds_pred = input_ds.build_eval_dataset(
    file_pattern=test_tfrecord_pattern,
    input_config=config.inputs,
    batch_size=batch_size,
    include_identifiers=True,
    include_labels=False
)
with tf.device('/CPU:0'):
    preds = model.predict(ds_pred)     # shape (N, C)

# Get the IDs
ids = []
for batch in ds_pred:
    ids.append(batch[1].numpy())  # batch[1] is 'astro_id'
ids = np.concatenate(ids)


print('Shape of preds:', preds.shape)
print('First 5 preds:', preds[:5])
print('First 5 ids:', ids[:5])

import pandas as pd

df=pd.DataFrame(ids, columns=["Astro ID"])

# df=pd.DataFrame(preds, columns=["disp_p", "disp_e", "disp_n", "disp_j"])


print(df.head())

tic_ids = [int(str(x)[:-2]) for x in ids]
planet_nos = [int(str(x)[-2:]) for x in ids]
model_nos = [0 for x in ids]

df['tic_id'] = tic_ids
df['planetno'] = planet_nos
df['model_no'] = model_nos

df['disp_p'] = preds[:,0]
df['disp_e'] = preds[:,1]
df['disp_n'] = preds[:,2]
df['disp_j'] = preds[:,3]

print(df.head())

print('Size of df:', df.shape)

#save the df to a csv
saving_path = '/pdo/users/pablomer/mnt/tess/models/vetting/20250723_modelwithtoisremoved_predictions_sector86.csv'
df.to_csv(saving_path, index=False)

print('Saved to:', saving_path)
