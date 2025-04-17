# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A convolutional model for classifying light curves for TESS vetting."""




"""

 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                               INPUT PIPE                                        │
 │  • multiple time-series views  →  {ts_feature_1, ts_feature_2, …, ts_feature_K} │
 │  • auxiliary scalars / vectors →  {aux_feature_1, …, aux_feature_M}             │
 └─────────────────────────────────────────────────────────────────────────────────┘
         │                                   │                                   |
         │ ts                                │ ts + aux                          | aux
         ▼                                   ▼                                   |
┌───────────────────────┐            ┌───────────────────────────────────────┐   │
│  VETTING TS BLOCKS    │            │        TRIAGE SUB-MODEL               │   │
│  (conv stack per view)│            │                                       │   │
│  ts_block_1 … K       │            │  ┌──────────────────────────────────┐ │   │
└───────────────────────┘            │  │  TRIAGE TS BLOCKS (conv per view)│ │   │
         │                           │  ├──────────────────────────────────┤ │   │
         │                           │  │  TRIAGE AUX INPUT CONCATENATION  │ │   │
         |                           │  ├──────────────────────────────────┤ │   │
         |                           │  │  TRIAGE FINAL FC LAYERS          │ │   │
         |                           │  └──────────────────────────────────┘ │   │
         |                           └───────────────────────────────────────┘   │
         |                                             |                         |
				 ▼																						 ▼                         |
 ┌───────────────────────┐					 ┌───────────────────────────────────────┐   │
 │     VETTING           │					 │         (a) triage logits             │   │
 │  TS EMBEDDINGS (K)    │           │               **or**                  │   │ aux
 └───────────────────────┘           │         (b) triage penultimate        │   │
         |                           │             embeddings                │   │
         |                           └─────────────────┬─────────────────────┘   │
         |                                             │                         |
         ▼                                             ▼                         ▼
        |__________________________________________________________________________|
												        |
         ┌──────────────────────────────────────────────────┐
         │     CONCATENATE ALL FEATURES (pre_logits)        │
         │  1) vetting ts embeddings (K)                    │
         │  2) triage output (logits or embeddings)         │
         │  3) auxiliary features (M)                       │
         └─────────────────────────┬────────────────────────┘
	                                 │
	                                 ▼
		                 ┌────────────────────────────────┐
		                 │  VETTING FINAL FC LAYERS       │
		                 └─────────────┬──────────────────┘
		                               │
		                               ▼
	                       ┌─────────────────────┐
	                       │  Vetting logits     │
	                       └─────────────────────┘
	                                 │ Softmax / Sigmoid
	                                 ▼
	                       ┌─────────────────────┐
	                       │  Vetting prediction │
	                       └─────────────────────┘


"""




from astronet.astro_cnn_model import astro_cnn_model, base


class AstroCNNModelVetting(astro_cnn_model.AstroCNNModel):
  """A convolutional model for classifying light curves for TESS vetting."""

  def __init__(self, config, triage_model):
    super().__init__(config)
    self.triage_model = triage_model
    if not config.hparams.use_preds_layer:
      self.triage_model.make_embeds_only()

  def call(self, inputs, training=None):
    ts_inputs, aux_inputs = base.unpack_inputs(inputs, self.config.hparams)
    y = []
    for k in sorted(ts_inputs.keys()):
      v = ts_inputs[k]
      y.append(base.apply_block(self.ts_blocks[k], v, training))
    y.extend([aux_inputs[k] for k in sorted(aux_inputs.keys())])
    y.append(self.triage_model(inputs, training=training))  # Triage embedding.
    y = base.apply_block(self.final, y, training)

    return y
