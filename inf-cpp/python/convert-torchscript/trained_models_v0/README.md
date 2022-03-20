# HSF Pipeline

Trained model for HSF pipeline. Model and inference code is associated in case for any modifications.

## Pipeline

- Config: `pipeline_test.yaml`

## Embedding

- Rescale: `3000, pi, 400`
- Model(Code): `Embedding/model`
- Config: `Embedding/model/embedding_test.yaml`
- Trained Model: `Embedding/trained/checkpoints/last.ckpt`

## Filter

- Model(Code): `Filter/model`
- Config: `Filter/model/embedding_test.yaml`
- Trained Model: `Filter/trained/checkpoints/last.ckpt`

## GNN
- Model(Code): `GNN/model`
- Config: `GNN/model/embedding_test.yaml`
- Trained Model: `GNN/trained/checkpoints/last.ckpt`

