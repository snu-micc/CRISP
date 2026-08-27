# Reproducibility guide

## 1. Environment

Use Python 3.10 or later and install the package in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set `OPENAI_API_KEY` in the environment or a local `.env` file. Never commit the key.

## 2. Protocol check before a full run

```bash
crisp compile --config configs/quickstart.json --output runs/quickstart
```

The quick start confirms API access, numbered-rule parsing, embedding dimensions, clustering, consolidation, code generation, and the static code gate. It is not a miniature performance experiment and produces no labels, targets, or metrics.

## 3. Full synthesizability-scale compilation

```bash
crisp compile \
  --config configs/paper_synthesizability.json \
  --output runs/synthesizability_full
```

The configuration uses 1,000 discovery calls with ten proposed rules per call, `text-embedding-3-large`, PCA to at most 256 dimensions, and 50 k-means clusters with 50 initializations and seed 42. Cluster identifiers are assigned in decreasing order of cluster size, with the original k-means label used only as a deterministic tie-breaker.

The workflow can resume from the same output directory. Completed API records are reused, and a changed configuration is rejected.

## 4. Output layout

```text
runs/example/
├── config_frozen.json
├── provenance.json
├── run_summary.json
├── raw/
│   ├── discovery/
│   ├── consolidation/
│   └── code_generation/
├── stages/
│   ├── 01_discovered_rules.json
│   ├── 02_embeddings.npy
│   ├── 02_embedding_usage.json
│   ├── 03_cluster_assignments.json
│   ├── 03_cluster_summary.json
│   ├── 04_consolidated_rules.json
│   └── 05_generated_code_manifest.json
└── generated_descriptors/
```

The run directory can contain raw API text and should be reviewed before sharing. It is excluded by `.gitignore`.

The generated functions are outputs of a new stochastic API run. They are not the frozen implementations used to produce the manuscript's benchmark results.

## 5. Dataset-blind boundary

The `compile` command accepts only a configuration and an output path. It has no dataset, CIF, label, target, split, prediction, or metric argument. The provenance file records an empty `dataset_paths_read` list.

## 6. Generated-code review

The static gate checks syntax, requires one `structure` argument, rejects imports and common file/network/process access patterns, and records findings in the code manifest. It is deliberately conservative and is not a sandbox. Generated code is never imported or executed by `crisp compile`; a qualified reviewer must approve it before use.

## 7. Model snapshots and stochasticity

The configuration records the model identifiers requested from the API. The discovery stage intentionally uses stochastic generation to explore a broad rule space. Even with identical prompts and model identifiers, provider updates and sampling can change individual rules. Reproduction therefore means reproducing the disclosed procedure, scale, constraints, and audit trail—not obtaining byte-identical language-model output.

The original consolidation stage used the GPT-5 model available during the study. Because a dated consolidation snapshot was not preserved in the study record, the public configuration uses the stable `gpt-5` alias and records the model identifier returned by every response.

The implementation uses the OpenAI [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) with response storage disabled and the [Embeddings API](https://developers.openai.com/api/docs/guides/embeddings) for semantic rule vectors.
