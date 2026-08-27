# Public release scope

This repository exposes the CRISP representation-construction method while keeping experimental and potentially sensitive research assets outside the public tree.

## Included

- dataset-blind prompts for synthesizability, formation energy, room-temperature ionic conductivity, and shear modulus;
- OpenAI API orchestration for rule discovery, embeddings, PCA, k-means clustering, rule-family consolidation, and descriptor-code generation;
- configuration, prompt, code, and output hashes needed to audit a run;
- static checks for generated descriptor code;
- the published 50-rule synthesizability catalog as rule names and operational summaries.

## Not included

- Materials Project or third-party crystal files;
- synthesis labels, regression targets, train/test identifiers, chemical-family holdouts, or external validation cohorts;
- raw production responses from the API runs used in the study;
- frozen executable descriptor implementations or CIF featurization utilities;
- per-material descriptor matrices, score vectors, fitted models, checkpoints, or figure-source tables;
- benchmark metrics, manuscript result tables, or scripts that reconstruct reported figures;
- partner-provided or institutionally restricted assets.

The included workflow can rerun the disclosed rule-compilation protocol on a new target and generate new candidate descriptor functions for review. The omitted frozen implementations and research assets are needed to recreate the manuscript's exact feature matrix and numerical benchmarks and are intentionally outside this public release.

## Intellectual-property boundary

The repository should not be interpreted as a patent license or a general open-source grant. The public code documents the disclosed scientific workflow and supports scholarly inspection. See [`NOTICE.md`](../NOTICE.md) before reuse or redistribution.
