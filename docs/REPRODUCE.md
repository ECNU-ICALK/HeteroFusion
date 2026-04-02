# Reproduce HeteroFusion Experiments

This document explains how the current repository maps to the paper experiments and what needs to be prepared on a new machine.

## 1. Required Assets

The paper experiments assume three categories of external assets:

- base models, referenced by `MODEL_ROOT`
- LoRA experts, referenced by `ADAPTER_ROOT`
- benchmark datasets compatible with the local `llamafactory` data pipeline

Example:

```bash
export MODEL_ROOT=/mnt/models
export ADAPTER_ROOT=/mnt/adapters
```

The YAML files then resolve paths such as:

```yaml
base_model_path: "${MODEL_ROOT}/llama-3.1-8b-instruct"
initial_target_lora: "${ADAPTER_ROOT}/llama_instruct/NER_mit-movie_sft"
```

## 2. Config Anatomy

Each experiment config follows the same structure:

```yaml
experiment_name: ...
output_dir: ...
base_model_path: ...
initial_target_lora: ...
data_global:
  dataset_dir: data
  template: llama3
  cutoff_len: 1024
  batch_size: 1
tasks:
  - task_name: ...
    source_lora_paths:
      - ...
    transfer_ratio: "1"
    datasets:
      - name: ...
        type: main
      - name: ...
        type: replay
        ratio: all
    training:
      embed_dim: 1024
      num_heads: 8
      num_epochs: 3
      lr: 5.0e-05
      alpha_init: 0.3
      mu_gate: 0.1
      lambda_reg: 0.005
```

Important fields:

- `initial_target_lora`: the target anchor adapter that preserves task identity
- `source_lora_paths`: the heterogeneous expert pool
- `datasets`: replay mixture used to optimize the transfer operator
- `alpha_init`: initial scale for predicted `lora_B` updates
- `mu_gate`: denoising gate shift
- `lambda_reg`: RDM regularization weight

## 3. Main Entry Point

Run any config through:

```bash
python main.py --config_path <path-to-yaml>
```

The pipeline will:

1. load the target base model
2. wrap it with the target LoRA
3. load source LoRAs
4. build mixed replay dataloaders
5. optimize the HeteroFusion transfer networks
6. export the fused adapter to `output_dir/.../merged_lora/`

## 4. Experiment Map

### Single-source heterogeneous transfer

Configs:

- `configs/heterofusion/paper_experiments/single_source_qwen_to_llama/llama_target_mit_movie.yaml`
- `configs/heterofusion/paper_experiments/single_source_qwen_to_llama/llama_target_tweetner7.yaml`
- `configs/heterofusion/paper_experiments/single_source_qwen_to_llama/llama_target_conll04.yaml`
- `configs/heterofusion/paper_experiments/single_source_qwen_to_llama/llama_target_new_york_times_re.yaml`
- `configs/heterofusion/paper_experiments/single_source_qwen_to_llama/llama_target_findvehicle.yaml`
- `configs/heterofusion/paper_experiments/single_source_qwen_to_llama/llama_target_fabner.yaml`

Batch runner:

```bash
bash run_paper_single_source_qwen_to_llama.sh
```

### Multi-source cross-family fusion

Configs:

- `configs/heterofusion/paper_experiments/multi_source_cross_family/llama_target_qwen_mistral.yaml`
- `configs/heterofusion/paper_experiments/multi_source_cross_family/qwen_target_llama_mistral.yaml`

### Noisy-source robustness

Config:

- `configs/heterofusion/paper_experiments/noise_robustness/llama_target_with_noisy_llama_experts.yaml`

### GLUE cross-family transfer

Configs:

- `configs/heterofusion/paper_experiments/glue_cross_family/llama_target/*.yaml`
- `configs/heterofusion/paper_experiments/glue_cross_family/qwen_target/*.yaml`

Batch runner:

```bash
bash run_paper_glue_cross_family.sh
```

### Sensitivity analysis

Configs:

- `configs/heterofusion/paper_experiments/sensitivity_alpha/*.yaml`
- `configs/heterofusion/paper_experiments/sensitivity_mu_gate/*.yaml`

Batch runner:

```bash
bash run_paper_hyperparameter_sweep.sh
```

## 5. Sample Replay Data

The repository currently ships lightweight replay subsets under `data/sample/`, for example:

- `mit-movie_sample300.json`
- `TweetNER7_sample_15000_sample300.json`
- `RE_conll04_sample300.json`
- `RE_New-York-Times-RE_sample_30000_sample300.json`
- `ET_FabNER_sample300.json`
- `ET_FindVehicle_sample300.json`

These files support the paper's replay-based fusion recipe, where 300 samples per involved task are mixed during HeteroFusion optimization.

## 6. Outputs

Each config writes to its own `output_dir`. The main artifact is:

```text
<output_dir>/<task_name>/merged_lora/adapter_model.safetensors
```

This exported adapter preserves the target LoRA basis and replaces aligned `lora_B` weights with fused weights predicted by the transfer network.

## 7. Practical Notes Before Public Release

- The repository currently exposes the research pipeline, not a fully packaged training framework.
- Base-model checkpoints and expert adapters are intentionally referenced externally because of size and license constraints.
- If you plan to release pretrained fused adapters, document their source models, tasks, and licenses explicitly.
- If you want fully one-command reproducibility for outsiders, the next step is to add a locked environment file and a formal evaluation script for each benchmark.

