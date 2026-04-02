import fire
import yaml
import torch
import os
import json
import gc
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from safetensors.torch import load_file
from data import MixDatasetBuilder
from src.trainer import HeteroFusionTrainer


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def load_adapter_conf_safe(path):
    with open(os.path.join(path, "adapter_config.json"), 'r') as f:
        conf = json.load(f)
    
    keys_to_remove = ['corda_config', 'eva_config', 'piss_config', 'auto_mapping', 'revision', 'trainable_token_indices', 'exclude_modules']
    for k in keys_to_remove: 
        if k in conf: del conf[k]
        
    try:
        return LoraConfig(**conf)
    except TypeError:
        valid = ['r', 'lora_alpha', 'target_modules', 'lora_dropout', 'bias', 'task_type', 'modules_to_save', 'layers_to_transform', 'layers_pattern', 'rank_pattern', 'alpha_pattern']
        print("Warning: Fallback to whitelist config.")
        return LoraConfig(**{k: v for k, v in conf.items() if k in valid})


def resolve_path(path_value, *, base_dir=PROJECT_ROOT):
    if not path_value:
        return path_value

    expanded = os.path.expandvars(os.path.expanduser(path_value))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(base_dir, expanded))


def resolve_pipeline_config(config, config_path):
    resolved = dict(config)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    base_dir = PROJECT_ROOT if config_dir.startswith(PROJECT_ROOT) else config_dir

    resolved['output_dir'] = resolve_path(config['output_dir'], base_dir=base_dir)
    resolved['base_model_path'] = resolve_path(config['base_model_path'], base_dir=base_dir)
    resolved['initial_target_lora'] = resolve_path(config['initial_target_lora'], base_dir=base_dir)

    data_global = dict(config.get('data_global', {}))
    if 'dataset_dir' in data_global:
        data_global['dataset_dir'] = resolve_path(data_global['dataset_dir'], base_dir=base_dir)
    resolved['data_global'] = data_global

    resolved_tasks = []
    for task in config.get('tasks', []):
        resolved_task = dict(task)
        if task.get('source_lora_path'):
            resolved_task['source_lora_path'] = resolve_path(task['source_lora_path'], base_dir=base_dir)
        if task.get('source_lora_paths'):
            resolved_task['source_lora_paths'] = [
                resolve_path(path, base_dir=base_dir) for path in task['source_lora_paths']
            ]
        resolved_tasks.append(resolved_task)
    resolved['tasks'] = resolved_tasks
    return resolved

def run_pipeline(config_path):
    print(f"=== Loading Pipeline Config from {config_path} ===")
    with open(config_path, 'r') as f:
        config = resolve_pipeline_config(yaml.safe_load(f), config_path)
    
    global_output_dir = config['output_dir']
    os.makedirs(global_output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    global_lpka_cache = {}

    print(f"Loading Base Model: {config['base_model_path']}")
    base_model_raw = AutoModelForCausalLM.from_pretrained(
        config['base_model_path'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    data_builder = MixDatasetBuilder(config, config['base_model_path'])
    current_target_lora_path = config['initial_target_lora']
    print(f"Initial Target LoRA: {current_target_lora_path}")

    for idx, task in enumerate(config['tasks']):
        task_name = task.get('task_name', f"Task_{idx}")
        print(f"\n{'='*20} Starting Task {idx+1}/{len(config['tasks'])}: {task_name} {'='*20}")
        
        task_output_dir = os.path.join(global_output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        print(f"Wrapping Base Model with Target LoRA: {current_target_lora_path}")
        peft_config = load_adapter_conf_safe(current_target_lora_path)
        
        model = PeftModel.from_pretrained(
            base_model_raw, 
            current_target_lora_path, 
            config=peft_config, 
            is_trainable=False,
            adapter_name="default" 
        )
        model.eval()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        if os.path.exists(os.path.join(current_target_lora_path, "adapter_model.safetensors")):
            tgt_state = load_file(os.path.join(current_target_lora_path, "adapter_model.safetensors"), device="cpu")
        else:
            tgt_state = torch.load(os.path.join(current_target_lora_path, "adapter_model.bin"), map_location="cpu")

        dataloader, _ = data_builder.get_mixed_dataloader(task['datasets'])
        
        trainer_config = {
            'output_dir': task_output_dir,
            'model': {
                'transfer_ratio': task.get('transfer_ratio', 1.0), 
            },
            'training': task['training']
        }
        
        source_paths = task.get('source_lora_paths') 
        if not source_paths:
             source_paths = [task.get('source_lora_path')]

        print(f"Initializing Trainer with {len(source_paths)} Sources...")
        
        trainer = HeteroFusionTrainer(
            base_model=model,
            config=trainer_config,
            device=device,
            train_dataloader=dataloader,
            target_lora_state=tgt_state, 
            source_lora_paths=source_paths, 
            lpka_cache=global_lpka_cache
        )
        
        next_target_path = trainer.train()
        print(f"Task {task_name} Completed. New Base is: {next_target_path}")
        current_target_lora_path = next_target_path
        
        del trainer
        del model 
        torch.cuda.empty_cache()
        gc.collect()

    print("\n✅ All tasks in pipeline completed successfully!")
    print(f"Final Combined Adapter is located at: {current_target_lora_path}")

if __name__ == "__main__":
    fire.Fire(run_pipeline)
