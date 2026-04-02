import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainingArguments
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.model import load_tokenizer
from llamafactory.hparams import get_infer_args
import numpy as np

def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
    }

class MixDatasetBuilder:
    def __init__(self, global_config, base_model_path):
        self.data_config = global_config['data_global']
        self.base_model_path = base_model_path
        
        print(f"Initializing Tokenizer for Base: {base_model_path}")
        self.infer_args_dict = dict(
            model_name_or_path=base_model_path,
            dataset_dir=self.data_config.get('dataset_dir', 'data'),
            template=self.data_config['template'],
            cutoff_len=self.data_config.get('cutoff_len', 1024),
            dataset="dummy"
        )
        model_args, data_args, _, _ = get_infer_args(self.infer_args_dict)
        tokenizer_module = load_tokenizer(model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        if hasattr(self.template, "mm_plugin"): self.template.mm_plugin.expand_mm_tokens = False
        
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir", remove_unused_columns=False)

    def _load_single_dataset(self, dataset_name):
        current_data_args = self.data_args
        current_data_args.dataset = [dataset_name]
        
        dataset_module = get_dataset(
            self.template, 
            self.model_args, 
            current_data_args, 
            self.training_args, 
            stage="sft", 
            tokenizer=self.tokenizer
        )
        return dataset_module["train_dataset"]

    def get_mixed_dataloader(self, task_datasets_config):
        main_datasets = []
        replay_configs = []

        for ds_cfg in task_datasets_config:
            if ds_cfg.get('type') == 'main':
                main_datasets.append(ds_cfg['name'])
            else:
                replay_configs.append(ds_cfg)
        
        if not main_datasets:
            raise ValueError("No 'main' dataset specified in task config!")

        loaded_mains = []
        print(f"Loading Main Datasets: {main_datasets}")
        for name in main_datasets:
            loaded_mains.append(self._load_single_dataset(name))
        
        final_main_dataset = ConcatDataset(loaded_mains) if len(loaded_mains) > 1 else loaded_mains[0]
        main_count = len(final_main_dataset)
        print(f"Total Main Samples: {main_count}")

        datasets_to_concat = [final_main_dataset]
        
        for rep_cfg in replay_configs:
            name = rep_cfg['name']
            raw_ratio = rep_cfg.get('ratio', 0.1)
            
            full_replay_ds = self._load_single_dataset(name)
            ds_len = len(full_replay_ds)

            if isinstance(raw_ratio, str) and raw_ratio.lower() == 'all':
                print(f"Loading Replay Dataset: {name} (Mode: ALL, Count: {ds_len})")
                datasets_to_concat.append(full_replay_ds)
                continue

            ratio = float(raw_ratio)
            target_count = int(main_count * ratio)
            print(f"Loading Replay Dataset: {name} (Target Ratio: {ratio}, Count: {target_count})")
            
            if ds_len > target_count:
                indices = np.random.choice(ds_len, target_count, replace=False)
                subset_replay = Subset(full_replay_ds, indices)
                datasets_to_concat.append(subset_replay)
            else:
                print(f"Warning: Replay dataset {name} has fewer samples ({ds_len}) than requested ({target_count}). Using all.")
                datasets_to_concat.append(full_replay_ds)

        final_dataset = ConcatDataset(datasets_to_concat)
        print(f"Final Mixed Dataset Size: {len(final_dataset)}")

        dataloader = DataLoader(
            final_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=True, 
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer),
            num_workers=self.data_config.get('num_workers', 4),
            pin_memory=True
        )
        return dataloader, self.tokenizer