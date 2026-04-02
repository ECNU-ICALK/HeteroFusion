import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import re
from safetensors.torch import load_file, save_file
from transformers import get_cosine_schedule_with_warmup

# Import the topology-aligned transfer network and RDM regularizer.
from src.model import HeteroFusionTransferNet
from src.losses import compute_rdm_reg

class WeightPatcher:
    def __init__(self, module_map, new_weights_dict):
        self.module_map = module_map
        self.new_weights_dict = new_weights_dict
        self.backup = {}

    def __enter__(self):
        for name, module in self.module_map.items():
            if name in self.new_weights_dict:
                if hasattr(module.lora_B, 'default'): target_layer = module.lora_B.default
                else: target_layer = module.lora_B
                self.backup[name] = target_layer.weight
                del target_layer.weight 
                target_layer.weight = self.new_weights_dict[name].to(self.backup[name].device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.module_map.items():
            if name in self.backup:
                if hasattr(module.lora_B, 'default'): target_layer = module.lora_B.default
                else: target_layer = module.lora_B
                del target_layer.weight
                target_layer.weight = self.backup[name]
        self.backup.clear()

class HeteroFusionTrainer:
    def __init__(self, base_model, config, device, train_dataloader, target_lora_state, source_lora_paths, lpka_cache=None):
        self.base_model = base_model
        self.config = config
        self.device = device
        self.dataloader = train_dataloader
        self.lpka_cache = lpka_cache if lpka_cache is not None else {}
        
        self.block_size = 4096 
        self.embed_dim = int(config['training'].get('embed_dim', 1024))
        self.num_heads = int(config['training'].get('num_heads', 8))
        self.max_pos_embeddings = int(config['training'].get('max_position_embeddings', 4096))
        
        # --- 获取分布匹配与稀疏门控参数 ---
        self.mu_gate = float(config['training'].get('mu_gate', -0.5))
        self.lambda_reg = float(config['training'].get('lambda_reg', 0.1))
        self.mu_target = float(config['training'].get('mu_target', -1.0))
        self.sigma_target = float(config['training'].get('sigma_target', 1.0))
        self.num_projections = int(config['training'].get('num_projections', 1024))
        
        print(
            f"⚙️  HeteroFusion Pipeline: BlockSize={self.block_size}, "
            f"SVD-guided sparse gate ON (mu={self.mu_gate}), RDM ON"
        )

        print("Loading Adapter Weights...")
        self.tgt_state = target_lora_state 
        if isinstance(source_lora_paths, str): source_lora_paths = [source_lora_paths]
        self.src_states_list = [self._load_adapter_state(p) for p in source_lora_paths]
        
        raw_ratio = config['model'].get('transfer_ratio', 1.0)
        self.transfer_ratio = self._parse_ratio(raw_ratio)

        self.groups = self._group_modules_by_block(self.tgt_state, self.src_states_list, ratio=self.transfer_ratio)
        
        self.param_groups = []
        self.active_groups = {} 
        group_name_prefix = config['training'].get(
            'fusion_group_name',
            config['training'].get('lpka_group_name', 'heterofusion_transfer')
        )

        for group_id, group in self.groups.items():
            if not group['modules']: continue
            module_type, rank = group_id
            cache_key = f"{group_name_prefix}_{module_type}_{rank}"

            if cache_key in self.lpka_cache:
                print(f"♻️  Reusing transfer net for [{module_type}] R={rank}")
                hypernet = self.lpka_cache[cache_key].to(device)
            else:
                print(f"🆕 Init HeteroFusion transfer net [{module_type}]: Rank={rank}")
                hypernet = HeteroFusionTransferNet(
                    rank=rank,
                    block_size=self.block_size,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    max_pos_embeddings=self.max_pos_embeddings,
                    mu_gate=self.mu_gate
                ).to(device)
                self.lpka_cache[cache_key] = hypernet

            alpha = nn.Parameter(torch.tensor(float(config['training'].get('alpha_init', 0.1)), device=device))
            group['hypernet'] = hypernet
            group['alpha'] = alpha
            self.param_groups.extend(list(hypernet.parameters()) + [alpha])
            self.active_groups[group_id] = group

        self.optimizer = torch.optim.AdamW(self.param_groups, lr=float(config['training']['lr']))
        
        accum_steps = int(config['training'].get('gradient_accumulation_steps', 8))
        total_raw_steps = len(self.dataloader) * self.config['training']['num_epochs']
        self.total_optim_steps = total_raw_steps // accum_steps
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=int(self.total_optim_steps * 0.05),
            num_training_steps=self.total_optim_steps
        )
        
        self._prepare_block_inputs()

    def _load_adapter_state(self, path):
        safe_path = os.path.join(path, "adapter_model.safetensors")
        bin_path = os.path.join(path, "adapter_model.bin")
        if os.path.exists(safe_path): return load_file(safe_path, device="cpu")
        elif os.path.exists(bin_path): return torch.load(bin_path, map_location="cpu")
        else: raise FileNotFoundError(f"No adapter found in {path}")

    def _parse_ratio(self, r):
        if isinstance(r, str) and '/' in r:
            a, b = r.split('/')
            return float(a) / float(b)
        return float(r)

    def _tensor_to_blocks(self, tensor, block_size):
        rows, rank = tensor.shape
        num_blocks = (rows + block_size - 1) // block_size
        target_len = num_blocks * block_size
        if target_len > rows:
            pad_len = target_len - rows
            tensor = F.pad(tensor, (0, 0, 0, pad_len), "constant", 0)
        return tensor.view(num_blocks, block_size, rank)

    def _compute_svd_s(self, tensor_blocks):
        tensor_blocks = tensor_blocks.float()
        try:
            S = torch.linalg.svdvals(tensor_blocks)
        except RuntimeError:
            S = torch.linalg.svdvals(tensor_blocks + 1e-6)
        return S

    def _prepare_block_inputs(self):
        for key, group in self.active_groups.items():
            all_tgt_flat, all_tgt_s, all_src_flat, all_src_s = [], [], [], []
            group['reconstruct_info'] = [] 
            
            for mod_data in group['modules']:
                tB = mod_data['tB'].to(self.device)
                tA = mod_data['tA'].to(self.device)
                
                tA_transposed = tA.t() 
                tB_blocks = self._tensor_to_blocks(tB, self.block_size)
                tA_blocks = self._tensor_to_blocks(tA_transposed, self.block_size)
                
                t_ctx_blocks = torch.cat([tA_blocks, tB_blocks], dim=0)
                all_tgt_s.append(self._compute_svd_s(t_ctx_blocks))
                all_tgt_flat.append(t_ctx_blocks.view(t_ctx_blocks.shape[0], -1))
                
                curr_src_flat_list, curr_src_s_list = [], []
                for sA, sB in zip(mod_data['src_tensors_A'], mod_data['src_tensors_B']):
                    sA, sB = sA.to(self.device), sB.to(self.device)
                    sA_transposed = sA.t()
                    sA_blocks = self._tensor_to_blocks(sA_transposed, self.block_size)
                    sB_blocks = self._tensor_to_blocks(sB, self.block_size)
                    s_ctx_blocks = torch.cat([sA_blocks, sB_blocks], dim=0)
                    curr_src_s_list.append(self._compute_svd_s(s_ctx_blocks))
                    curr_src_flat_list.append(s_ctx_blocks.view(s_ctx_blocks.shape[0], -1))
                
                all_src_flat.append(torch.cat(curr_src_flat_list, dim=0))
                all_src_s.append(torch.cat(curr_src_s_list, dim=0))
                
                group['reconstruct_info'].append({
                    'orig_rows': tB.shape[0],
                    'num_blocks_B': tB_blocks.shape[0],
                    'num_blocks_A': tA_blocks.shape[0] 
                })

            try:
                group['inputs'] = {
                    'tgt_flat': torch.stack(all_tgt_flat).to(self.device),
                    'tgt_s': torch.stack(all_tgt_s).to(self.device),
                    'src_flat': torch.stack(all_src_flat).to(self.device),
                    'src_s': torch.stack(all_src_s).to(self.device)
                }
                group['batched'] = True
            except RuntimeError:
                group['inputs'] = {
                    'tgt_flat': [t.to(self.device) for t in all_tgt_flat],
                    'tgt_s': [t.to(self.device) for t in all_tgt_s],
                    'src_flat': [s.to(self.device) for s in all_src_flat],
                    'src_s': [s.to(self.device) for s in all_src_s]
                }
                group['batched'] = False
            torch.cuda.empty_cache()

    def _group_modules_by_block(self, tgt_state, src_states_list, ratio=1.0):
        groups = {}
        tgt_keys = [k for k in tgt_state.keys() if "lora_B" in k]
        
        def get_max_layer(keys):
            indices = [int(re.search(r'layers\.(\d+)\.', k).group(1)) for k in keys if re.search(r'layers\.(\d+)\.', k)]
            return max(indices) + 1 if indices else 0 
        
        tgt_layer_count = get_max_layer(tgt_keys)
        # Build per-source tail-alignment plan so heterogeneous source backbones
        # (e.g., Qwen + Mistral) can be aligned independently.
        src_align_plans = []
        for s_state in src_states_list:
            src_layer_count = get_max_layer(s_state.keys())
            layer_offset = tgt_layer_count - src_layer_count
            valid_layer_pairs = [
                (t_idx, t_idx - layer_offset)
                for t_idx in range(tgt_layer_count)
                if 0 <= t_idx - layer_offset < src_layer_count
            ]
            keep_count = max(1, int(len(valid_layer_pairs) * ratio))
            valid_tgt_indices = set(p[0] for p in valid_layer_pairs[-keep_count:])
            src_align_plans.append({
                'state': s_state,
                'layer_offset': layer_offset,
                'valid_tgt_indices': valid_tgt_indices,
            })

        for tgt_key_B in tgt_keys:
            match = re.search(r'layers\.(\d+)\.', tgt_key_B)
            if not match: continue
            tgt_layer_idx = int(match.group(1))
            tgt_splitter = f"layers.{tgt_layer_idx}."
            suffix = tgt_key_B.split(tgt_splitter)[-1]
            
            layer_src_A, layer_src_B = [], []
            for plan in src_align_plans:
                if tgt_layer_idx not in plan['valid_tgt_indices']:
                    continue

                s_state = plan['state']
                src_layer_idx = tgt_layer_idx - plan['layer_offset']
                src_search_pattern = f"layers.{src_layer_idx}.{suffix}"
                src_key_B = next((k for k in s_state.keys() if src_search_pattern in k), None)
                if src_key_B:
                    src_key_A = src_key_B.replace("lora_B", "lora_A")
                    if src_key_A in s_state:
                        layer_src_B.append(s_state[src_key_B].float())
                        layer_src_A.append(s_state[src_key_A].float())
            
            if not layer_src_B: continue 
            
            tB = tgt_state[tgt_key_B].float()
            tgt_key_A = tgt_key_B.replace("lora_B", "lora_A")
            tA_key = next((k for k in tgt_state if tgt_key_A in k), None)
            if not tA_key: continue
            tA = tgt_state[tA_key].float()

            parts = tgt_key_B.split('.')
            module_type = parts[parts.index("lora_B")-1] if "lora_B" in parts else "unknown"
            rank = tB.shape[1]
            group_key = (module_type, rank)
            
            if group_key not in groups: groups[group_key] = {'modules': []}

            found_module = self._get_module_from_root(self.base_model, tgt_key_B)
            if found_module:
                groups[group_key]['modules'].append({
                    'name': self._clean_name(tgt_key_B),
                    'tA': tA, 'tB': tB, 
                    'src_tensors_A': layer_src_A, 'src_tensors_B': layer_src_B,
                    'ref_module': found_module
                })
        return groups

    def _get_module_from_root(self, root_model, path_str):
        clean_path = path_str.replace(".lora_B.weight", "").replace(".default.weight", "")
        parts = clean_path.split('.')
        candidates = [parts, parts[1:] if len(parts)>1 else [], parts[2:] if len(parts)>2 else [], parts[3:] if len(parts)>3 else []]
        for candidate_parts in candidates:
            if not candidate_parts: continue
            curr = root_model
            found = True
            for part in candidate_parts:
                if hasattr(curr, part): curr = getattr(curr, part)
                else: found = False; break
            if found: return curr
        return None

    def _clean_name(self, key):
        key = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        key = key.replace(".lora_A.default.weight", "").replace(".lora_B.default.weight", "")
        key = key.replace("base_model.model.", "")
        while "base_model.model." in key: key = key.replace("base_model.model.", "")
        return key

    def train(self):
        torch.cuda.empty_cache()
        accum_steps = self.config['training'].get('gradient_accumulation_steps', 8)
        print(f"🚀 Starting HeteroFusion training... (RDM weight: {self.lambda_reg})")
        
        total_steps = len(self.dataloader) * self.config['training']['num_epochs']
        pbar = tqdm(range(total_steps))
        self.optimizer.zero_grad()
        
        for epoch in range(self.config['training']['num_epochs']):
            for step, batch in enumerate(self.dataloader):
                model_device = self.base_model.device
                batch = {k: v.to(model_device) for k, v in batch.items()}
                
                full_patch_dict = {}
                module_map_ref = {}
                total_rdm_loss = 0.0 # 记录每一步分布匹配 Loss 
                
                for group_id, group in self.active_groups.items():
                    hypernet = group['hypernet']
                    inputs = group['inputs']
                    
                    if group.get('batched', False):
                        delta_flat, _, tgt_emb, src_emb = hypernet(
                            inputs['tgt_flat'], inputs['tgt_s'], inputs['src_flat'], inputs['src_s']
                        )
                        rdm_t = compute_rdm_reg(tgt_emb, self.mu_target, self.sigma_target, self.num_projections)
                        rdm_s = compute_rdm_reg(src_emb, self.mu_target, self.sigma_target, self.num_projections)
                        total_rdm_loss += (rdm_t + rdm_s)
                    else:
                        outs, per_item_rdm = [], []
                        for tf, ts, sf, ss in zip(inputs['tgt_flat'], inputs['tgt_s'], inputs['src_flat'], inputs['src_s']):
                            d, _, te, se = hypernet(tf.unsqueeze(0), ts.unsqueeze(0), sf.unsqueeze(0), ss.unsqueeze(0))
                            outs.append(d.squeeze(0))
                            # Variable source counts can make sequence lengths differ across items.
                            # Compute per-item RDM and average, instead of concatenating embeddings.
                            rdm_t_i = compute_rdm_reg(te, self.mu_target, self.sigma_target, self.num_projections)
                            rdm_s_i = compute_rdm_reg(se, self.mu_target, self.sigma_target, self.num_projections)
                            per_item_rdm.append(rdm_t_i + rdm_s_i)
                        delta_flat = outs

                        if per_item_rdm:
                            total_rdm_loss += torch.stack(per_item_rdm).mean()

                    for i, mod_data in enumerate(group['modules']):
                        clean_name = mod_data['name'] 
                        ref_module = mod_data['ref_module']
                        info = group['reconstruct_info'][i]
                        
                        start_idx = info['num_blocks_A']
                        count = info['num_blocks_B']
                        delta_B = delta_flat[i][start_idx : start_idx + count]
                        
                        curr_delta = delta_B.view(-1, self.block_size, hypernet.rank)
                        curr_delta = curr_delta.view(-1, hypernet.rank)
                        curr_delta = curr_delta[:info['orig_rows'], :]
                        
                        if hasattr(ref_module.lora_B, 'default'): orig_B = ref_module.lora_B.default.weight
                        else: orig_B = ref_module.lora_B.weight
                        
                        if group['alpha'].dtype != curr_delta.dtype: group['alpha'].data = group['alpha'].data.to(curr_delta.dtype)
                        
                        update_term = group['alpha'] * curr_delta
                        full_patch_dict[clean_name] = orig_B + update_term.to(orig_B.device)
                        module_map_ref[clean_name] = ref_module
                
                if full_patch_dict:
                    with WeightPatcher(module_map_ref, full_patch_dict):
                        outputs = self.base_model(**batch)
                        lm_loss = outputs.loss
                        
                        avg_rdm_loss = total_rdm_loss / max(1, len(self.active_groups))
                        loss = (lm_loss + self.lambda_reg * avg_rdm_loss) / accum_steps
                        
                    loss.backward()
                    
                    if (step + 1) % accum_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        pbar.update(accum_steps)
                        
                        first_group = next(iter(self.active_groups.values()))
                        alpha_val = first_group['alpha'].item()
                        
                        pbar.set_postfix(
                            LM=f"{lm_loss.item():.4f}", 
                            RDM=f"{(avg_rdm_loss.item() if isinstance(avg_rdm_loss, torch.Tensor) else avg_rdm_loss):.4f}",
                            alpha=f"{alpha_val:.2e}"
                        )
                else: break
        return self._save_merged_lora()

    def _save_merged_lora(self):
        print("🚀 Starting fused adapter export...")
        save_dir = os.path.join(self.config['output_dir'], "merged_lora")
        os.makedirs(save_dir, exist_ok=True)
        final_state_dict = {}
        saved_modules = set()
        
        with torch.no_grad():
            for group_id, group in self.active_groups.items():
                hypernet = group['hypernet']
                inputs = group['inputs']
                
                if group.get('batched', False):
                    delta_flat, _, _, _ = hypernet(inputs['tgt_flat'], inputs['tgt_s'], inputs['src_flat'], inputs['src_s'])
                else:
                    outs = []
                    for tf, ts, sf, ss in zip(inputs['tgt_flat'], inputs['tgt_s'], inputs['src_flat'], inputs['src_s']):
                        d, _, _, _ = hypernet(tf.unsqueeze(0), ts.unsqueeze(0), sf.unsqueeze(0), ss.unsqueeze(0))
                        outs.append(d.squeeze(0))
                    delta_flat = outs

                for i, mod_data in enumerate(group['modules']):
                    clean_name = mod_data['name']
                    saved_modules.add(clean_name)
                    info = group['reconstruct_info'][i]
                    
                    key_A = f"base_model.model.{clean_name}.lora_A.weight"
                    key_B = f"base_model.model.{clean_name}.lora_B.weight"
                    
                    final_state_dict[key_A] = mod_data['tA'].clone().cpu()
                    
                    target_dtype = mod_data['tB'].dtype
                    orig_B = mod_data['tB'].to(self.device)
                    
                    start_idx = info['num_blocks_A']
                    count = info['num_blocks_B']
                    delta_B = delta_flat[i][start_idx : start_idx + count]
                    
                    curr_delta = delta_B.view(-1, self.block_size, hypernet.rank)
                    curr_delta = curr_delta.view(-1, hypernet.rank)
                    curr_delta = curr_delta[:info['orig_rows'], :]
                    
                    final_B = orig_B + group['alpha'] * curr_delta
                    final_state_dict[key_B] = final_B.clone().to(target_dtype).cpu()
            
            for k, v in self.tgt_state.items():
                clean_name = self._clean_name(k)
                if clean_name in saved_modules: continue
                final_state_dict[k] = v.clone().cpu()
        
        save_file(final_state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
        if hasattr(self.base_model, "peft_config"):
            self.base_model.peft_config['default'].save_pretrained(save_dir)
        print(f"✅ Fused LoRA exported successfully to: {save_dir}")
        return save_dir


# Backward-compatible alias for older scripts/imports.
HeterogeneousTrainer = HeteroFusionTrainer
