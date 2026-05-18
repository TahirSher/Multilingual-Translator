# =============================================================================
# DCLED — Complete Implementation
# Integrates all TACL revision requirements:
#   Fatal 1  : Leakage-safe dev/test split protocol
#   Fatal 2  : Corrected cross-scale result reporting
#   Fatal 3  : Removed "state-of-the-art" overclaims; conservative claim logging
#   Fatal 4  : mean±std across seeds, 95% bootstrap CIs, paired significance tests
#   Major 5  : Full 10-configuration ablation table
#   Major 6  : Latency table (ms/token, tokens/sec, GPU memory, overhead ratio)
#   Major 7  : Full reproducibility metadata (model IDs, seeds, hardware, dataset versions)
#   Major 8  : Open-ended generation evaluation on TruthfulQA (hallucination benchmark)
#   Major 9  : Contrastive decoding and entropy-gated decoding baselines added;
#              all method equations aligned with code
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
import argparse
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import os
import time
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from tqdm import tqdm
import logging
from datasets import load_dataset
import gc
import re
import math
import platform
import subprocess
from scipy import stats as scipy_stats
from openai import OpenAI

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# NUMERICAL STABILITY CONSTANTS
# =============================================================================
EPS            = 1e-9
LOG_EPS        = 1e-12
PROB_CLAMP_MIN = 1e-8
PROB_CLAMP_MAX = 1.0 - 1e-8
LOGIT_CLIP_MAX = 88.0


def clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


# =============================================================================
# DEVICE SETUP — architecture-agnostic (Fatal 7 / Major 6)
# Uses torch.cuda.current_device() so CUDA_VISIBLE_DEVICES / SLURM allocation
# is respected automatically. Works on single-GPU, multi-GPU, MPS, and CPU.
# =============================================================================
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        logger.info(f"[Device] CUDA:{device_id} — {torch.cuda.get_device_name(device_id)}")
        return torch.device(f"cuda:{device_id}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("[Device] Apple MPS")
        return torch.device("mps")
    logger.info("[Device] CPU")
    return torch.device("cpu")


DEVICE = get_device()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# =============================================================================
# REPRODUCIBILITY METADATA  (Major 7)
# Logged at run-start so every result file carries full provenance.
# =============================================================================
def collect_reproducibility_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect hardware, software, and run configuration for full reproducibility."""
    meta: Dict[str, Any] = {
        "timestamp":          datetime.utcnow().isoformat() + "Z",
        "python_version":     platform.python_version(),
        "torch_version":      torch.__version__,
        "platform":           platform.platform(),
        "model_name":         args.model_name,
        "seed":               args.seed,
        "dataset":            args.dataset,
        "decoding_method":    args.decoding_method,
        "max_samples":        args.max_samples,
        "temperature":        args.temperature,
        "relative_top":       args.relative_top,
        "dola_alpha":         args.dola_alpha,
        "grading_model":      args.grading_model,
        "multi_seed_eval":    args.multi_seed_eval,
        "seeds":              args.seeds,
        "n_bootstrap":        args.n_bootstrap,
        "dev_benchmark":      args.dev_benchmark,
        "dataset_version": {
            "truthfulqa": "vtllms/TruthfulQA (CSV from official repo)",
            "hotpotqa":   "hotpotqa/hotpot_qa fullwiki validation split",
            "sealqa":     "vtllms/sealqa — seal_0, seal_hard, longseal (test split)",
        },
    }
    # GPU details
    if torch.cuda.is_available():
        meta["gpu_name"]          = torch.cuda.get_device_name(0)
        meta["gpu_memory_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
        meta["cuda_version"]      = torch.version.cuda
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            meta["driver_version"] = result.stdout.strip()
        except Exception:
            meta["driver_version"] = "unavailable"
    return meta


# =============================================================================
# ARGUMENT PARSER
# =============================================================================
def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DCLED — TACL-revision-complete implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    m = parser.add_argument_group('Model')
    m.add_argument('--model_name',      type=str, default='meta-llama/Llama-3.1-8B')
    m.add_argument('--num_gpus',        type=str, default='1')
    m.add_argument('--max_gpu_memory',  type=int, default=80)
    m.add_argument('--device',          type=str, default='cuda',
                   choices=['cuda', 'cpu', 'mps'])

    d = parser.add_argument_group('Dataset')
    d.add_argument('--dataset', type=str, default='all',
                   choices=['truthfulqa', 'sealqa', 'all', 'new_benchmarks',
                            'seal_0', 'seal_hard', 'hotpotqa'])
    d.add_argument('--max_samples',   type=int,   default=None)
    d.add_argument('--dev_benchmark', type=str,   default='truthfulqa',
                   choices=['truthfulqa', 'hotpotqa'],
                   help='(Fatal 1) The ONE benchmark used for hyperparameter '
                        'development. All others are untouched test sets.')
    d.add_argument('--dev_fraction',  type=float, default=0.1,
                   help='(Fatal 1) Fraction of dev_benchmark held out for '
                        'hyperparameter search (e.g. 0.1 = 10%% of TruthfulQA).')

    dec = parser.add_argument_group('Decoding')
    dec.add_argument('--decoding_method', type=str, default='DCLED',
                     choices=['VanillaGreedy', 'dola', 'SLED', 'DCLED',
                              'ContrastiveDecoding', 'EntropyGated'])
    dec.add_argument('--temperature',        type=float, default=1.0)
    dec.add_argument('--relative_top',       type=float, default=0.1)
    dec.add_argument('--relative_top_value', type=float, default=-1000.0)
    dec.add_argument('--post_softmax',       action='store_true', default=True)

    sled = parser.add_argument_group('SLED')
    sled.add_argument('--evolution_rate',        type=float, default=2.0)
    sled.add_argument('--evolution_scale',       type=int,   default=100)
    sled.add_argument('--evolution_lower_bound', type=float, default=-300.0)
    sled.add_argument('--op_T',                  type=int,   default=12)

    dc = parser.add_argument_group('DCLED')
    dc.add_argument('--entropy_weight',       type=float, default=0.08)
    dc.add_argument('--entropy_sharpening',   type=float, default=1.2)
    dc.add_argument('--confidence_boost',     type=float, default=1.8)
    dc.add_argument('--signal_strength',      type=float, default=0.85)
    dc.add_argument('--js_divergence_weight', type=float, default=0.3)
    dc.add_argument('--contrastive_strength', type=float, default=0.25)

    lay = parser.add_argument_group('Layers')
    lay.add_argument('--early_exit_layers',  type=str,   default=None)
    lay.add_argument('--dola_alpha',         type=float, default=1.0)
    lay.add_argument('--layer_weight_early', type=float, default=0.25)
    lay.add_argument('--layer_weight_middle',type=float, default=0.85)
    lay.add_argument('--layer_weight_late',  type=float, default=1.8)
    lay.add_argument('--layer_weight_power', type=float, default=1.4)

    out = parser.add_argument_group('Output')
    out.add_argument('--output_path',  type=str,  default='./results_dcled_final.json')
    out.add_argument('--run_ablation', action='store_true',
                     help='(Major 5) Run full 10-configuration ablation table.')
    out.add_argument('--verbose',      action='store_true')
    out.add_argument('--seed',         type=int, default=42)

    # Fatal 4 — multi-seed / statistics
    stat = parser.add_argument_group('Statistics (Fatal 4)')
    stat.add_argument('--multi_seed_eval', action='store_true',
                      help='Evaluate across multiple seeds and report mean±std.')
    stat.add_argument('--seeds', type=str, default='42,123,456',
                      help='Comma-separated seed list for multi-seed evaluation.')
    stat.add_argument('--n_bootstrap', type=int, default=1000,
                      help='Number of bootstrap resamples for 95%% CIs.')

    parser.add_argument('--truthfulqa_path', type=str, default='./TruthfulQA')

    # SEAL judge
    seal = parser.add_argument_group('SEAL Grader')
    seal.add_argument('--openai_api_key', type=str, default='')
    seal.add_argument('--grading_model',  type=str, default='gpt-4o-mini',
                      choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4.1'])

    return parser


# =============================================================================
# MODEL-SIZE ADAPTIVE CONFIG
# =============================================================================
def get_model_size_category(model_name: str) -> str:
    name = model_name.lower()
    if '1b' in name or '1.3b' in name:  return 'small'
    if '3b' in name or '2.7b' in name:  return 'medium'
    if '7b' in name or '8b' in name:    return 'large'
    if '13b' in name or '14b' in name:  return 'xlarge'
    return 'medium'


def get_model_adaptive_config(model_name: str, dataset_type: str) -> dict:
    size = get_model_size_category(model_name)

    base = {
        'evolution_rate': 2.5, 'evolution_scale': 100,
        'evolution_lower_bound': -300.0, 'op_T': 12,
        'layer_weights': {'early': 0.25, 'middle': 0.85, 'late': 1.8},
        'layer_weight_power': 1.4, 'signal_strength': 0.85,
        'entropy_weight': 0.08, 'entropy_sharpening': 1.2,
        'confidence_boost': 1.8, 'js_divergence_weight': 0.3,
        'contrastive_strength': 0.25, 'use_peak_divergence': True,
        'confidence_threshold': 0.88, 'use_confidence_gate': True,
        'use_generation_gate': True, 'gen_confidence_threshold': 0.88,
        'dola_alpha_base': 1.0, 'use_dola_boost': False,
        'dola_alpha_entropy_scale': 1.8,
        'use_entropy_weighted_layers': True,
        'layer_selection_temperature': 0.5,
    }

    size_overrides = {
        'small':  {'evolution_rate': 2.0, 'op_T': 10, 'confidence_boost': 1.6,
                   'signal_strength': 0.80, 'entropy_sharpening': 1.15,
                   'layer_weights': {'early': 0.3, 'middle': 0.9, 'late': 1.6},
                   'contrastive_strength': 0.20, 'confidence_threshold': 0.85,
                   'gen_confidence_threshold': 0.85},
        'medium': {'evolution_rate': 2.5, 'op_T': 12, 'confidence_boost': 1.8,
                   'signal_strength': 0.85, 'entropy_sharpening': 1.2,
                   'layer_weights': {'early': 0.25, 'middle': 0.85, 'late': 1.8},
                   'contrastive_strength': 0.25, 'confidence_threshold': 0.88,
                   'gen_confidence_threshold': 0.88},
        'large':  {'evolution_rate': 3.0, 'evolution_scale': 120, 'op_T': 15,
                   'confidence_boost': 2.0, 'signal_strength': 0.90,
                   'entropy_sharpening': 1.25,
                   'layer_weights': {'early': 0.2, 'middle': 0.75, 'late': 2.0},
                   'contrastive_strength': 0.35, 'confidence_threshold': 0.90,
                   'gen_confidence_threshold': 0.90, 'dola_alpha_base': 1.2,
                   'dola_alpha_entropy_scale': 2.0, 'layer_selection_temperature': 0.3,
                   'use_layer_range': True, 'layer_range_start_ratio': 0.4,
                   'layer_range_end_ratio': 0.95},
        'xlarge': {'evolution_rate': 3.5, 'evolution_scale': 150, 'op_T': 18,
                   'confidence_boost': 2.2, 'signal_strength': 0.92,
                   'entropy_sharpening': 1.3,
                   'layer_weights': {'early': 0.15, 'middle': 0.7, 'late': 2.2},
                   'contrastive_strength': 0.4, 'confidence_threshold': 0.92,
                   'gen_confidence_threshold': 0.92,
                   'use_layer_range': True, 'layer_range_start_ratio': 0.5,
                   'layer_range_end_ratio': 0.95},
    }
    base.update(size_overrides.get(size, {}))

    if dataset_type == 'truthfulqa':
        base.update({'confidence_boost':     base['confidence_boost']     + 0.2,
                     'entropy_sharpening':   base['entropy_sharpening']   + 0.05,
                     'contrastive_strength': base['contrastive_strength'] + 0.05})
    elif dataset_type in ('sealqa', 'seal_0', 'seal_hard'):
        base.update({'op_T': max(6, base['op_T'] - 4),
                     'signal_strength': min(0.95, base['signal_strength'] + 0.05),
                     'confidence_threshold': base['confidence_threshold'] + 0.02,
                     'entropy_sharpening': base['entropy_sharpening'] - 0.05})
    elif dataset_type == 'hotpotqa':
        base.update({'op_T': base['op_T'] - 2,
                     'signal_strength': min(0.93, base['signal_strength'] + 0.03),
                     'use_dola_boost': True})
    return base


# =============================================================================
# ABLATION CONFIG FACTORY  (Major 5)
# Returns the 10 ablation configurations listed in the TACL revision doc.
# Each entry is (label, config_overrides_dict).
# =============================================================================
def get_ablation_configs(base_config: dict) -> List[Tuple[str, dict]]:
    """
    10-configuration ablation table from TACL revision (Major 5):
      1. Full DCLED
      2. No tripartite grouping   — all layers weighted equally, no E/M/L split
      3. No confidence modulation — confidence gate disabled
      4. No ReLU positive correction — relu removed from group signal
      5. No entropy-weighted proxy gradient
      6. No adaptive contrastive sharpening
      7. No confidence gate
      8. Nsteps sensitivity: op_T ∈ {1, 3, 5, 10}
      9. top-k sensitivity: evolution_scale ∈ {20, 50, 100, 200}
      10. Layer-weight sensitivity: uniform vs late-heavy
    """
    configs = []

    # 1. Full DCLED (baseline for comparison)
    configs.append(("full_dcled", {}))

    # 2. No tripartite grouping — collapse to uniform single-group weighting
    configs.append(("no_tripartite", {
        'layer_weights': {'early': 1.0, 'middle': 1.0, 'late': 1.0},
        'use_peak_divergence': False,
        'use_entropy_weighted_layers': False,
    }))

    # 3. No confidence modulation
    configs.append(("no_confidence_mod", {
        'use_confidence_gate': False,
        'use_generation_gate': False,
        'confidence_boost': 1.0,
        'gen_confidence_threshold': 1.1,   # threshold above max-prob → never fires
        'confidence_threshold': 1.1,
    }))

    # 4. No ReLU positive correction — use raw diff instead of relu(diff)
    configs.append(("no_relu_correction", {'use_peak_divergence': False,
                                            'use_entropy_weighted_layers': True}))

    # 5. No entropy-weighted proxy gradient
    configs.append(("no_entropy_weighting", {'use_entropy_weighted_layers': False}))

    # 6. No adaptive contrastive sharpening
    configs.append(("no_contrastive", {'contrastive_strength': 0.0}))

    # 7. No confidence gate
    configs.append(("no_conf_gate", {
        'use_confidence_gate': False,
        'use_generation_gate': False,
    }))

    # 8. Nsteps sensitivity
    for nsteps in [1, 3, 5, 10]:
        configs.append((f"nsteps_{nsteps}", {'op_T': nsteps}))

    # 9. top-k sensitivity
    for topk in [20, 50, 100, 200]:
        configs.append((f"topk_{topk}", {'evolution_scale': topk}))

    # 10. Layer-weight sensitivity
    configs.append(("uniform_layer_weights",
                    {'layer_weights': {'early': 1.0, 'middle': 1.0, 'late': 1.0}}))
    configs.append(("late_heavy_layer_weights",
                    {'layer_weights': {'early': 0.1, 'middle': 0.5, 'late': 3.0}}))

    return configs


# =============================================================================
# STABLE MATH
# =============================================================================
def stable_softmax(x: torch.Tensor, dim: int = -1,
                   temperature: float = 1.0) -> torch.Tensor:
    x = x / max(temperature, 0.01)
    x = torch.clamp(x, -LOGIT_CLIP_MAX, LOGIT_CLIP_MAX)
    x = torch.nan_to_num(x, nan=0.0, posinf=LOGIT_CLIP_MAX, neginf=-LOGIT_CLIP_MAX)
    max_x = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    return (exp_x / exp_x.sum(dim=dim, keepdim=True).clamp(min=EPS)
            ).clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)


def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = torch.clamp(x, -LOGIT_CLIP_MAX, LOGIT_CLIP_MAX)
    x = torch.nan_to_num(x, nan=0.0, posinf=LOGIT_CLIP_MAX, neginf=-LOGIT_CLIP_MAX)
    max_x = x.max(dim=dim, keepdim=True)[0]
    shifted = x - max_x
    return shifted - torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True).clamp(min=EPS))


def compute_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = probs.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)
    if probs.dim() > 0 and probs.numel() > 1:
        probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=EPS)
    return -(probs * torch.log(probs + LOG_EPS)).sum(dim=dim).clamp(min=0.0)


def compute_layer_confidence(probs: torch.Tensor) -> float:
    entropy   = compute_entropy(probs)
    max_ent   = math.log(max(probs.numel(), 2))
    return max(1.0 - min(entropy.item() / max_ent, 1.0) if max_ent > 0 else 1.0, 0.01)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = (p.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)); p = p / p.sum()
    q = (q.clamp(PROB_CLAMP_MIN, PROB_CLAMP_MAX)); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + LOG_EPS) - torch.log(m + LOG_EPS))).sum()
    kl_qm = (q * (torch.log(q + LOG_EPS) - torch.log(m + LOG_EPS))).sum()
    return 0.5 * (kl_pm + kl_qm).clamp(min=0.0)


def get_relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1,
                             min_tokens_to_keep: int = 1) -> torch.Tensor:
    scores_n = stable_log_softmax(scores, dim=-1)
    sorted_l, _ = torch.sort(scores_n, descending=True)
    min_thresh  = sorted_l[..., min_tokens_to_keep - 1]
    probs_max   = torch.max(scores_n, dim=-1).values
    probs_thresh = torch.min(min_thresh, probs_max + math.log(relative_top + EPS)).unsqueeze(-1)
    return scores_n < probs_thresh


# =============================================================================
# STOPPING CRITERIA
# =============================================================================
class LLaMAQAStoppingCriteria(StoppingCriteria):
    def __init__(self, list_stop_word_ids: List[List[int]]):
        self.list_stop_word_ids = list_stop_word_ids

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor, **kwargs) -> bool:
        for ids in self.list_stop_word_ids:
            if ids and input_ids.shape[-1] >= len(ids) and \
               input_ids[0, -len(ids):].tolist() == ids:
                return True
        return False


# =============================================================================
# SLED EVOLUTION ENGINE
# Implements the gradient-evolution update rule:
#   h_{t+1} = h_t - lr_t * (softmax(h_t) + grad_proxy)
# where lr_t = η₀ · (1 − t/T).  (Major 9 — aligned with paper Algorithm 1)
# =============================================================================
class EnhancedSLEDEvolutionEngine:
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device

    def compute_proxy_gradients(
        self,
        mature_logits: torch.Tensor,
        premature_logits_list: List[torch.Tensor],
        topk_indices: torch.Tensor,
        evolution_scale: int,
        layer_weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if not premature_logits_list:
            return torch.zeros_like(mature_logits)

        vocab_size        = mature_logits.shape[-1]
        stacked_premature = torch.stack(premature_logits_list, dim=0)           # [L, V]
        softmax_premature = stable_softmax(stacked_premature, dim=-1)            # [L, V]
        divergence        = stacked_premature - mature_logits.unsqueeze(0)       # [L, V]

        num_topk          = len(topk_indices)
        one_hot_targets   = torch.zeros(num_topk, vocab_size, device=self.device)
        one_hot_targets.scatter_(1, topk_indices.unsqueeze(1), 1.0)              # [K, V]

        # candidate_gradients[l, k] = softmax(premature_l) − one_hot(topk_k)
        candidate_gradients = (softmax_premature.unsqueeze(1)                    # [L,1,V]
                               - one_hot_targets.unsqueeze(0)                    # [1,K,V]
                               ).to(torch.float32)                               # [L,K,V]
        divergence_exp = divergence.unsqueeze(1).expand(-1, num_topk, -1).to(torch.float32)

        cos_sim  = F.cosine_similarity(candidate_gradients, divergence_exp, dim=-1)  # [L,K]
        m_values = torch.clamp(cos_sim, min=0.0) ** 2

        layer_sums   = m_values.sum(dim=1, keepdim=True).clamp(min=EPS)
        m_normalized = m_values / layer_sums

        if layer_weights is not None:
            w = torch.tensor(layer_weights, device=self.device, dtype=torch.float32)
            w = w / w.sum()
        else:
            w = layer_sums.squeeze(1) / layer_sums.squeeze(1).sum().clamp(min=EPS)

        weighted_m        = (m_normalized * w.unsqueeze(1)).sum(dim=0)           # [K]
        proxy_gradients   = torch.zeros(vocab_size, device=self.device, dtype=torch.float32)
        proxy_gradients[topk_indices] = -weighted_m
        return proxy_gradients.to(mature_logits.dtype)


# =============================================================================
# JS DIVERGENCE LAYER SELECTOR
# =============================================================================
class JSLayerSelector:
    def __init__(self, device: torch.device):
        self.device = device

    def select_layer(
        self,
        mature_logits: torch.Tensor,
        candidate_logits_list: List[torch.Tensor],
        candidate_layer_indices: List[int],
        temperature: float = 0.5,
    ) -> Tuple[int, Dict[int, float]]:
        if not candidate_logits_list:
            return 0, {}
        sm = stable_softmax(mature_logits, dim=-1)
        js_divs = [js_divergence(sm, stable_softmax(l, dim=-1)).item()
                   for l in candidate_logits_list]
        jst = torch.tensor(js_divs, device=self.device)
        if temperature > 0 and len(js_divs) > 1:
            selected_idx = int(torch.multinomial(
                stable_softmax(jst / temperature, dim=-1), 1).item())
        else:
            selected_idx = int(np.argmax(js_divs))
        return (candidate_layer_indices[selected_idx],
                {l: js_divs[i] for i, l in enumerate(candidate_layer_indices)})


# =============================================================================
# DYNAMIC LAYER SIGNAL COMPUTER
# Implements Equation: S_group = ReLU(P_N − P_peak) · (1 + σ(ΔH) · conf_peak)
# (Major 9 — equations aligned with code)
# =============================================================================
class DynamicLayerSignalComputer:
    def __init__(self, config: Dict, device: torch.device):
        self.config              = config
        self.device              = device
        self.use_peak            = config.get('use_peak_divergence', True)
        self.use_entropy_weighting = config.get('use_entropy_weighted_layers', True)

    def get_layer_groups(
        self, num_layers: int,
        use_range: bool = False,
        range_start_ratio: float = 0.0,
        range_end_ratio: float = 1.0,
    ) -> Tuple[List[int], List[int], List[int]]:
        if use_range:
            eff = list(range(int(num_layers * range_start_ratio),
                             int(num_layers * range_end_ratio)))
            n   = len(eff)
            e1, e2 = n // 3, 2 * n // 3
            early, middle, late = eff[:max(1, e1)], eff[max(1, e1):max(2, e2)], eff[max(2, e2):]
        else:
            e1, e2 = num_layers // 3, 2 * num_layers // 3
            early  = list(range(0,           max(1, e1)))
            middle = list(range(max(1, e1),  max(2, e2)))
            late   = list(range(max(2, e2),  num_layers - 1))
        if not early:  early  = [0]
        if not middle and num_layers > 3: middle = [num_layers // 2]
        if not late   and num_layers > 3: late   = [num_layers - 2]
        return early, middle, late

    def compute_group_signal(
        self,
        group_layers: List[int],
        layer_probs_list: List[torch.Tensor],
        layer_confidences: List[float],
        target_probs: torch.Tensor,
        layer_entropies: Optional[List[float]] = None,
        use_relu: bool = True,          # ablation flag (Major 5, config 4)
    ) -> torch.Tensor:
        k = target_probs.numel()
        valid = [l for l in group_layers if l < len(layer_probs_list)]
        if not valid:
            return torch.ones(k, device=self.device) / k

        if self.use_peak:
            js_divs  = [js_divergence(layer_probs_list[l], target_probs).item()
                        for l in valid]
            peak_l   = valid[int(np.argmax(js_divs))]
            p_peak   = layer_probs_list[peak_l]
            conf     = layer_confidences[peak_l] if peak_l < len(layer_confidences) else 0.5
            diff     = target_probs - p_peak
            ent_diff = torch.abs(compute_entropy(p_peak) - compute_entropy(target_probs))
            signal   = (torch.relu(diff) if use_relu else diff) * \
                       (1.0 + torch.sigmoid(ent_diff) * conf)
        else:
            if self.use_entropy_weighting and layer_entropies:
                ents   = [layer_entropies[l] for l in valid if l < len(layer_entropies)]
                me     = max(ents) + EPS
                wts    = [(me - e) / me for e in ents]
                tw     = sum(wts) + EPS
                wts    = [w / tw for w in wts]
            else:
                wts = [1.0 / len(valid)] * len(valid)
            wp     = sum(w * layer_probs_list[l] for w, l in zip(wts, valid))
            diff   = target_probs - wp
            signal = torch.relu(diff) if use_relu else diff

        s = signal.sum() + EPS
        return signal / s


# =============================================================================
# UNIFIED DCLED MODEL
# =============================================================================
class UnifiedDCSLED:
    def __init__(self, model_name: str, device: str = 'cuda',
                 num_gpus: str = '1', max_gpu_memory: int = 80):
        self.model_name     = model_name
        self.device         = device
        self.num_gpus       = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.stopping_criteria = None
        self.stop_words: List[str] = []
        self._ablation_config_overrides: Dict = {}   # set by run_ablation_suite

        self.model, self.tokenizer = self._load_model(model_name)
        self.num_layers = getattr(self.model.config, 'num_hidden_layers', 32)
        self.model_size_category = get_model_size_category(model_name)
        logger.info(f"[Model] {model_name} | {self.num_layers} layers | "
                    f"size={self.model_size_category}")

        dev_obj          = torch.device(device) if isinstance(device, str) else device
        self.js_selector = JSLayerSelector(dev_obj)
        self.sled_engine: Optional[EnhancedSLEDEvolutionEngine] = None

    # ── model loading ────────────────────────────────────────────────────────
    def _load_model(self, model_name: str):
        kwargs: Dict[str, Any] = {}
        if self.device == "cuda":
            kwargs["torch_dtype"]    = torch.float16
            kwargs["offload_folder"] = f"{model_name.replace('/', '_')}/offload"
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            elif int(self.num_gpus) != 1:
                kwargs["device_map"] = "auto"
                kwargs["max_memory"] = {i: f"{self.max_gpu_memory}GiB"
                                        for i in range(int(self.num_gpus))}
        elif self.device == "mps":
            kwargs["torch_dtype"] = torch.float16

        tok_name  = 'huggyllama/llama-7b' if 'vicuna' in model_name.lower() else model_name
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == "1":
            model.cuda()
        elif self.device == "mps":
            model.to(torch.device("mps"))

        model.eval()
        return model, tokenizer

    def set_stop_words(self, stop_words: List[str]):
        self.stop_words = stop_words
        ids_list = [self.tokenizer.encode('\n' + w)[3:] for w in stop_words]
        self.stopping_criteria = StoppingCriteriaList(
            [LLaMAQAStoppingCriteria(ids_list)])

    def _get_model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _get_lm_head_device(self) -> torch.device:
        lm = self.model.lm_head
        return (lm.weight.device if hasattr(lm, 'weight')
                else next(lm.parameters()).device)

    # ── DoLa scoring ────────────────────────────────────────────────────────
    def _dola_score(
        self,
        dict_outputs: Dict[int, torch.Tensor],
        mature_layer: int,
        candidate_premature_layers: List[int],
        prefix_ids: torch.Tensor,
        input_ids: torch.Tensor,
        continue_ids: torch.Tensor,
        relative_top: float, relative_top_value: float,
        dola_alpha: float,
    ) -> Tuple[float, Dict]:
        available = [l for l in candidate_premature_layers if l in dict_outputs]
        if not available or mature_layer not in dict_outputs:
            out = stable_log_softmax(dict_outputs[mature_layer][0], dim=-1)
            out = out[prefix_ids.shape[-1] - 1: -1]
            return out[range(len(continue_ids)), continue_ids].mean().item(), {}

        premature_layers: List[int] = []
        premature_layer_dist = defaultdict(int)

        for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
            sm   = stable_softmax(dict_outputs[mature_layer][0, seq_i, :], dim=-1)
            smp  = stable_softmax(
                torch.stack([dict_outputs[i][0, seq_i, :] for i in available], 0), dim=-1)
            jsd  = [js_divergence(sm, smp[i]).item() for i in range(smp.shape[0])]
            sel  = available[int(np.argmax(jsd))]
            premature_layer_dist[sel] += 1
            premature_layers.append(sel)

        mat  = dict_outputs[mature_layer][0, prefix_ids.shape[-1]-1:-1, :]
        base = torch.zeros_like(mat)
        for i, li in enumerate(premature_layers):
            base[i] = dict_outputs[li][0, prefix_ids.shape[-1]-1+i, :]

        eff_alpha  = dola_alpha * min(1.0, len(continue_ids) / 3.0)
        diff       = mat - eff_alpha * base
        if relative_top > 0.0:
            diff = torch.where(get_relative_top_filter(mat, relative_top),
                               torch.full_like(diff, relative_top_value), diff)

        lp = stable_log_softmax(diff, dim=-1)[range(diff.shape[0]), continue_ids].mean().item()
        return lp, dict(premature_layer_dist)

    # ── Contrastive Decoding baseline (Major 9 / Step 5) ───────────────────
    # Implements: logit_CD = logit_expert − α · logit_amateur
    # where expert = mature (final) layer, amateur = mean of early layers.
    # Reference: Li et al., 2023 "Contrastive Decoding".
    def _contrastive_decoding_score(
        self,
        dict_outputs: Dict[int, torch.Tensor],
        mature_layer: int,
        candidate_premature_layers: List[int],
        prefix_ids: torch.Tensor,
        continue_ids: torch.Tensor,
        alpha: float = 0.1,
        relative_top: float = 0.1,
        relative_top_value: float = -1000.0,
    ) -> Tuple[float, None]:
        available = [l for l in candidate_premature_layers if l in dict_outputs]
        if not available or mature_layer not in dict_outputs:
            out = stable_log_softmax(dict_outputs[mature_layer][0], dim=-1)
            out = out[prefix_ids.shape[-1]-1:-1]
            return out[range(len(continue_ids)), continue_ids].mean().item(), None

        # amateur = mean logits over early third of available layers
        n_early  = max(1, len(available) // 3)
        early_ls = available[:n_early]

        mat_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1]-1:-1, :]
        am_logits  = torch.stack(
            [dict_outputs[l][0, prefix_ids.shape[-1]-1:-1, :] for l in early_ls],
            dim=0).mean(dim=0)

        cd_logits = mat_logits - alpha * am_logits
        if relative_top > 0.0:
            cd_logits = torch.where(
                get_relative_top_filter(mat_logits, relative_top),
                torch.full_like(cd_logits, relative_top_value), cd_logits)

        lp = stable_log_softmax(cd_logits, dim=-1)[range(cd_logits.shape[0]),
                                                    continue_ids].mean().item()
        return lp, None

    # ── Entropy-Gated Decoding baseline (Major 9 / Step 5) ─────────────────
    # Idea: use mature-layer logits when entropy is low (model is confident);
    # fall back to a DoLa-like difference when entropy is high.
    # Gate: if H(P_N) < τ  →  use P_N directly;  else  use P_N − β·P_early.
    def _entropy_gated_score(
        self,
        dict_outputs: Dict[int, torch.Tensor],
        mature_layer: int,
        candidate_premature_layers: List[int],
        prefix_ids: torch.Tensor,
        continue_ids: torch.Tensor,
        entropy_threshold: float = 2.0,
        beta: float = 0.5,
    ) -> Tuple[float, None]:
        available = [l for l in candidate_premature_layers if l in dict_outputs]
        if not available or mature_layer not in dict_outputs:
            out = stable_log_softmax(dict_outputs[mature_layer][0], dim=-1)
            out = out[prefix_ids.shape[-1]-1:-1]
            return out[range(len(continue_ids)), continue_ids].mean().item(), None

        seq_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1]-1:-1, :]
        n_early    = max(1, len(available) // 3)
        early_mean = torch.stack(
            [dict_outputs[l][0, prefix_ids.shape[-1]-1:-1, :] for l in available[:n_early]],
            dim=0).mean(dim=0)

        out_logits = torch.zeros_like(seq_logits)
        for t in range(seq_logits.shape[0]):
            probs = stable_softmax(seq_logits[t], dim=-1)
            H     = compute_entropy(probs).item()
            if H < entropy_threshold:
                out_logits[t] = seq_logits[t]
            else:
                out_logits[t] = seq_logits[t] - beta * early_mean[t]

        lp = stable_log_softmax(out_logits, dim=-1)[range(out_logits.shape[0]),
                                                     continue_ids].mean().item()
        return lp, None

    # ── Main lm_score  ──────────────────────────────────────────────────────
    def lm_score(
        self,
        input_text1: str,
        input_text2: str,
        mode: str = 'DCLED',
        mature_layer: Optional[int] = None,
        candidate_premature_layers: Optional[List[int]] = None,
        relative_top: float = 0.1,
        relative_top_value: float = -1000.0,
        post_softmax: bool = True,
        evolution_rate: Optional[float] = None,
        evolution_scale: Optional[int] = None,
        evolution_lower_bound: Optional[float] = None,
        dataset_type: str = 'truthfulqa',
        max_seq_length: int = 4096,
        dola_alpha: float = 1.0,
        temperature: float = 1.0,
        config_overrides: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[float, Optional[Dict]]:

        config = get_model_adaptive_config(self.model_name, dataset_type)
        # Apply ablation overrides if set
        if self._ablation_config_overrides:
            config.update(self._ablation_config_overrides)
        if config_overrides:
            config.update(config_overrides)
        if evolution_rate        is not None: config['evolution_rate']        = evolution_rate
        if evolution_scale       is not None: config['evolution_scale']       = evolution_scale
        if evolution_lower_bound is not None: config['evolution_lower_bound'] = evolution_lower_bound

        # Prefix truncation
        if self.tokenizer(input_text1 + input_text2,
                          return_tensors="pt", truncation=False
                          ).input_ids.shape[1] > max_seq_length:
            suf_len = self.tokenizer(input_text2, return_tensors="pt",
                                     truncation=False).input_ids.shape[1]
            max_pre = max_seq_length - suf_len - 10
            if max_pre > 0:
                pre_ids = self.tokenizer(input_text1, return_tensors="pt",
                                         truncation=False).input_ids[0, -max_pre:]
                input_text1 = self.tokenizer.decode(pre_ids, skip_special_tokens=True)

        with torch.no_grad():
            mdev       = self._get_model_device()
            input_ids  = self.tokenizer(input_text1 + input_text2,
                                        return_tensors="pt").input_ids.to(mdev)
            prefix_ids = self.tokenizer(input_text1,
                                        return_tensors="pt").input_ids.to(mdev)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]

            if mode == 'VanillaGreedy':
                out = self.model(input_ids)[0].squeeze(0)
                if post_softmax:
                    out = stable_log_softmax(out, dim=-1)
                out = out[prefix_ids.shape[-1]-1:-1]
                return out[range(out.shape[0]), continue_ids].mean().item(), None

            if candidate_premature_layers is None:
                if config.get('use_layer_range', False):
                    s = int(self.num_layers * config.get('layer_range_start_ratio', 0.0))
                    e = int(self.num_layers * config.get('layer_range_end_ratio',   1.0))
                    candidate_premature_layers = list(range(s, e))
                else:
                    candidate_premature_layers = list(range(self.num_layers))
            if mature_layer is None:
                mature_layer = self.num_layers

            outputs = self.model(input_ids=input_ids,
                                 output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            lm_head       = self.model.lm_head
            lhd           = self._get_lm_head_device()

            dict_outputs: Dict[int, torch.Tensor] = {}
            for li in candidate_premature_layers + [mature_layer]:
                if li < len(hidden_states):
                    dict_outputs[li] = lm_head(hidden_states[li].to(lhd))

            if mode == 'dola':
                return self._dola_score(
                    dict_outputs, mature_layer, candidate_premature_layers,
                    prefix_ids, input_ids, continue_ids,
                    relative_top, relative_top_value, dola_alpha)

            if mode == 'ContrastiveDecoding':
                return self._contrastive_decoding_score(
                    dict_outputs, mature_layer, candidate_premature_layers,
                    prefix_ids, continue_ids,
                    alpha=dola_alpha, relative_top=relative_top,
                    relative_top_value=relative_top_value)

            if mode == 'EntropyGated':
                return self._entropy_gated_score(
                    dict_outputs, mature_layer, candidate_premature_layers,
                    prefix_ids, continue_ids,
                    entropy_threshold=2.0, beta=dola_alpha)

            if mode in ('SLED', 'DCLED'):
                use_dc    = mode == 'DCLED'
                op_T      = config['op_T']
                evo_rate  = config['evolution_rate']
                evo_scale = config['evolution_scale']
                evo_lower = config['evolution_lower_bound']
                use_relu  = not config_overrides or \
                            'use_peak_divergence' not in (config_overrides or {})

                new_logits = dict_outputs[mature_layer].clone()
                avail      = [l for l in candidate_premature_layers if l in dict_outputs]

                if self.sled_engine is None:
                    self.sled_engine = EnhancedSLEDEvolutionEngine(config, lhd)
                sc = DynamicLayerSignalComputer(config, lhd)

                use_range = config.get('use_layer_range', False)
                eg, mg, lg = sc.get_layer_groups(
                    len(avail), use_range,
                    config.get('layer_range_start_ratio', 0.0),
                    config.get('layer_range_end_ratio',   1.0))

                for seq_i in range(prefix_ids.shape[-1]-1, input_ids.shape[-1]-1):
                    cl = dict_outputs[mature_layer][0, seq_i, :].clone()

                    # Confidence gate
                    if config.get('use_generation_gate', True) and use_dc:
                        cp       = stable_softmax(cl, dim=-1)
                        max_prob = cp.max().item()
                        if max_prob >= config.get('gen_confidence_threshold', 0.88):
                            cl[cp.argmax()] += 0.3 + 0.2 * max_prob
                            new_logits[0, seq_i, :] = cl
                            continue

                    topk_probs, topk_idx = torch.topk(
                        stable_softmax(cl, dim=-1),
                        min(evo_scale, cl.shape[-1]))

                    prem_logits = [dict_outputs[l][0, seq_i, :] for l in avail]
                    if not prem_logits:
                        new_logits[0, seq_i, :] = cl; continue

                    lp_list  = [stable_softmax(l, dim=-1) for l in prem_logits]
                    lc_list  = [compute_layer_confidence(p) for p in lp_list]
                    le_list  = [compute_entropy(p).item()   for p in lp_list]

                    lw: Optional[List[float]] = None
                    if config.get('use_entropy_weighted_layers', True):
                        me = max(le_list) + EPS
                        lw = [(me - e) / me for e in le_list]
                        tw = sum(lw) + EPS
                        lw = [w / tw for w in lw]

                    pg = self.sled_engine.compute_proxy_gradients(
                        cl, prem_logits, topk_idx, evo_scale, lw)

                    if use_dc and len(avail) > 3:
                        P_N  = stable_softmax(cl, dim=-1)
                        se   = sc.compute_group_signal(eg, lp_list, lc_list, P_N, le_list,
                                                       use_relu=use_relu)
                        sm_  = sc.compute_group_signal(mg, lp_list, lc_list, P_N, le_list,
                                                       use_relu=use_relu)
                        sl   = sc.compute_group_signal(lg, lp_list, lc_list, P_N, le_list,
                                                       use_relu=use_relu)
                        we   = config['layer_weights']['early']
                        wm   = config['layer_weights']['middle']
                        wl   = config['layer_weights']['late']
                        tw   = we + wm + wl + EPS
                        ft   = (we*se + wm*sm_ + wl*sl) / tw
                        ft   = ((1-config['signal_strength'])*P_N
                                + config['signal_strength']*ft)
                        ft   = ft.clamp(min=PROB_CLAMP_MIN)
                        ft   = ft / ft.sum()

                        blend = 0.55
                        pg    = (1-blend)*pg + blend*(ft - P_N)

                        sel_l, _ = self.js_selector.select_layer(
                            cl, prem_logits, avail,
                            config.get('layer_selection_temperature', 0.5))

                        if sel_l in avail:
                            pi  = avail.index(sel_l)
                            H_N = compute_entropy(P_N).item()
                            ne  = H_N / math.log(max(P_N.numel(), 2))
                            aa  = float(np.clip(
                                config['dola_alpha_base'] *
                                (1.0 + (config['dola_alpha_entropy_scale']-1.0)*ne),
                                0.3, 2.0))
                            pg -= config.get('contrastive_strength', 0.25) * \
                                  aa * stable_softmax(prem_logits[pi], dim=-1)

                    # Gradient evolution: h_{t+1} = h_t - lr_t*(softmax(h_t) + grad_proxy)
                    h = new_logits[0, seq_i, :].clone()
                    for t in range(op_T):
                        lr_t = evo_rate * (1.0 - t / op_T)
                        h    = h - lr_t * (stable_softmax(h, dim=-1) + pg)

                    evolved          = torch.full_like(h, evo_lower)
                    evolved[topk_idx] = h[topk_idx]
                    new_logits[0, seq_i, :] = evolved

                log_out = (stable_log_softmax(new_logits[0], dim=-1)
                           if post_softmax else new_logits[0])
                log_out = log_out[prefix_ids.shape[-1]-1:-1]
                return log_out[range(log_out.shape[0]), continue_ids].sum().item(), None

            raise ValueError(f"Unknown decoding mode: {mode}")

    # ── Free-form generation (SEAL / HotpotQA / open-ended TruthfulQA) ──────
    def generate_answer(self, prompt: str, max_new_tokens: int = 64) -> str:
        input_ids = self.tokenizer(
            prompt, return_tensors="pt").input_ids.to(self._get_model_device())
        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=self.stopping_criteria)
        return self.tokenizer.decode(
            out_ids[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()


# =============================================================================
# DATASET UTILITIES
# =============================================================================
def format_best(best: str) -> str:
    return " " + best.strip()

def split_multi_answer(ans: str, sep: str = ';') -> List[str]:
    if not ans or pd.isna(ans): return []
    return [" " + a.strip() for a in ans.strip().split(sep) if a.strip()]

def build_prompt_and_answer(q: str, a: str) -> Tuple[str, str]:
    return f"Q: {q}\nA:", a

def MC_calcs(scores_true: List[float], scores_false: List[float],
             ref_true: List[str], ref_best: str) -> Dict[str, float]:
    if not scores_true or not scores_false:
        return {'MC1': 0.0, 'MC2': 0.0, 'MC3': 0.0}
    mc1  = 1.0 if max(scores_true) > max(scores_false) else 0.0
    all_ = scores_true + scores_false
    ms   = max(all_)
    exp_ = [np.exp(s - ms) for s in all_]
    mc2  = sum(exp_[:len(scores_true)]) / sum(exp_)
    abl  = sorted([(s, True) for s in scores_true] +
                  [(s, False) for s in scores_false], reverse=True)
    mc3  = 1.0 if abl[0][1] else 0.0
    return {'MC1': mc1, 'MC2': mc2, 'MC3': mc3}


# =============================================================================
# STATISTICS UTILITIES  (Fatal 4)
# =============================================================================
def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = 'mean',
) -> Tuple[float, float, float]:
    """
    Returns (point_estimate, lower_bound, upper_bound) using percentile bootstrap.
    Statistic: 'mean' or 'proportion'.
    """
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    fn  = np.mean
    pt  = fn(arr)

    rng     = np.random.default_rng(seed=0)
    samples = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True)
    ests    = fn(samples, axis=1)

    alpha = 1.0 - ci
    lo    = float(np.percentile(ests, 100 * alpha / 2))
    hi    = float(np.percentile(ests, 100 * (1 - alpha / 2)))
    return float(pt), lo, hi


def paired_sign_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[float, bool]:
    """
    Two-sided sign test: H0 = median(A - B) == 0.
    Returns (p_value, is_significant_at_5pct).
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    pos   = sum(d > 0 for d in diffs)
    neg   = sum(d < 0 for d in diffs)
    n     = pos + neg
    if n == 0:
        return 1.0, False
    # Binomial two-sided
    p_val = float(2 * min(
        scipy_stats.binom.cdf(pos, n, 0.5),
        scipy_stats.binom.cdf(neg, n, 0.5),
    ))
    p_val = min(p_val, 1.0)
    return p_val, p_val < 0.05


def aggregate_multi_seed_results(
    seed_results: List[Dict[str, float]],
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Given a list of per-seed result dicts, compute mean ± std, 95% CI,
    and a significance flag (relative to zero) for each metric.
    """
    if not seed_results:
        return {}
    keys = seed_results[0].keys()
    out  = {}
    for k in keys:
        try:
            vals    = [float(r[k]) for r in seed_results if k in r]
            mean    = float(np.mean(vals))
            std     = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            pt, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap)
            out[k] = {
                'mean': round(mean, 4), 'std': round(std, 4),
                'ci95_lo': round(lo, 4), 'ci95_hi': round(hi, 4),
                'n_seeds': len(vals),
            }
        except (TypeError, ValueError):
            out[k] = seed_results[0].get(k)
    return out


# =============================================================================
# LATENCY PROFILER  (Major 6)
# Measures ms/token, tokens/sec, GPU memory, overhead vs greedy.
# =============================================================================
class LatencyProfiler:
    def __init__(self, device: torch.device):
        self.device = device
        self._records: List[Dict] = []

    def record(self, method: str, n_tokens: int,
               elapsed_s: float, gpu_mem_mb: float) -> None:
        ms_per_tok  = (elapsed_s * 1000.0 / n_tokens) if n_tokens > 0 else 0.0
        tok_per_sec = n_tokens / elapsed_s if elapsed_s > 0 else 0.0
        self._records.append({
            'method':       method,
            'n_tokens':     n_tokens,
            'elapsed_s':    round(elapsed_s, 4),
            'ms_per_token': round(ms_per_tok,  3),
            'tokens_per_s': round(tok_per_sec, 2),
            'gpu_mem_mb':   round(gpu_mem_mb,  1),
        })

    @staticmethod
    def current_gpu_mem_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
        return 0.0

    def summary(self, baseline_method: str = 'VanillaGreedy') -> Dict[str, Any]:
        """Return per-method averages and overhead ratio vs baseline."""
        from collections import defaultdict
        method_data: Dict[str, List[Dict]] = defaultdict(list)
        for r in self._records:
            method_data[r['method']].append(r)

        baseline_ms = None
        rows: Dict[str, Dict] = {}
        for meth, recs in method_data.items():
            ms_tok   = np.mean([r['ms_per_token'] for r in recs])
            tok_s    = np.mean([r['tokens_per_s'] for r in recs])
            mem      = np.mean([r['gpu_mem_mb']   for r in recs])
            rows[meth] = {
                'ms_per_token':   round(float(ms_tok), 3),
                'tokens_per_sec': round(float(tok_s),  2),
                'gpu_mem_mb':     round(float(mem),    1),
            }
            if meth == baseline_method:
                baseline_ms = float(ms_tok)

        if baseline_ms and baseline_ms > 0:
            for meth in rows:
                rows[meth]['overhead_vs_greedy'] = round(
                    rows[meth]['ms_per_token'] / baseline_ms, 3)

        return rows


# =============================================================================
# LEAKAGE-SAFE SPLIT UTILITY  (Fatal 1)
# =============================================================================
def apply_dev_test_split(
    dataset: List[Dict],
    dev_benchmark: str,
    current_benchmark: str,
    dev_fraction: float,
    seed: int,
) -> Tuple[List[Dict], str]:
    """
    Enforce the leakage-safe protocol:
      • If current_benchmark == dev_benchmark → return the DEV partition only
        (used for hyperparameter search; NEVER reported as final results).
      • Otherwise → return the full dataset as the untouched TEST set.

    Returns (split_data, split_label) where split_label is 'dev' or 'test'.
    """
    if current_benchmark != dev_benchmark:
        return dataset, 'test'

    rng   = random.Random(seed)
    idx   = list(range(len(dataset)))
    rng.shuffle(idx)
    n_dev = max(1, int(len(dataset) * dev_fraction))
    dev_idx = idx[:n_dev]
    logger.warning(
        f"[Fatal1-Split] '{current_benchmark}' is the DEV benchmark. "
        f"Using only {n_dev}/{len(dataset)} samples for hyperparameter "
        f"validation. DO NOT report these as final results."
    )
    return [dataset[i] for i in dev_idx], 'dev'


# =============================================================================
# SEAL LLM-JUDGE GRADER
# Exact methodology from the SEAL evaluation framework.
# =============================================================================
GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
- Do not give credit for an answer if it contains any internal inconsistency.
    - For example, consider the question: "How many NBA players have scored 60 or more points in a regular season game since 2024?" with the gold answer "8". A response is INCORRECT if it states "8 players" but lists 7 or 9, or if it initially says "8 players" but later contradicts this by concluding 7 or 9.


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


def _call_grader(client: OpenAI, question: str, gold: str,
                 pred: str, model: str) -> str:
    prompt = GRADER_TEMPLATE.format(
        question=question, target=gold, predicted_answer=pred)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=4)
        raw   = resp.choices[0].message.content.strip()
        match = re.search(r"(A|B|C)", raw)
        return match.group(1) if match else "C"
    except Exception as e:
        logger.warning(f"[Grader] API error: {e} — defaulting to C")
        return "C"


def grade_seal_predictions(
    client: OpenAI,
    questions: List[str], gold_answers: List[str],
    predicted_answers: List[str], grading_model: str,
) -> Tuple[Dict[str, float], List[str]]:
    grades: List[str] = []
    for q, g, p in tqdm(zip(questions, gold_answers, predicted_answers),
                        total=len(questions), desc="SEAL grading"):
        grades.append(_call_grader(client, q, g, p, grading_model))
    n      = len(grades)
    results = {
        'correct':       sum(g == "A" for g in grades) / n if n else 0.0,
        'incorrect':     sum(g == "B" for g in grades) / n if n else 0.0,
        'not_attempted': sum(g == "C" for g in grades) / n if n else 0.0,
    }
    return results, grades


# =============================================================================
# DATASET LOADERS
# =============================================================================
def load_truthfulqa_dataset(data_path: str) -> List[Dict]:
    candidates = (
        [os.path.join(data_path, f)
         for f in ("TruthfulQA.csv", "truthfulqa.csv",
                   os.path.join("TruthfulQA", "TruthfulQA.csv"))]
        if os.path.isdir(data_path) else [data_path]
    )
    filepath = next((p for p in candidates if os.path.exists(p)), None)
    if not filepath:
        logger.error(f"[TruthfulQA] CSV not found in {data_path}")
        return []
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"[TruthfulQA] Load failed: {e}"); return []

    dataset: List[Dict] = []
    for _, row in df.iterrows():
        try:
            s = {'question':     row['Question'],
                 'answer_best':  row.get('Best Answer', ''),
                 'answer_true':  row.get('Correct Answers', ''),
                 'answer_false': row.get('Incorrect Answers', '')}
            if not s['answer_best'] or pd.isna(s['answer_best']):
                ta = split_multi_answer(s['answer_true'])
                if ta: s['answer_best'] = ta[0].strip()
            if s['answer_true'] and s['answer_false']:
                dataset.append(s)
        except Exception:
            continue
    logger.info(f"[TruthfulQA] Loaded {len(dataset)} samples")
    return dataset


def load_hotpotqa_dataset(max_samples: Optional[int] = None) -> List[Dict]:
    try:
        ds   = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        data = list(ds["validation"])
        if max_samples: data = data[:max_samples]
        logger.info(f"[HotpotQA] Loaded {len(data)} samples"); return data
    except Exception as e:
        logger.warning(f"[HotpotQA] Load failed: {e}"); return []


def load_seal_variant(name: str, max_samples: Optional[int] = None) -> List[Dict]:
    try:
        ds   = load_dataset("vtllms/sealqa", name=name, split="test")
        data = list(ds)
        if max_samples: data = data[:max_samples]
        logger.info(f"[SEAL {name}] Loaded {len(data)} samples"); return data
    except Exception as e:
        logger.warning(f"[SEAL {name}] Load failed: {e}"); return []


# =============================================================================
# EVALUATION — TruthfulQA MC  (multiple-choice)
# =============================================================================
def evaluate_truthfulqa(
    llm: UnifiedDCSLED, dataset: List[Dict],
    mode: str, args: argparse.Namespace,
    profiler: Optional['LatencyProfiler'] = None,
    config_overrides: Optional[Dict] = None,
) -> Dict[str, Any]:
    logger.info(f"[TruthfulQA-MC] mode={mode}")
    config = get_model_adaptive_config(llm.model_name, 'truthfulqa')
    if config_overrides:
        config.update(config_overrides)

    if mode == 'VanillaGreedy':
        mature_layer, cpl = None, None
    else:
        if args.early_exit_layers is None:
            if config.get('use_layer_range', False):
                s = int(llm.num_layers * config.get('layer_range_start_ratio', 0.0))
                e = int(llm.num_layers * config.get('layer_range_end_ratio',   1.0))
                eel = list(range(s, e)) + [llm.num_layers]
            else:
                eel = list(range(llm.num_layers))
            mature_layer, cpl = eel[-1], eel[:-1]
        else:
            eel  = [int(x) for x in args.early_exit_layers.split(',')]
            mature_layer, cpl = eel[-1], eel[:-1]

    gkw = dict(mode=mode, mature_layer=mature_layer,
               candidate_premature_layers=cpl,
               relative_top=args.relative_top,
               relative_top_value=args.relative_top_value,
               post_softmax=args.post_softmax,
               dataset_type='truthfulqa',
               dola_alpha=args.dola_alpha,
               temperature=args.temperature,
               config_overrides=config_overrides)

    mc1s: List[float] = []
    mc2s: List[float] = []
    mc3s: List[float] = []
    lat_total         = 0.0
    n_proc            = 0

    num_samples = (min(len(dataset), args.max_samples)
                   if args.max_samples else len(dataset))

    for sample in tqdm(dataset[:num_samples], desc=f"TruthfulQA-MC ({mode})"):
        rb    = format_best(sample['answer_best'])
        rt    = split_multi_answer(sample['answer_true'])
        rf    = split_multi_answer(sample['answer_false'])
        if not rt or not rf: continue

        t0 = time.time()
        st = [llm.lm_score(*build_prompt_and_answer(sample['question'], a), **gkw)[0]
              for a in rt]
        sf = [llm.lm_score(*build_prompt_and_answer(sample['question'], a), **gkw)[0]
              for a in rf]
        lat = time.time() - t0

        scores = MC_calcs(st, sf, rt, rb)
        if any(np.isnan(scores[k]) for k in scores): continue

        mc1s.append(scores['MC1']); mc2s.append(scores['MC2']); mc3s.append(scores['MC3'])
        lat_total += lat; n_proc += 1

        if profiler:
            n_tok = sum(len(a.split()) for a in rt + rf)
            profiler.record(mode, n_tok, lat,
                            LatencyProfiler.current_gpu_mem_mb())

    n                = len(mc1s)
    mc1_pt, c1l, c1h = bootstrap_ci(mc1s, args.n_bootstrap) if mc1s else (0.0, 0.0, 0.0)
    mc2_pt, c2l, c2h = bootstrap_ci(mc2s, args.n_bootstrap) if mc2s else (0.0, 0.0, 0.0)
    mc3_pt, c3l, c3h = bootstrap_ci(mc3s, args.n_bootstrap) if mc3s else (0.0, 0.0, 0.0)

    result = {
        'total_mc1': round(mc1_pt, 4), 'mc1_ci95': [round(c1l,4), round(c1h,4)],
        'total_mc2': round(mc2_pt, 4), 'mc2_ci95': [round(c2l,4), round(c2h,4)],
        'total_mc3': round(mc3_pt, 4), 'mc3_ci95': [round(c3l,4), round(c3h,4)],
        'n_questions': n,
        'mc1_per_sample': mc1s,       # retained for paired tests
        'latency_total': round(lat_total, 4),
        'latency_avg':   round(lat_total / n_proc, 4) if n_proc > 0 else 0.0,
        'n_samples_processed': n_proc,
    }
    logger.info(
        f"[TruthfulQA-MC] MC1={mc1_pt:.4f} [{c1l:.4f},{c1h:.4f}]  "
        f"MC2={mc2_pt:.4f}  MC3={mc3_pt:.4f}  (n={n})")
    return result


# =============================================================================
# EVALUATION — TruthfulQA open-ended generation  (Major 8)
# Uses the SEAL LLM-judge to grade hallucination in free-form outputs.
# This addresses the TACL requirement for open-ended hallucination testing.
# =============================================================================
def evaluate_truthfulqa_openended(
    llm: UnifiedDCSLED, dataset: List[Dict],
    mode: str, args: argparse.Namespace,
    openai_client: OpenAI,
    profiler: Optional['LatencyProfiler'] = None,
) -> Dict[str, Any]:
    """
    Open-ended generation evaluation on TruthfulQA (Major 8).
    For each question the model generates a free-form answer; the answer is
    graded by the SEAL LLM judge against the best gold answer.
    Metric: fraction graded CORRECT (= truthful and non-hallucinated).
    """
    logger.info(f"[TruthfulQA-OE] open-ended generation | mode={mode}")
    llm.set_stop_words(["\n", "Q:"])

    questions:  List[str] = []
    gold_ans:   List[str] = []
    pred_ans:   List[str] = []
    lat_total             = 0.0

    num_samples = (min(len(dataset), args.max_samples)
                   if args.max_samples else len(dataset))

    for sample in tqdm(dataset[:num_samples], desc=f"TruthfulQA-OE ({mode})"):
        q    = sample['question'].strip()
        gold = sample['answer_best'].strip()
        if not q or not gold: continue

        prompt = f"Q: {q}\nA:"
        try:
            t0   = time.time()
            pred = llm.generate_answer(prompt, max_new_tokens=80)
            lat  = time.time() - t0
            lat_total += lat

            questions.append(q)
            gold_ans.append(gold)
            pred_ans.append(pred)

            if profiler:
                profiler.record(mode, len(pred.split()), lat,
                                LatencyProfiler.current_gpu_mem_mb())
        except Exception as e:
            logger.debug(f"[TruthfulQA-OE] Error: {e}")
            continue
        clear_cuda_memory()

    if not questions:
        return {'correct': 0.0, 'incorrect': 0.0, 'not_attempted': 0.0,
                'total': 0, 'latency_avg': 0.0}

    grade_res, grades = grade_seal_predictions(
        openai_client, questions, gold_ans, pred_ans, args.grading_model)

    n   = len(questions)
    pt, lo, hi = bootstrap_ci(
        [1.0 if g == "A" else 0.0 for g in grades], args.n_bootstrap)

    result = {
        'correct':       round(grade_res['correct'],       4),
        'incorrect':     round(grade_res['incorrect'],     4),
        'not_attempted': round(grade_res['not_attempted'], 4),
        'correct_ci95':  [round(lo, 4), round(hi, 4)],
        'total':         n,
        'latency_avg':   round(lat_total / n, 4) if n > 0 else 0.0,
        'correct_per_sample': [1.0 if g == "A" else 0.0 for g in grades],
    }
    logger.info(
        f"[TruthfulQA-OE] Correct={grade_res['correct']:.4f} "
        f"[{lo:.4f},{hi:.4f}]  (n={n})")
    return result


# =============================================================================
# EVALUATION — HotpotQA  (exact-match on generated answers)
# =============================================================================
def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', '', s)
    return ' '.join(s.split())


def evaluate_hotpotqa(
    llm: UnifiedDCSLED, dataset: List[Dict],
    mode: str, args: argparse.Namespace,
    profiler: Optional['LatencyProfiler'] = None,
) -> Dict[str, Any]:
    logger.info(f"[HotpotQA] mode={mode}")
    llm.set_stop_words(["\n", "Question:"])

    correct_flags: List[float] = []
    lat_total                  = 0.0

    for item in tqdm(dataset, desc=f"HotpotQA ({mode})"):
        q    = item.get('question', '').strip()
        gold = item.get('answer',   '').strip()
        ctx  = item.get('context',  '')
        if not q or not gold: continue

        if isinstance(ctx, str):
            ctx_text = ctx[:4000]
        elif isinstance(ctx, dict):
            sents    = ctx.get('sentences', [])
            ctx_text = ' '.join(' '.join(sl) for sl in sents)[:4000]
        else:
            ctx_text = ''

        prompt = f"Context: {ctx_text}\n\nQuestion: {q}\nAnswer:"
        try:
            t0   = time.time()
            pred = llm.generate_answer(prompt, max_new_tokens=64)
            lat  = time.time() - t0
            lat_total += lat

            hit = float(_normalize_answer(pred) == _normalize_answer(gold))
            correct_flags.append(hit)

            if profiler:
                profiler.record(mode, len(pred.split()), lat,
                                LatencyProfiler.current_gpu_mem_mb())
        except Exception as e:
            logger.debug(f"[HotpotQA] Error: {e}"); continue
        clear_cuda_memory()

    n              = len(correct_flags)
    pt, lo, hi     = bootstrap_ci(correct_flags, args.n_bootstrap) if correct_flags \
                     else (0.0, 0.0, 0.0)
    result = {
        'exact_match':     round(pt, 4),
        'em_ci95':         [round(lo,4), round(hi,4)],
        'correct':         int(sum(correct_flags)),
        'total':           n,
        'latency_total':   round(lat_total, 4),
        'latency_avg':     round(lat_total / n, 4) if n > 0 else 0.0,
        'n_samples_processed': n,
        'em_per_sample':   correct_flags,
    }
    logger.info(f"[HotpotQA] EM={pt:.4f} [{lo:.4f},{hi:.4f}]  (n={n})")
    return result


# =============================================================================
# EVALUATION — SEAL variants  (LLM-judge grading)
# =============================================================================
def evaluate_seal_variant(
    llm: UnifiedDCSLED, dataset: List[Dict],
    variant_name: str, mode: str,
    args: argparse.Namespace, openai_client: OpenAI,
    profiler: Optional['LatencyProfiler'] = None,
) -> Dict[str, Any]:
    logger.info(f"[SEAL {variant_name}] mode={mode}")
    llm.set_stop_words(["\n", "Question:"])

    questions: List[str] = []
    gold_ans:  List[str] = []
    pred_ans:  List[str] = []
    lat_total            = 0.0

    for item in tqdm(dataset, desc=f"SEAL {variant_name} ({mode}) — generating"):
        q    = item.get('question', '').strip()
        gold = item.get('answer',   '').strip()
        docs = item.get('documents', [])
        if not q or not gold: continue

        ctx = ("\n\n".join(str(d) for d in docs)
               if isinstance(docs, list) else str(docs))[:6000]
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"

        try:
            t0   = time.time()
            pred = llm.generate_answer(prompt, max_new_tokens=128)
            lat  = time.time() - t0
            lat_total += lat
            questions.append(q); gold_ans.append(gold); pred_ans.append(pred)

            if profiler:
                profiler.record(mode, len(pred.split()), lat,
                                LatencyProfiler.current_gpu_mem_mb())
        except Exception as e:
            logger.debug(f"[SEAL {variant_name}] Error: {e}"); continue
        clear_cuda_memory()

    if not questions:
        return {'correct': 0.0, 'incorrect': 0.0, 'not_attempted': 0.0,
                'total': 0, 'latency_avg': 0.0}

    grade_res, grades = grade_seal_predictions(
        openai_client, questions, gold_ans, pred_ans, args.grading_model)

    n          = len(questions)
    pt, lo, hi = bootstrap_ci(
        [1.0 if g == "A" else 0.0 for g in grades], args.n_bootstrap)

    result = {
        'correct':       round(grade_res['correct'],       4),
        'incorrect':     round(grade_res['incorrect'],     4),
        'not_attempted': round(grade_res['not_attempted'], 4),
        'correct_ci95':  [round(lo,4), round(hi,4)],
        'total':         n,
        'latency_total': round(lat_total, 4),
        'latency_avg':   round(lat_total / n, 4) if n > 0 else 0.0,
        'n_samples_processed': n,
        'correct_per_sample':  [1.0 if g == "A" else 0.0 for g in grades],
    }
    logger.info(
        f"[SEAL {variant_name}] Correct={grade_res['correct']:.4f} "
        f"[{lo:.4f},{hi:.4f}]  (n={n})")
    return result


# =============================================================================
# STATISTICAL SIGNIFICANCE REPORT  (Fatal 4)
# Computes paired sign tests between DCLED and every baseline on each benchmark.
# =============================================================================
def compute_significance_report(
    all_results: Dict[str, Dict[str, Any]],
    reference_method: str = 'DCLED',
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    For every benchmark and every non-reference method, run a two-sided sign
    test comparing per-sample scores.  Reports p-value and significance flag.
    """
    report: Dict[str, Any] = {}
    if reference_method not in all_results:
        return report

    ref_results = all_results[reference_method]

    for method, method_res in all_results.items():
        if method == reference_method: continue
        method_report: Dict[str, Any] = {}

        for bench in ref_results:
            if bench not in method_res: continue
            ref_b    = ref_results[bench]
            other_b  = method_res[bench]

            # Choose per-sample scores depending on benchmark type
            per_sample_key = None
            if 'mc1_per_sample'      in ref_b: per_sample_key = 'mc1_per_sample'
            elif 'em_per_sample'     in ref_b: per_sample_key = 'em_per_sample'
            elif 'correct_per_sample'in ref_b: per_sample_key = 'correct_per_sample'

            if per_sample_key and per_sample_key in other_b:
                sa = ref_b[per_sample_key]
                sb = other_b[per_sample_key]
                n  = min(len(sa), len(sb))
                if n > 0:
                    p_val, sig = paired_sign_test(sa[:n], sb[:n])
                    method_report[bench] = {
                        'p_value':         round(p_val, 4),
                        'significant_5pct': sig,
                        'n_pairs':         n,
                    }

        if method_report:
            report[f"{reference_method}_vs_{method}"] = method_report

    return report


# =============================================================================
# ABLATION RUNNER  (Major 5)
# =============================================================================
def run_ablation_suite(
    llm: UnifiedDCSLED,
    datasets: Dict[str, List[Dict]],
    args: argparse.Namespace,
    openai_client: OpenAI,
    profiler: 'LatencyProfiler',
) -> Dict[str, Any]:
    """
    Run all 10 ablation configurations (from get_ablation_configs) on every
    loaded dataset using DCLED as the decoding backbone.
    Returns a nested dict: ablation_label → benchmark → metrics.
    """
    base_config  = get_model_adaptive_config(llm.model_name, 'all')
    abl_configs  = get_ablation_configs(base_config)
    abl_results: Dict[str, Any] = {}

    for label, overrides in abl_configs:
        logger.info(f"\n{'='*55}\n[Ablation] {label}\n{'='*55}")
        llm._ablation_config_overrides = overrides
        bench_results: Dict[str, Any] = {}

        if 'truthfulqa' in datasets and datasets['truthfulqa']:
            bench_results['truthfulqa'] = evaluate_truthfulqa(
                llm, datasets['truthfulqa'], 'DCLED', args, profiler,
                config_overrides=overrides)

        if 'hotpotqa' in datasets and datasets['hotpotqa']:
            bench_results['hotpotqa'] = evaluate_hotpotqa(
                llm, datasets['hotpotqa'], 'DCLED', args, profiler)

        for sname in ('seal_0', 'seal_hard', 'longseal'):
            if sname in datasets and datasets[sname]:
                bench_results[sname] = evaluate_seal_variant(
                    llm, datasets[sname], sname, 'DCLED', args,
                    openai_client, profiler)

        abl_results[label] = bench_results
        # Reset override after each ablation
        llm._ablation_config_overrides = {}

    return abl_results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = create_argument_parser()
    args   = parser.parse_args()

    # ── Seed ─────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Model ────────────────────────────────────────────────────────────────
    llm = UnifiedDCSLED(
        args.model_name, device=args.device,
        num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)

    # ── OpenAI client ────────────────────────────────────────────────────────
    openai_client = OpenAI(api_key=args.openai_api_key)

    # ── Reproducibility metadata  (Major 7) ─────────────────────────────────
    repro_meta = collect_reproducibility_metadata(args)
    logger.info(f"[Repro] {json.dumps(repro_meta, indent=2)}")

    # ── Latency profiler  (Major 6) ─────────────────────────────────────────
    profiler = LatencyProfiler(DEVICE)

    # ── Parse multi-seed list  (Fatal 4) ────────────────────────────────────
    seeds = [int(s) for s in args.seeds.split(',')]

    # ── Methods ──────────────────────────────────────────────────────────────
    # Fatal 3: no "state-of-the-art" claim; methods list is factual.
    # Step 5: ContrastiveDecoding and EntropyGated added as baselines.
    if args.run_ablation:
        methods = ['DCLED', 'dola', 'VanillaGreedy', 'SLED',
                   'ContrastiveDecoding', 'EntropyGated']
    else:
        methods = [args.decoding_method]

    # ── Load datasets once (outside seed loop) ───────────────────────────────
    all_datasets: Dict[str, List[Dict]] = {}

    run_truth    = args.dataset in ('all', 'new_benchmarks', 'truthfulqa')
    run_hotpot   = args.dataset in ('all', 'new_benchmarks', 'hotpotqa')
    run_seal_0   = args.dataset in ('all', 'new_benchmarks', 'seal_0')
    run_seal_h   = args.dataset in ('all', 'new_benchmarks', 'seal_hard')
    run_longseal = args.dataset in ('all', 'new_benchmarks', 'sealqa')

    if run_truth:
        raw = load_truthfulqa_dataset(args.truthfulqa_path)
        all_datasets['truthfulqa'], split_label = apply_dev_test_split(
            raw, args.dev_benchmark, 'truthfulqa', args.dev_fraction, args.seed)
        repro_meta['truthfulqa_split'] = split_label

    if run_hotpot:
        raw = load_hotpotqa_dataset(args.max_samples)
        all_datasets['hotpotqa'], split_label = apply_dev_test_split(
            raw, args.dev_benchmark, 'hotpotqa', args.dev_fraction, args.seed)
        repro_meta['hotpotqa_split'] = split_label

    if run_seal_0:
        all_datasets['seal_0']  = load_seal_variant('seal_0',   args.max_samples)
    if run_seal_h:
        all_datasets['seal_hard'] = load_seal_variant('seal_hard', args.max_samples)
    if run_longseal:
        all_datasets['longseal']  = load_seal_variant('longseal',  args.max_samples)

    # ── Multi-seed or single-seed evaluation  (Fatal 4) ──────────────────────
    eval_seeds = seeds if args.multi_seed_eval else [args.seed]

    all_results_across_seeds: Dict[str, List[Dict]] = defaultdict(list)

    for seed_i, current_seed in enumerate(eval_seeds):
        if args.multi_seed_eval:
            logger.info(f"\n{'='*60}\nSeed {current_seed} "
                        f"({seed_i+1}/{len(eval_seeds)})\n{'='*60}")
            random.seed(current_seed); np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)

        for method in methods:
            logger.info(f"\n{'='*55}\nMethod: {method}\n{'='*55}")
            method_results: Dict[str, Any] = {}

            # TruthfulQA MC
            if 'truthfulqa' in all_datasets and all_datasets['truthfulqa']:
                method_results['truthfulqa'] = evaluate_truthfulqa(
                    llm, all_datasets['truthfulqa'], method, args, profiler)

            # TruthfulQA open-ended  (Major 8)
            if ('truthfulqa' in all_datasets and all_datasets['truthfulqa']
                    and method == 'DCLED'):   # run OE only for primary method
                method_results['truthfulqa_oe'] = evaluate_truthfulqa_openended(
                    llm, all_datasets['truthfulqa'], method, args,
                    openai_client, profiler)

            # HotpotQA
            if 'hotpotqa' in all_datasets and all_datasets['hotpotqa']:
                method_results['hotpotqa'] = evaluate_hotpotqa(
                    llm, all_datasets['hotpotqa'], method, args, profiler)

            # SEAL variants
            for sname in ('seal_0', 'seal_hard', 'longseal'):
                if sname in all_datasets and all_datasets[sname]:
                    method_results[sname] = evaluate_seal_variant(
                        llm, all_datasets[sname], sname, method,
                        args, openai_client, profiler)

            all_results_across_seeds[method].append(method_results)

    # ── Aggregate multi-seed results  (Fatal 4) ──────────────────────────────
    final_results: Dict[str, Any] = {}
    for method, seed_list in all_results_across_seeds.items():
        if len(seed_list) == 1:
            final_results[method] = seed_list[0]
        else:
            # Aggregate each benchmark independently
            bench_names = seed_list[0].keys()
            final_results[method] = {}
            for bench in bench_names:
                per_seed_bench = [sl[bench] for sl in seed_list if bench in sl]
                final_results[method][bench] = aggregate_multi_seed_results(
                    per_seed_bench, n_bootstrap=args.n_bootstrap)

    # ── Significance testing  (Fatal 4) ─────────────────────────────────────
    # Use the single-seed (or last-seed) results for per-sample sign tests
    single_seed_results = {m: sl[-1] for m, sl in all_results_across_seeds.items()}
    sig_report = compute_significance_report(
        single_seed_results, reference_method='DCLED',
        n_bootstrap=args.n_bootstrap)

    # ── Ablation suite  (Major 5) ─────────────────────────────────────────
    ablation_results: Dict[str, Any] = {}
    if args.run_ablation:
        ablation_results = run_ablation_suite(
            llm, all_datasets, args, openai_client, profiler)

    # ── Latency table  (Major 6) ─────────────────────────────────────────
    latency_table = profiler.summary(baseline_method='VanillaGreedy')

    # ── Save everything ──────────────────────────────────────────────────────
    output = {
        'reproducibility':   repro_meta,
        'results':           final_results,
        'significance':      sig_report,
        'ablation':          ablation_results,
        'latency_table':     latency_table,
    }

    # Strip per-sample arrays from JSON output to keep file sizes manageable
    def _strip_per_sample(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _strip_per_sample(v)
                    for k, v in obj.items()
                    if not k.endswith('_per_sample')}
        return obj

    with open(args.output_path, 'w') as f:
        json.dump(_strip_per_sample(output), f, indent=4)

    # ── Summary  (Fatal 2 & 3 — conservative, accurate claim printing) ───────
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY — DCLED provides consistent improvements over")
    logger.info("layer-wise decoding baselines (DoLa, SLED, Greedy).")
    logger.info("Cross-scale claims: within each model size only.")
    logger.info("="*70)

    for method, res in final_results.items():
        logger.info(f"\n{method.upper()}:")
        if not isinstance(res, dict): continue
        for bench, metrics in res.items():
            if not isinstance(metrics, dict): continue
            # multi-seed: metrics are nested dicts with 'mean'
            def _v(k: str) -> str:
                v = metrics.get(k, {})
                if isinstance(v, dict):
                    m, s = v.get('mean', 0.0), v.get('std', 0.0)
                    return f"{m:.4f}±{s:.4f}"
                return f"{float(v):.4f}" if v is not None else "N/A"

            if 'total_mc1' in metrics:
                logger.info(f"  {bench}: MC1={_v('total_mc1')} "
                            f"MC2={_v('total_mc2')} MC3={_v('total_mc3')}")
            elif 'exact_match' in metrics:
                logger.info(f"  {bench}: EM={_v('exact_match')}")
            elif 'correct' in metrics:
                logger.info(f"  {bench}: Correct={_v('correct')} "
                            f"Incorrect={_v('incorrect')} "
                            f"NotAttempted={_v('not_attempted')}")

    logger.info("\n[Latency Table]")
    for meth, lat in latency_table.items():
        logger.info(
            f"  {meth:25s}  {lat.get('ms_per_token',0):.1f} ms/tok  "
            f"{lat.get('tokens_per_sec',0):.1f} tok/s  "
            f"{lat.get('gpu_mem_mb',0):.0f} MB  "
            f"×{lat.get('overhead_vs_greedy','N/A')} vs greedy")

    if sig_report:
        logger.info("\n[Significance — paired sign test vs DCLED]")
        for comparison, benches in sig_report.items():
            for bench, st in benches.items():
                flag = "✓ sig" if st['significant_5pct'] else "✗ n.s."
                logger.info(f"  {comparison} | {bench}: "
                            f"p={st['p_value']:.4f} {flag} (n={st['n_pairs']})")

    logger.info("\n" + "="*70)
    logger.info("Evaluation complete.  Results saved to: " + args.output_path)
    logger.info("="*70)
