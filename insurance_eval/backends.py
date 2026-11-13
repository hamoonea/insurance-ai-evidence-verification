"""Local model adapter and explicitly non-experimental software fixture."""
import importlib.metadata
import json
import os
import time
from pathlib import Path


class FixtureBackend:
    kind = 'software_fixture'

    def __init__(self, **kwargs):
        self.metadata = {'backend': 'fixture', 'model': None, 'experimental_results': False}

    def generate(self, messages, *, policy, structured, seed):
        # Deliberately abstains on every case. Never reads gold labels.
        sid = next(s['id'] for s in policy['segments'] if 'Nationally Covered' in s['text'])
        obj = {'label': 'insufficient_information', 'explanation': 'Software fixture; no model inference.',
               'citations': [sid], 'missing_fields': [], 'clarification_questions': []}
        if structured:
            obj.update(criteria=[{'id': 'C1', 'condition': 'Software fixture condition', 'status': 'unknown',
                                  'policy_refs': [sid], 'case_evidence': [], 'missing_fields': []}],
                       logic={'ref': 'C1'})
        return {'text': json.dumps(obj), 'finish_reason': 'stop', 'elapsed_seconds': 0.0,
                'prompt_tokens': None, 'generation_tokens': None, 'seed': seed}


class MLXBackend:
    kind = 'model_inference'

    def __init__(self, model_path, max_tokens=2000, temperature=0.0):
        if not Path(model_path).is_dir():
            raise RuntimeError('Local model directory does not exist; this runner never downloads weights.')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        try:
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError('MLX cannot initialize. Run from a local Apple Silicon session with GPU access and mlx-lm installed. ' + str(exc)) from exc
        self.mx, self.stream_generate, self.make_sampler = mx, stream_generate, make_sampler
        self.model, self.tokenizer = load(str(model_path))
        self.max_tokens, self.temperature = max_tokens, temperature
        self.metadata = {'backend': 'mlx', 'model_path': str(Path(model_path).resolve()),
                         'mlx': importlib.metadata.version('mlx'), 'mlx_lm': importlib.metadata.version('mlx-lm'),
                         'temperature': temperature, 'max_tokens': max_tokens, 'experimental_results': True}

    def generate(self, messages, *, policy, structured, seed):
        self.mx.random.seed(seed)
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        start = time.perf_counter()
        parts, last = [], None
        for item in self.stream_generate(self.model, self.tokenizer, prompt=rendered,
                                        max_tokens=self.max_tokens,
                                        sampler=self.make_sampler(temp=self.temperature)):
            parts.append(item.text)
            last = item
        if last is None:
            raise RuntimeError('Model returned no tokens')
        result = {'text': ''.join(parts), 'finish_reason': last.finish_reason,
                  'elapsed_seconds': time.perf_counter() - start,
                  'prompt_tokens': last.prompt_tokens, 'generation_tokens': last.generation_tokens, 'seed': seed}
        self.mx.clear_cache()
        return result


class TransformersCPUBackend:
    """Local safetensors only; model download is a separate explicit step."""
    kind = 'model_inference'

    def __init__(self, model_path, max_tokens=2000, temperature=0.0, threads=4):
        if not Path(model_path).is_dir():
            raise RuntimeError('Local model directory does not exist')
        if temperature != 0:
            raise ValueError('This CPU pilot uses greedy decoding (temperature zero)')
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.bfloat16,
            device_map={'': 'cpu'}, attn_implementation='sdpa').eval()
        self.max_tokens = max_tokens
        self.metadata = {'backend': 'transformers', 'device': 'cpu', 'dtype': 'bfloat16',
                         'threads': threads, 'torch': importlib.metadata.version('torch'),
                         'transformers': importlib.metadata.version('transformers'),
                         'temperature': 0.0, 'max_tokens': max_tokens, 'experimental_results': True,
                         'model_source': json.loads((Path(model_path)/'source.json').read_text())}

    def generate(self, messages, *, policy, structured, seed):
        self.torch.manual_seed(seed)
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
        prompt_tokens = inputs['input_ids'].shape[-1]
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        generated = output[0, prompt_tokens:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        return {'text': self.tokenizer.decode(generated, skip_special_tokens=True),
                'finish_reason': 'stop' if generated and generated[-1] in eos else 'length',
                'elapsed_seconds': time.perf_counter()-start,
                'prompt_tokens': prompt_tokens, 'generation_tokens': len(generated), 'seed': seed}
