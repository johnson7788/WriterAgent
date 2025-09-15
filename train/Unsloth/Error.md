# bash train_SFT.sh报错，发现是trl版本问题，trl==0.21.0好像没有tokenizer选项
改成processing_class等于tokenizer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
Traceback (most recent call last):
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 461, in <module>
    main()
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 432, in main
    trainer = build_trainer(args, model, tokenizer, dataset)
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 229, in build_trainer
    trainer = SFTTrainer(
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'

# bash train_SFT.sh报错，发现也是trl版本问题，trl==0.21.0
Traceback (most recent call last):
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 461, in <module>
    main()
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 432, in main
    trainer = build_trainer(args, model, tokenizer, dataset)
  File "/workspace/verl/tools/train_unsloth_qwen_SFT.py", line 229, in build_trainer
    trainer = SFTTrainer(
  File "/usr/local/lib/python3.10/dist-packages/trl/trainer/sft_trainer.py", line 380, in __init__
    raise ValueError(
ValueError: The specified `eos_token` ('<EOS_TOKEN>') is not found in the vocabulary of the given `processing_class` (Qwen2TokenizerFast). Ensure that the `eos_token` exists in the vocabulary before using it as an EOS token.

# FlashInfer报错
安装对应版本或者先临时禁用查看：https://flashinfer.ai/whl的版本
临时禁用，强制 vLLM 用 FlashAttention，绕过 FlashInfer： export VLLM_ATTENTION_BACKEND=FLASH_ATTN 
INFO 09-07 07:06:56 [parallel_state.py:1134] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
WARNING 09-07 07:06:56 [topk_topp_sampler.py:61] FlashInfer is not available. Falling back to the PyTorch-native implementation of top-p & top-k sampling. For the best performance, please install FlashInfer.
INFO 09-07 07:06:56 [gpu_model_runner.py:1953] Starting to load model unsloth/Qwen3-4B-Base...
INFO 09-07 07:06:56 [gpu_model_runner.py:1985] Loading model from scratch...
INFO 09-07 07:06:57 [cuda.py:275] Using FlashInfer backend on V1 engine.
[rank0]: Traceback (most recent call last):
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/flashinfer/jit/__init__.py", line 56, in <module>
[rank0]:     from .. import flashinfer_kernels, flashinfer_kernels_sm90  # noqa: F401
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: ImportError: /opt/conda/lib/python3.11/site-packages/flashinfer/flashinfer_kernels.abi3.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb

[rank0]: The above exception was the direct cause of the following exception:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth_zoo/vllm_utils.py", line 1555, in load_vllm
[rank0]:     llm = LLM(**engine_args)
[rank0]:           ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/entrypoints/llm.py", line 285, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/engine/llm_engine.py", line 490, in from_engine_args
[rank0]:     return engine_cls.from_vllm_config(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/llm_engine.py", line 127, in from_vllm_config
[rank0]:     return cls(vllm_config=vllm_config,
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/llm_engine.py", line 104, in __init__
[rank0]:     self.engine_core = EngineCoreClient.make_client(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 82, in make_client
[rank0]:     return InprocClient(vllm_config, executor_class, log_stats)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/core_client.py", line 245, in __init__
[rank0]:     self.engine_core = EngineCore(*args, **kwargs)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 80, in __init__
[rank0]:     self.model_executor = executor_class(vllm_config)
[rank0]:                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/executor/executor_base.py", line 54, in __init__
[rank0]:     self._init_executor()
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/executor/uniproc_executor.py", line 49, in _init_executor
[rank0]:     self.collective_rpc("load_model")
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/executor/uniproc_executor.py", line 58, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/utils/__init__.py", line 3007, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/worker/gpu_worker.py", line 212, in load_model
[rank0]:     self.model_runner.load_model(eep_scale_up=eep_scale_up)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/worker/gpu_model_runner.py", line 1986, in load_model
[rank0]:     self.model = model_loader.load_model(
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/model_loader/base_loader.py", line 44, in load_model
[rank0]:     model = initialize_model(vllm_config=vllm_config,
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/model_loader/utils.py", line 63, in initialize_model
[rank0]:     return model_class(vllm_config=vllm_config, prefix=prefix)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 287, in __init__
[rank0]:     self.model = Qwen3Model(vllm_config=vllm_config,
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/compilation/decorators.py", line 183, in __init__
[rank0]:     old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 259, in __init__
[rank0]:     super().__init__(vllm_config=vllm_config,
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/compilation/decorators.py", line 183, in __init__
[rank0]:     old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen2.py", line 316, in __init__
[rank0]:     self.start_layer, self.end_layer, self.layers = make_layers(
[rank0]:                                                     ^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/utils.py", line 640, in make_layers
[rank0]:     [PPMissingLayer() for _ in range(start_layer)] + [
[rank0]:                                                      ^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/utils.py", line 641, in <listcomp>
[rank0]:     maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
[rank0]:                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen2.py", line 318, in <lambda>
[rank0]:     lambda prefix: decoder_layer_type(config=config,
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 189, in __init__
[rank0]:     self.self_attn = Qwen3Attention(
[rank0]:                      ^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/qwen3.py", line 123, in __init__
[rank0]:     self.attn = Attention(
[rank0]:                 ^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/attention/layer.py", line 164, in __init__
[rank0]:     self.attn_backend = get_attn_backend(head_size,
[rank0]:                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/attention/selector.py", line 154, in get_attn_backend
[rank0]:     return _cached_get_attn_backend(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/attention/selector.py", line 211, in _cached_get_attn_backend
[rank0]:     return resolve_obj_by_qualname(attention_cls)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/utils/__init__.py", line 2568, in resolve_obj_by_qualname
[rank0]:     module = importlib.import_module(module_name)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank0]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank0]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank0]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank0]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank0]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank0]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/vllm/v1/attention/backends/flashinfer.py", line 10, in <module>
[rank0]:     from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/flashinfer/__init__.py", line 18, in <module>
[rank0]:     from .activation import gelu_and_mul as gelu_and_mul
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/flashinfer/activation.py", line 21, in <module>
[rank0]:     from .jit import gen_act_and_mul_module, has_prebuilt_ops, load_cuda_ops    # noqa: F401
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/flashinfer/jit/__init__.py", line 62, in <module>
[rank0]:     raise ImportError("Loading prebuilt ops failed.") from e
[rank0]: ImportError: Loading prebuilt ops failed.

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/docs/Unsloth/unsloth_GRPO.py", line 15, in <module>
[rank0]:     model, tokenizer = FastLanguageModel.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth/models/loader.py", line 404, in from_pretrained
[rank0]:     model, tokenizer = dispatch_model.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth/models/qwen3.py", line 436, in from_pretrained
[rank0]:     return FastLlamaModel.from_pretrained(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth/models/llama.py", line 2088, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/opt/conda/lib/python3.11/site-packages/unsloth_zoo/vllm_utils.py", line 1578, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError: Loading prebuilt ops failed.
[rank0]:[W907 07:07:00.626827755 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())


# 推理报错, 一定要选择transformers==4.55.4这个版本
Please restructure your imports with 'import unsloth' at the top of your file.
  from unsloth import FastModel
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
/opt/conda/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[2025-09-07 09:13:05,229] [INFO] [real_accelerator.py:254:get_accelerator] Setting ds_accelerator to cuda (auto detect)
INFO 09-07 09:13:06 [__init__.py:241] Automatically detected platform cuda.
Unsloth: Your Flash Attention 2 installation seems to be broken?
A possible explanation is you have a new CUDA version which isn't
yet compatible with FA2? Please file a ticket to Unsloth or FA2.
We shall now use Xformers instead, which does not have any performance hits!
We found this negligible impact by benchmarking on 1x A100.
🦥 Unsloth Zoo will now patch everything to make training faster!
[09:13:11] INFO - 日志写入: ./outputs/qwen3_4b_sft_lora/logs_infer/logs/train_20250907_091311.log
[09:13:11] INFO - 随机种子已设置: 3407
[09:13:11] INFO - PyTorch: 2.7.1+cu126
[09:13:11] INFO - GPU: NVIDIA GeForce RTX 4090 D  显存上限: 23.546 GB  启动保留: 0.0 GB
[09:13:11] INFO - 加载 Tokenizer 自: ./outputs/qwen3_4b_sft_lora
[09:13:11] INFO - 检测到 LoRA 适配器目录，基础模型: unsloth/Qwen3-4B-Instruct-2507
==((====))==  Unsloth 2025.9.1: Fast Qwen3 patching. Transformers: 4.56.1. vLLM: 0.10.1.1.
   \\   /|    NVIDIA GeForce RTX 4090 D. Num GPUs = 1. Max memory: 23.546 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.1+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.3.1
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
[09:13:41] INFO - 推理消息: [{'role': 'user', 'content': 'Continue the sequence: 1, 1, 2, 3, 5, 8,'}]
<|im_start|>user
Continue the sequence: 1, 1, 2, 3, 5, 8,<|im_end|>
<|im_start|>assistant
Traceback (most recent call last):
  File "/opt/conda/lib/python3.11/site-packages/unsloth/models/vision.py", line 236, in unsloth_base_fast_generate
    output = self._old_generate(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/generation/utils.py", line 2464, in generate
    raise ValueError("assisted generate is not supported with Static cache classes`")
ValueError: assisted generate is not supported with Static cache classes`

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspace/verl/tmp/inference_sft.py", line 320, in <module>
    main()
  File "/workspace/verl/tmp/inference_sft.py", line 315, in main
    run_inference(model, tokenizer, messages, args)
  File "/workspace/verl/tmp/inference_sft.py", line 286, in run_inference
    out = model.generate(
          ^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/peft/peft_model.py", line 823, in generate
    return self.get_base_model().generate(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/unsloth/models/vision.py", line 241, in unsloth_base_fast_generate
    output = self._old_generate(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/generation/utils.py", line 2399, in generate
    self._prepare_cache_for_generation(
  File "/opt/conda/lib/python3.11/site-packages/transformers/generation/utils.py", line 1965, in _prepare_cache_for_generation
    model_kwargs[cache_name] = self._get_cache(
                               ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/generation/utils.py", line 1835, in _get_cache
    or cache_to_check.max_batch_size != batch_size
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/cache_utils.py", line 904, in max_batch_size
    values = [layer.max_batch_size for layer in self.layers]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/transformers/cache_utils.py", line 904, in <listcomp>
    values = [layer.max_batch_size for layer in self.layers]
              ^^^^^^^^^^^^^^^^^^^^
AttributeError: 'StaticLayer' object has no attribute 'max_batch_size'

# 报错 同一进程多次load模型
INFO 09-07 11:23:32 [gpu_model_runner.py:1875] Loading model from scratch...
[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/vllm_utils.py", line 1504, in load_vllm
[rank0]:     llm = LLM(**engine_args)
[rank0]:           ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/llm.py", line 273, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 497, in from_engine_args
[rank0]:     return engine_cls.from_vllm_config(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/llm_engine.py", line 126, in from_vllm_config
[rank0]:     return cls(vllm_config=vllm_config,
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/llm_engine.py", line 103, in __init__
[rank0]:     self.engine_core = EngineCoreClient.make_client(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 79, in make_client
[rank0]:     return InprocClient(vllm_config, executor_class, log_stats)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 235, in __init__
[rank0]:     self.engine_core = EngineCore(*args, **kwargs)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 77, in __init__
[rank0]:     self.model_executor = executor_class(vllm_config)
[rank0]:                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 53, in __init__
[rank0]:     self._init_executor()
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 49, in _init_executor
[rank0]:     self.collective_rpc("load_model")
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 58, in collective_rpc
[rank0]:     answer = run_method(self.driver_worker, method, args, kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py", line 2985, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 201, in load_model
[rank0]:     self.model_runner.load_model(eep_scale_up=eep_scale_up)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 1876, in load_model
[rank0]:     self.model = model_loader.load_model(
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 44, in load_model
[rank0]:     model = initialize_model(vllm_config=vllm_config,
[rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 67, in initialize_model
[rank0]:     return model_class(vllm_config=vllm_config, prefix=prefix)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3.py", line 271, in __init__
[rank0]:     self.model = Qwen3Model(vllm_config=vllm_config,
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 183, in __init__
[rank0]:     old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3.py", line 243, in __init__
[rank0]:     super().__init__(vllm_config=vllm_config,
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 183, in __init__
[rank0]:     old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2.py", line 316, in __init__
[rank0]:     self.start_layer, self.end_layer, self.layers = make_layers(
[rank0]:                                                     ^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 640, in make_layers
[rank0]:     maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
[rank0]:                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2.py", line 318, in <lambda>
[rank0]:     lambda prefix: decoder_layer_type(config=config,
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3.py", line 174, in __init__
[rank0]:     self.self_attn = Qwen3Attention(
[rank0]:                      ^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3.py", line 117, in __init__
[rank0]:     self.attn = Attention(self.num_heads,
[rank0]:                 ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/vllm/attention/layer.py", line 193, in __init__
[rank0]:     raise ValueError(f"Duplicate layer name: {prefix}")
[rank0]: ValueError: Duplicate layer name: model.layers.0.self_attn.attn

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/workspace/verl/docs/Unsloth/train_GRPO.py", line 431, in <module>
[rank0]:     main()
[rank0]:   File "/workspace/verl/docs/Unsloth/train_GRPO.py", line 298, in main
[rank0]:     model, tokenizer = FastLanguageModel.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth/models/loader.py", line 402, in from_pretrained
[rank0]:     model, tokenizer = dispatch_model.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth/models/qwen3.py", line 420, in from_pretrained
[rank0]:     return FastLlamaModel.from_pretrained(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth/models/llama.py", line 2041, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/dist-packages/unsloth_zoo/vllm_utils.py", line 1516, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError: Duplicate layer name: model.layers.0.self_attn.attn
[rank0]:[W907 11:23:36.468842731 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())


# 需要在training_args = GRPOConfig(， 需要设置vllm_mode和vllm_server_base_url才行
use_vllm = True,
vllm_mode="server",
vllm_server_base_url="http://127.0.0.1:8000",

    return super().__getattr__(name)  # defer to nn.Module's logic
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1928, in __getattr__
    raise AttributeError(
AttributeError: 'PeftModelForCausalLM' object has no attribute 'vllm_engine'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/lora/model.py", line 359, in __getattr__
    return super().__getattr__(name)  # defer to nn.Module's logic
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1928, in __getattr__
    raise AttributeError(
AttributeError: 'LoraModel' object has no attribute 'vllm_engine'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspace/verl/docs/Unsloth/train_GRPO.py", line 432, in <module>
    main()
  File "/workspace/verl/docs/Unsloth/train_GRPO.py", line 412, in main
    trainer = GRPOTrainer(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/trainer.py", line 209, in new_init
    original_init(self, *args, **kwargs)
  File "/workspace/verl/docs/Unsloth/unsloth_compiled_cache/UnslothGRPOTrainer.py", line 2896, in __init__
    super().__init__(
  File "/workspace/verl/docs/Unsloth/unsloth_compiled_cache/UnslothGRPOTrainer.py", line 1438, in __init__
    self.llm = model.vllm_engine
  File "/usr/local/lib/python3.10/dist-packages/peft/peft_model.py", line 797, in __getattr__
    return getattr(self.base_model, name)
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/lora/model.py", line 363, in __getattr__
    return getattr(self.model, name)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1928, in __getattr__
    raise AttributeError(
AttributeError: 'Qwen3ForCausalLM' object has no attribute 'vllm_engine'


# 没有save_lora这个参数
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/peft/peft_model.py", line 793, in __getattr__
    return super().__getattr__(name)  # defer to nn.Module's logic
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1928, in __getattr__
    raise AttributeError(
AttributeError: 'PeftModelForCausalLM' object has no attribute 'save_lora'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/lora/model.py", line 359, in __getattr__
    return super().__getattr__(name)  # defer to nn.Module's logic
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1928, in __getattr__
    raise AttributeError(
AttributeError: 'LoraModel' object has no attribute 'save_lora'

During handling of the above exception, another exception occurred:
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/workspace/verl/docs/Unsloth/train_GRPO.py", line 434, in <module>
    main()
  File "/workspace/verl/docs/Unsloth/train_GRPO.py", line 426, in main
    model.save_lora(LORA_SAVE_DIR)
  File "/usr/local/lib/python3.10/dist-packages/peft/peft_model.py", line 797, in __getattr__
    return getattr(self.base_model, name)
  File "/usr/local/lib/python3.10/dist-packages/peft/tuners/lora/model.py", line 363, in __getattr__
    return getattr(self.model, name)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1928, in __getattr__
    raise AttributeError(
AttributeError: 'Qwen3ForCausalLM' object has no attribute 'save_lora'