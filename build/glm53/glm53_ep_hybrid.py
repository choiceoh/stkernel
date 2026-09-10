"""Opt-in routed-expert EP2 x TP2 layout on the existing physical TP4 group.

Pinned to the CT NVFP4 runtime captured from image a3dd4c0f.
Startup canaries and full-model acceptance are required before adoption.
"""
from dataclasses import dataclass, replace


# Disk-source contract checked once per loaded package root, only when flag1.
# These immutable current-image base files are not model/torch imports.
_RUNTIME_SOURCE_PINS = {'model_executor/layers/fused_moe/config.py': '036633db18b77a1923ec36585a65b37353cb051421cac4e388704b076bda203c',
 'model_executor/layers/fused_moe/layer.py': '688711dd909e107c1712b81038cb1ed055fc4ecc9dce89bb4beef2d30f0541c5',
 'model_executor/layers/fused_moe/expert_map_manager.py': '308c00f63fcb6f44518ec3a56d8f902c7bba75928c2e3aea30fbe1f3f473973b',
 'model_executor/layers/fused_moe/routed_experts.py': 'b35557b83c103710272025821b9fcabb37b6db481ca3087a9f2abf823ebbb9df',
 'model_executor/layers/fused_moe/runner/moe_runner.py': 'a19ecf2417c80b93c0fe5594e74528f686665847732196f9b29ef967b38110b0',
 'model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py': '80d185e8d519da8ff7266fe37554640465672f54426f1bebd0131d62f4f67ec3',
 'model_executor/layers/fused_moe/oracle/nvfp4.py': '35b830ad8f9e1cf5a107698792ae5c9e2f9e0ccdd33e6d41094322f0f118b13d',
 'model_executor/layers/fused_moe/all2all_utils.py': 'd5e27dc317fbba083ef3c290bca5c5374b697cabaffd10e51db3d075bfb59319',
 'model_executor/layers/quantization/utils/flashinfer_fp4_moe.py': '11eb43daad6de9f57d8f7914b158ad1a63a19aa2c0716957866f947e19a6bb7e',
 'model_executor/layers/quantization/compressed_tensors/compressed_tensors.py': '180d2579f7031dfa67f00c9980187705ba0429c4a16678defe59c4a6116d1d5a',
 'model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py': 'bffd615fb2a31be3e10c88eaf4ca39bb12711fd448b0cbf63d89be04ca8b825b',
 'model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py': 'de7386c7229b9a5bc499f1e226071f64656661d1346c56e74c3bb18022173ecf',
 'model_executor/layers/quantization/compressed_tensors/utils.py': '92e228c931dd893504698e89d75d8964aeee2917fbdbfe6083686755222ac185',
 'model_executor/layers/quantization/utils/quant_utils.py': 'b8cd5aa5aacaa7658b5f5a51d5b9766df80832904281b24e429d03ee0f9eb0ad',
 'model_executor/layers/fused_moe/modular_kernel.py': '22193772b14b3de3f672ad468cc1416e9a52c86bcfead66fb39c86043adf43ca'}
_RUNTIME_SOURCE_CACHE = {}


def hybrid_runtime_sources(base_cls, quant_config):
    import hashlib
    import sys
    from pathlib import Path
    base_module = 'vllm.model_executor.layers.fused_moe.routed_experts'
    quant_module = 'vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors'
    if (base_cls.__name__ != 'RoutedExperts' or base_cls.__module__ != base_module
            or type(quant_config).__name__ != 'CompressedTensorsConfig'
            or type(quant_config).__module__ != quant_module):
        raise RuntimeError('hybrid requires the pinned actual RoutedExperts/CT classes')
    base_path = Path(sys.modules[base_module].__file__).resolve(strict=True)
    root = base_path.parents[3]
    if base_path != root / 'model_executor/layers/fused_moe/routed_experts.py':
        raise RuntimeError('hybrid base class package location mismatch')
    if Path(sys.modules[quant_module].__file__).resolve(strict=True) != root / 'model_executor/layers/quantization/compressed_tensors/compressed_tensors.py':
        raise RuntimeError('hybrid CT class is not from the same vLLM package')
    key = str(root)
    if key not in _RUNTIME_SOURCE_CACHE:
        verified = {}
        for relative, expected in _RUNTIME_SOURCE_PINS.items():
            path = (root / relative).resolve(strict=True)
            if path != root / relative:
                raise RuntimeError('hybrid pinned source redirects outside fixed path')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError('hybrid runtime source mismatch: ' + relative)
            verified[relative] = {'path': str(path), 'sha256': actual}
        _RUNTIME_SOURCE_CACHE[key] = verified
    # New nested dictionaries prevent an owner receipt mutating the pin cache.
    return {name: dict(value) for name, value in _RUNTIME_SOURCE_CACHE[key].items()}


@dataclass(frozen=True)
class HybridShard:
    physical_rank: int
    ep_rank: int
    tp_rank: int
    expert_start: int
    expert_stop: int
    intermediate_start: int
    intermediate_stop: int

    @classmethod
    def for_rank(cls, rank):
        if type(rank) is not int or not 0 <= rank < 4:
            raise ValueError('exact physical TP4 rank required')
        ep, tp = divmod(rank, 2)
        return cls(rank, ep, tp, ep*144, (ep+1)*144, tp*1024, (tp+1)*1024)

    @property
    def identity(self):
        return ('glm53_ep2_tp2_loader_v1', 288, 144, 4096, 2048, 1024,
                self.physical_rank, self.ep_rank, self.tp_rank,
                self.expert_start, self.expert_stop,
                self.intermediate_start, self.intermediate_stop)


def hybrid_routed_experts_class(base_cls, runtime_sources):
    """Use ONLY via FusedMoEFactory(routed_experts_cls=..., routed_experts_args=...).

    Caller checks flag, TP4/SP1/DP1/PCP1, GLM main model, NVFP4, no bias/LoRA/
    EPLB/redundant experts, and leaves the shared expert/attention classes alone.
    No global class/function, environment, or process group is mutated.
    """
    class HybridRoutedExperts(base_cls):
        def _get_quant_method(self, prefix, quant_config, moe_config):
            method = super()._get_quant_method(prefix, quant_config, moe_config)
            if (type(method).__name__ != 'CompressedTensorsW4A4Nvfp4MoEMethod'
                    or type(method).__module__ != 'vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4'
                    or getattr(method, 'group_size', None) != 16
                    or getattr(method, 'use_global_sf', None) is not True
                    or getattr(getattr(method, 'nvfp4_backend', None), 'name', None) != 'FLASHINFER_B12X'):
                raise RuntimeError('hybrid requires actual CT W4A4 NVFP4 B12X before allocation')
            return method

        def __init__(self, layer_name, params_dtype, moe_config, quant_config,
                     expert_map_manager, *, physical_tp_rank,
                     physical_tp_size, **kwargs):
            shard = HybridShard.for_rank(physical_tp_rank)
            pc = moe_config.moe_parallel_config
            actual = (physical_tp_size, pc.tp_size, pc.tp_rank, pc.ep_size,
                      pc.ep_rank, pc.dp_size, pc.pcp_size, pc.sp_size,
                      pc.use_ep, pc.enable_eplb, pc.use_all2all_kernels,
                      moe_config.num_experts, moe_config.num_local_experts,
                      moe_config.hidden_dim, moe_config.intermediate_size,
                      moe_config.intermediate_size_per_partition,
                      moe_config.intermediate_size_per_partition_unpadded,
                      moe_config.experts_per_token, moe_config.has_bias,
                      moe_config.is_lora_enabled, moe_config.skip_final_all_reduce,
                      expert_map_manager.placement_strategy)
            expected = (4, 1, 0, 4, physical_tp_rank, 1, 1, 1, True, False,
                        False, 288, 72, 4096, 2048, 2048, 2048, 8,
                        False, False, False, 'linear')
            if actual != expected or expert_map_manager.moe_parallel_config is not pc:
                raise ValueError('hybrid loader requires exact untouched GLM EP4 input config')
            if (moe_config.moe_backend not in ('flashinfer_b12x', 'b12x')
                    or str(moe_config.in_dtype) != 'torch.bfloat16'
                    or not moe_config.is_act_and_mul):
                raise ValueError('hybrid requires gated B12X BF16 expert backend')
            if kwargs.get('apply_router_weight_on_input', False):
                raise ValueError('hybrid expects routed weight applied after down projection')
            # This replaces ONLY this layer's configuration. The global TP4
            # communicator remains the sole final output sum.
            hybrid_pc = replace(pc, tp_size=2, tp_rank=shard.tp_rank,
                                ep_size=2, ep_rank=shard.ep_rank)
            expert_map_manager.update(hybrid_pc, 288)
            if expert_map_manager.local_num_experts != 144:
                raise RuntimeError('hybrid expert-map allocation did not select E144')
            moe_config.moe_parallel_config = hybrid_pc
            moe_config.num_local_experts = 144
            moe_config.intermediate_size_per_partition = 1024
            moe_config.intermediate_size_per_partition_unpadded = 1024
            moe_config._glm53_hybrid_loader_identity = shard.identity
            moe_config._glm53_hybrid_runtime_sources = {
                name: dict(value) for name, value in runtime_sources.items()}
            # Upstream create_weights now sees E144/I1024 and its existing
            # _load_w13/_load_w2 read matching tp_rank slices of each expert.
            super().__init__(layer_name, params_dtype, moe_config, quant_config,
                             expert_map_manager=expert_map_manager, **kwargs)
            if (type(self.quant_method).__name__ != 'CompressedTensorsW4A4Nvfp4MoEMethod'
                    or self.quant_method.use_global_sf is not True
                    or moe_config.intermediate_size_per_partition != 1024):
                raise RuntimeError('hybrid quant method changed allocation contract')
            expected_shapes = {
                'w13_weight_packed': (144,2048,2048), 'w2_weight_packed': (144,4096,512),
                'w13_weight_scale': (144,2048,256), 'w2_weight_scale': (144,4096,64),
                'w13_weight_global_scale': (144,2), 'w2_weight_global_scale': (144,),
                'w13_input_global_scale': (144,2), 'w2_input_global_scale': (144,),
            }
            if any(tuple(getattr(self,name).shape) != shape
                   for name,shape in expected_shapes.items()):
                raise RuntimeError('hybrid weights/scales allocated with wrong topology')
    return HybridRoutedExperts


def validate_hybrid_loader_identity(value, physical_tp_rank=None):
    if not isinstance(value, tuple) or len(value) != 13:
        raise ValueError('hybrid loader identity must be an exact tuple13')
    if any(type(v) is not int for v in value[1:]):
        raise ValueError('hybrid identity geometry/ranks must be integers')
    expected = HybridShard.for_rank(value[6]).identity
    if value != expected or (physical_tp_rank is not None and value[6] != physical_tp_rank):
        raise ValueError('hybrid loader identity rank/shape/ranges mismatch')
    return value


def hybrid_factory_kwargs(config, parallel_config, quant_config, *,
                          physical_tp_rank, physical_tp_size,
                          main_layer=True, env=None, base_cls=None):
    """Exact initial gate, before constructing or binding a custom class."""
    if env is None:
        import os
        env = os.environ
    flag = env.get('VLLM_GLM53_EP_HYBRID_TP2', '0')
    if flag == '0':
        return {}
    if flag != '1':
        raise ValueError('VLLM_GLM53_EP_HYBRID_TP2 must be exactly 0 or 1')
    if (main_layer is not True or physical_tp_size != 4
            or env.get('VLLM_GLM53_EP_TILED') != '1'
            or env.get('VLLM_GLM53_TP_SF6_Q0', '0') != '0'
            or env.get('VLLM_GLM53_EP_DECODE_OPT', '0') != '0'
            or 'sf6' not in env.get('VLLM_GLM53_B12X_STATIC_V2', '').split(',')):
        raise ValueError('hybrid requires main-model TP4, EP tiled SF6, baseline decode implementation')
    HybridShard.for_rank(physical_tp_rank)
    shape = (getattr(config,'hidden_size',None), getattr(config,'n_routed_experts',None),
             getattr(config,'moe_intermediate_size',None), getattr(config,'num_experts_per_token',None),
             getattr(config,'hidden_act',None), getattr(config,'swiglu_limit',None))
    if shape != (4096,288,2048,8,'silu',10):
        raise ValueError('hybrid requires exact GLM expert geometry/activation')
    if (getattr(parallel_config,'enable_expert_parallel',None) is not True
            or getattr(parallel_config,'enable_eplb',None) is not False
            or getattr(parallel_config,'use_sequence_parallel_moe',None) is not False
            or getattr(parallel_config,'expert_placement_strategy',None) != 'linear'
            or getattr(parallel_config,'tensor_parallel_size',None) != 4
            or any(getattr(parallel_config,n,None) != 1 for n in
                   ('pipeline_parallel_size','data_parallel_size','prefill_context_parallel_size','decode_context_parallel_size'))
            or getattr(getattr(parallel_config,'eplb_config',None),'num_redundant_experts',None) != 0):
        raise ValueError('hybrid requires TP4/DP1/PCP1/DCP1/PP1, linear EP, no SP/EPLB')
    if quant_config is None or quant_config.get_name() != 'compressed-tensors':
        raise ValueError('hybrid requires the actual compressed-tensors configuration')
    # Import/bind only after all exact gates above. Real quant backend and
    # no-bias/LoRA/config checks are repeated before weight allocation.
    if base_cls is None:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        base_cls = RoutedExperts
    runtime_sources = hybrid_runtime_sources(base_cls, quant_config)
    return {'routed_experts_cls': hybrid_routed_experts_class(base_cls, runtime_sources),
            'routed_experts_args': {'physical_tp_rank':physical_tp_rank,
                                   'physical_tp_size':physical_tp_size}}
