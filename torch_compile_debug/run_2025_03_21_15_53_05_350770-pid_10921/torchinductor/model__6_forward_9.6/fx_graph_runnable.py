
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config.allowed_functions_module_string_ignorelist = {'torch.distributions', 'torch.testing', 'torch._prims', 'torch._refs', 'torch._decomp'}
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'repro_level', 'skipfiles_inline_module_allowlist', 'constant_functions', 'repro_after'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._dynamo.config.ignore_logger_methods = set()
torch._dynamo.config._autograd_backward_strict_mode_banned_ops = ['stride', 'requires_grad', 'storage_offset', 'layout', 'data', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_ipu', 'is_leaf', 'is_maia', 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu']
torch._dynamo.config.compiled_autograd_kwargs_override = {}
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.comprehensive_padding = True
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.6.0+cu118
# torch cuda version: 11.8
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Wed_Sep_21_10:33:58_PDT_2022 
# Cuda compilation tools, release 11.8, V11.8.89 
# Build cuda_11.8.r11.8/compiler.31833905_0 

# GPU Hardware Info: 
# NVIDIA A100 80GB PCIe MIG 3g.40gb : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
        sub = torch.ops.aten.sub.Tensor(1, primals_2)
        sub_1 = torch.ops.aten.sub.Tensor(1, primals_1)
        div = torch.ops.aten.div.Tensor(primals_2, primals_1)
        sub_2 = torch.ops.aten.sub.Tensor(1, div)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 0.5)
        mul = torch.ops.aten.mul.Tensor(pow_1, primals_3);  pow_1 = primals_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(primals_4, mul);  mul = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(primals_2, 0.5)
        div_1 = torch.ops.aten.div.Tensor(sub_3, pow_2);  sub_3 = pow_2 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(primals_1, 0.5)
        mul_1 = torch.ops.aten.mul.Tensor(pow_3, sub_2);  pow_3 = sub_2 = None
        div_2 = torch.ops.aten.div.Tensor(mul_1, sub);  mul_1 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(div, 0.5);  div = None
        mul_2 = torch.ops.aten.mul.Tensor(pow_4, sub_1);  pow_4 = sub_1 = None
        div_3 = torch.ops.aten.div.Tensor(mul_2, sub);  mul_2 = sub = None
        mul_3 = torch.ops.aten.mul.Tensor(div_2, div_1);  div_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(div_3, primals_4);  div_3 = primals_4 = None
        add = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        gt = torch.ops.aten.gt.Scalar(primals_5, 0);  primals_5 = None
        return (gt, div_1, add, primals_1, primals_2)
        
def load_args(reader):
    buf0 = reader.storage(None, 4)
    reader.tensor(buf0, (), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (), storage_offset=999, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf2, (4, 32, 32), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf3, (4, 32, 32), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf4, (), dtype=torch.int64, is_leaf=True)  # primals_5
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)