# AOT ID: ['6_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_elicer/ju/cju3pmixw4f3zn6knrihvphy5dme4ci7mkzir5tlsjv4hznbqczy.py
# Topologically Sorted Source Nodes: [beta_prod_t, beta_prod_t_prev, current_alpha_t, current_beta_t, pow_1, mul, sub_3, pow_2, pred_original_sample, pow_3, mul_1, pred_original_sample_coeff, pow_4, mul_2, current_sample_coeff, mul_3, mul_4, pred_prev_sample], Original ATen: [aten.rsub, aten.div, aten.pow, aten.mul, aten.sub, aten.add]
# Source node to ATen node mapping:
#   beta_prod_t => sub
#   beta_prod_t_prev => sub_1
#   current_alpha_t => div
#   current_beta_t => sub_2
#   current_sample_coeff => div_3
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pred_original_sample => div_1
#   pred_original_sample_coeff => div_2
#   pred_prev_sample => add
#   sub_3 => sub_3
# Graph fragment:
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %primals_2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %primals_1), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, %primals_3), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %mul), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_2, 0.5), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, %pow_2), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_3, %sub_2), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_1, %sub), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div, 0.5), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_4, %sub_1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %sub), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %div_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %primals_4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_4), kwargs = {})
triton_poi_fused_add_div_mul_pow_rsub_sub_0 = async_compile.triton('triton_poi_fused_add_div_mul_pow_rsub_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_pow_rsub_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_pow_rsub_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp11 = in_ptr3
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = libdevice.sqrt(tmp4)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 - tmp7
    tmp9 = libdevice.sqrt(tmp2)
    tmp10 = tmp8 / tmp9
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tmp2 / tmp11
    tmp14 = tmp3 - tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp15 / tmp4
    tmp17 = tmp16 * tmp10
    tmp18 = libdevice.sqrt(tmp13)
    tmp19 = tmp3 - tmp11
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20 / tmp4
    tmp22 = tmp21 * tmp0
    tmp23 = tmp17 + tmp22
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp23, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_elicer/ec/cecsc2wnp55hpvdre46akhmyk25b35wca4tj3neicrlrcbknhy3y.py
# Topologically Sorted Source Nodes: [gt], Original ATen: [aten.gt]
# Source node to ATen node mapping:
#   gt => gt
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%primals_5, 0), kwargs = {})
triton_poi_fused_gt_1 = async_compile.triton('triton_poi_fused_gt_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=42, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gt_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'D805D6112358B1404BAFE009681C8B83D24CEB9D69C06D6B675EC44CAB42D254', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gt_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 > tmp2
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (), ())
    assert_size_stride(primals_2, (), ())
    assert_size_stride(primals_3, (4, 32, 32), (1024, 32, 1))
    assert_size_stride(primals_4, (4, 32, 32), (1024, 32, 1))
    assert_size_stride(primals_5, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 32, 32), (1024, 32, 1), torch.float32)
        buf1 = empty_strided_cuda((4, 32, 32), (1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [beta_prod_t, beta_prod_t_prev, current_alpha_t, current_beta_t, pow_1, mul, sub_3, pow_2, pred_original_sample, pow_3, mul_1, pred_original_sample_coeff, pow_4, mul_2, current_sample_coeff, mul_3, mul_4, pred_prev_sample], Original ATen: [aten.rsub, aten.div, aten.pow, aten.mul, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_pow_rsub_sub_0.run(primals_4, primals_2, primals_3, primals_1.item(), buf0, buf1, 4096, grid=grid(4096), stream=stream0)
        del primals_3
        del primals_4
        buf2 = empty_strided_cuda((), (), torch.bool)
        # Topologically Sorted Source Nodes: [gt], Original ATen: [aten.gt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gt_1.run(primals_5, buf2, 1, grid=grid(1), stream=stream0)
        del primals_5
    return (buf2, buf0, buf1, primals_1, primals_2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 32, 32), (1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
