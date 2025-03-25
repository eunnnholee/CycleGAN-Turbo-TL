class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 3, 1, 1]", primals_2: "f32[2, 3, 256, 256]", primals_3: "f32[1, 3, 1, 1]", primals_4: "f32[2, 3, 256, 256]", primals_5: "f32[64, 3, 3, 3]", primals_6: "f32[64]", primals_7: "f32[64, 64, 3, 3]", primals_8: "f32[64]", primals_9: "f32[128, 64, 3, 3]", primals_10: "f32[128]", primals_11: "f32[128, 128, 3, 3]", primals_12: "f32[128]", primals_13: "f32[256, 128, 3, 3]", primals_14: "f32[256]", primals_15: "f32[256, 256, 3, 3]", primals_16: "f32[256]", primals_17: "f32[256, 256, 3, 3]", primals_18: "f32[256]", primals_19: "f32[512, 256, 3, 3]", primals_20: "f32[512]", primals_21: "f32[512, 512, 3, 3]", primals_22: "f32[512]", primals_23: "f32[512, 512, 3, 3]", primals_24: "f32[512]", primals_25: "f32[512, 512, 3, 3]", primals_26: "f32[512]", primals_27: "f32[512, 512, 3, 3]", primals_28: "f32[512]", primals_29: "f32[512, 512, 3, 3]", primals_30: "f32[512]", primals_31: "f32[1, 64, 1, 1]", primals_32: "f32[1, 128, 1, 1]", primals_33: "f32[1, 256, 1, 1]", primals_34: "f32[1, 512, 1, 1]", primals_35: "f32[1, 512, 1, 1]"):
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:154 in forward, code: return (inp - self.shift) / self.scale
        sub: "f32[2, 3, 256, 256]" = torch.ops.aten.sub.Tensor(primals_2, primals_1);  primals_2 = None
        div: "f32[2, 3, 256, 256]" = torch.ops.aten.div.Tensor(sub, primals_3);  sub = None
        sub_1: "f32[2, 3, 256, 256]" = torch.ops.aten.sub.Tensor(primals_4, primals_1);  primals_4 = primals_1 = None
        div_1: "f32[2, 3, 256, 256]" = torch.ops.aten.div.Tensor(sub_1, primals_3);  sub_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:121 in forward, code: h = self.slice1(X)
        convolution: "f32[2, 64, 256, 256]" = torch.ops.aten.convolution.default(div, primals_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu: "f32[2, 64, 256, 256]" = torch.ops.aten.relu.default(convolution);  convolution = None
        convolution_1: "f32[2, 64, 256, 256]" = torch.ops.aten.convolution.default(relu, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_1: "f32[2, 64, 256, 256]" = torch.ops.aten.relu.default(convolution_1);  convolution_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:123 in forward, code: h = self.slice2(h)
        _low_memory_max_pool2d_with_offsets = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_1, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem: "f32[2, 64, 128, 128]" = _low_memory_max_pool2d_with_offsets[0]
        getitem_1: "i8[2, 64, 128, 128]" = _low_memory_max_pool2d_with_offsets[1];  _low_memory_max_pool2d_with_offsets = None
        convolution_2: "f32[2, 128, 128, 128]" = torch.ops.aten.convolution.default(getitem, primals_9, primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_2: "f32[2, 128, 128, 128]" = torch.ops.aten.relu.default(convolution_2);  convolution_2 = None
        convolution_3: "f32[2, 128, 128, 128]" = torch.ops.aten.convolution.default(relu_2, primals_11, primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_3: "f32[2, 128, 128, 128]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:125 in forward, code: h = self.slice3(h)
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_3, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_2: "f32[2, 128, 64, 64]" = _low_memory_max_pool2d_with_offsets_1[0]
        getitem_3: "i8[2, 128, 64, 64]" = _low_memory_max_pool2d_with_offsets_1[1];  _low_memory_max_pool2d_with_offsets_1 = None
        convolution_4: "f32[2, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem_2, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_4: "f32[2, 256, 64, 64]" = torch.ops.aten.relu.default(convolution_4);  convolution_4 = None
        convolution_5: "f32[2, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_4, primals_15, primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_5: "f32[2, 256, 64, 64]" = torch.ops.aten.relu.default(convolution_5);  convolution_5 = None
        convolution_6: "f32[2, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_5, primals_17, primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_6: "f32[2, 256, 64, 64]" = torch.ops.aten.relu.default(convolution_6);  convolution_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:127 in forward, code: h = self.slice4(h)
        _low_memory_max_pool2d_with_offsets_2 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_6, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_4: "f32[2, 256, 32, 32]" = _low_memory_max_pool2d_with_offsets_2[0]
        getitem_5: "i8[2, 256, 32, 32]" = _low_memory_max_pool2d_with_offsets_2[1];  _low_memory_max_pool2d_with_offsets_2 = None
        convolution_7: "f32[2, 512, 32, 32]" = torch.ops.aten.convolution.default(getitem_4, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_7: "f32[2, 512, 32, 32]" = torch.ops.aten.relu.default(convolution_7);  convolution_7 = None
        convolution_8: "f32[2, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_7, primals_21, primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_8: "f32[2, 512, 32, 32]" = torch.ops.aten.relu.default(convolution_8);  convolution_8 = None
        convolution_9: "f32[2, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_8, primals_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_9: "f32[2, 512, 32, 32]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:129 in forward, code: h = self.slice5(h)
        _low_memory_max_pool2d_with_offsets_3 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_6: "f32[2, 512, 16, 16]" = _low_memory_max_pool2d_with_offsets_3[0]
        getitem_7: "i8[2, 512, 16, 16]" = _low_memory_max_pool2d_with_offsets_3[1];  _low_memory_max_pool2d_with_offsets_3 = None
        convolution_10: "f32[2, 512, 16, 16]" = torch.ops.aten.convolution.default(getitem_6, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_10: "f32[2, 512, 16, 16]" = torch.ops.aten.relu.default(convolution_10);  convolution_10 = None
        convolution_11: "f32[2, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_10, primals_27, primals_28, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_11: "f32[2, 512, 16, 16]" = torch.ops.aten.relu.default(convolution_11);  convolution_11 = None
        convolution_12: "f32[2, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_11, primals_29, primals_30, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        relu_12: "f32[2, 512, 16, 16]" = torch.ops.aten.relu.default(convolution_12)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:121 in forward, code: h = self.slice1(X)
        convolution_13: "f32[2, 64, 256, 256]" = torch.ops.aten.convolution.default(div_1, primals_5, primals_6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  div_1 = primals_6 = None
        relu_13: "f32[2, 64, 256, 256]" = torch.ops.aten.relu.default(convolution_13);  convolution_13 = None
        convolution_14: "f32[2, 64, 256, 256]" = torch.ops.aten.convolution.default(relu_13, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = primals_8 = None
        relu_14: "f32[2, 64, 256, 256]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:123 in forward, code: h = self.slice2(h)
        _low_memory_max_pool2d_with_offsets_4 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_14, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_8: "f32[2, 64, 128, 128]" = _low_memory_max_pool2d_with_offsets_4[0];  _low_memory_max_pool2d_with_offsets_4 = None
        convolution_15: "f32[2, 128, 128, 128]" = torch.ops.aten.convolution.default(getitem_8, primals_9, primals_10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_8 = primals_10 = None
        relu_15: "f32[2, 128, 128, 128]" = torch.ops.aten.relu.default(convolution_15);  convolution_15 = None
        convolution_16: "f32[2, 128, 128, 128]" = torch.ops.aten.convolution.default(relu_15, primals_11, primals_12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_15 = primals_12 = None
        relu_16: "f32[2, 128, 128, 128]" = torch.ops.aten.relu.default(convolution_16);  convolution_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:125 in forward, code: h = self.slice3(h)
        _low_memory_max_pool2d_with_offsets_5 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_16, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_10: "f32[2, 128, 64, 64]" = _low_memory_max_pool2d_with_offsets_5[0];  _low_memory_max_pool2d_with_offsets_5 = None
        convolution_17: "f32[2, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem_10, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_10 = primals_14 = None
        relu_17: "f32[2, 256, 64, 64]" = torch.ops.aten.relu.default(convolution_17);  convolution_17 = None
        convolution_18: "f32[2, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_17, primals_15, primals_16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_17 = primals_16 = None
        relu_18: "f32[2, 256, 64, 64]" = torch.ops.aten.relu.default(convolution_18);  convolution_18 = None
        convolution_19: "f32[2, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_18, primals_17, primals_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_18 = primals_18 = None
        relu_19: "f32[2, 256, 64, 64]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:127 in forward, code: h = self.slice4(h)
        _low_memory_max_pool2d_with_offsets_6 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_19, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_12: "f32[2, 256, 32, 32]" = _low_memory_max_pool2d_with_offsets_6[0];  _low_memory_max_pool2d_with_offsets_6 = None
        convolution_20: "f32[2, 512, 32, 32]" = torch.ops.aten.convolution.default(getitem_12, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_12 = primals_20 = None
        relu_20: "f32[2, 512, 32, 32]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None
        convolution_21: "f32[2, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_20, primals_21, primals_22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_20 = primals_22 = None
        relu_21: "f32[2, 512, 32, 32]" = torch.ops.aten.relu.default(convolution_21);  convolution_21 = None
        convolution_22: "f32[2, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_21, primals_23, primals_24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_21 = primals_24 = None
        relu_22: "f32[2, 512, 32, 32]" = torch.ops.aten.relu.default(convolution_22);  convolution_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/pretrained_networks.py:129 in forward, code: h = self.slice5(h)
        _low_memory_max_pool2d_with_offsets_7 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(relu_22, [2, 2], [2, 2], [0, 0], [1, 1], False)
        getitem_14: "f32[2, 512, 16, 16]" = _low_memory_max_pool2d_with_offsets_7[0];  _low_memory_max_pool2d_with_offsets_7 = None
        convolution_23: "f32[2, 512, 16, 16]" = torch.ops.aten.convolution.default(getitem_14, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  getitem_14 = primals_26 = None
        relu_23: "f32[2, 512, 16, 16]" = torch.ops.aten.relu.default(convolution_23);  convolution_23 = None
        convolution_24: "f32[2, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_23, primals_27, primals_28, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_23 = primals_28 = None
        relu_24: "f32[2, 512, 16, 16]" = torch.ops.aten.relu.default(convolution_24);  convolution_24 = None
        convolution_25: "f32[2, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_24, primals_29, primals_30, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_24 = primals_30 = None
        relu_25: "f32[2, 512, 16, 16]" = torch.ops.aten.relu.default(convolution_25)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_1: "f32[2, 64, 256, 256]" = torch.ops.aten.pow.Tensor_Scalar(relu_1, 2)
        sum_1: "f32[2, 1, 256, 256]" = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        sqrt: "f32[2, 1, 256, 256]" = torch.ops.aten.sqrt.default(sum_1);  sum_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add: "f32[2, 1, 256, 256]" = torch.ops.aten.add.Tensor(sqrt, 1e-10)
        div_2: "f32[2, 64, 256, 256]" = torch.ops.aten.div.Tensor(relu_1, add)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_2: "f32[2, 64, 256, 256]" = torch.ops.aten.pow.Tensor_Scalar(relu_14, 2)
        sum_2: "f32[2, 1, 256, 256]" = torch.ops.aten.sum.dim_IntList(pow_2, [1], True);  pow_2 = None
        sqrt_1: "f32[2, 1, 256, 256]" = torch.ops.aten.sqrt.default(sum_2);  sum_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_1: "f32[2, 1, 256, 256]" = torch.ops.aten.add.Tensor(sqrt_1, 1e-10);  sqrt_1 = None
        div_3: "f32[2, 64, 256, 256]" = torch.ops.aten.div.Tensor(relu_14, add_1);  relu_14 = add_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        sub_2: "f32[2, 64, 256, 256]" = torch.ops.aten.sub.Tensor(div_2, div_3);  div_3 = None
        pow_3: "f32[2, 64, 256, 256]" = torch.ops.aten.pow.Tensor_Scalar(sub_2, 2)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_4: "f32[2, 128, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(relu_3, 2)
        sum_3: "f32[2, 1, 128, 128]" = torch.ops.aten.sum.dim_IntList(pow_4, [1], True);  pow_4 = None
        sqrt_2: "f32[2, 1, 128, 128]" = torch.ops.aten.sqrt.default(sum_3);  sum_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_2: "f32[2, 1, 128, 128]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-10)
        div_4: "f32[2, 128, 128, 128]" = torch.ops.aten.div.Tensor(relu_3, add_2)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_5: "f32[2, 128, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(relu_16, 2)
        sum_4: "f32[2, 1, 128, 128]" = torch.ops.aten.sum.dim_IntList(pow_5, [1], True);  pow_5 = None
        sqrt_3: "f32[2, 1, 128, 128]" = torch.ops.aten.sqrt.default(sum_4);  sum_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_3: "f32[2, 1, 128, 128]" = torch.ops.aten.add.Tensor(sqrt_3, 1e-10);  sqrt_3 = None
        div_5: "f32[2, 128, 128, 128]" = torch.ops.aten.div.Tensor(relu_16, add_3);  relu_16 = add_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        sub_3: "f32[2, 128, 128, 128]" = torch.ops.aten.sub.Tensor(div_4, div_5);  div_5 = None
        pow_6: "f32[2, 128, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(sub_3, 2)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_7: "f32[2, 256, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(relu_6, 2)
        sum_5: "f32[2, 1, 64, 64]" = torch.ops.aten.sum.dim_IntList(pow_7, [1], True);  pow_7 = None
        sqrt_4: "f32[2, 1, 64, 64]" = torch.ops.aten.sqrt.default(sum_5);  sum_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_4: "f32[2, 1, 64, 64]" = torch.ops.aten.add.Tensor(sqrt_4, 1e-10)
        div_6: "f32[2, 256, 64, 64]" = torch.ops.aten.div.Tensor(relu_6, add_4)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_8: "f32[2, 256, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(relu_19, 2)
        sum_6: "f32[2, 1, 64, 64]" = torch.ops.aten.sum.dim_IntList(pow_8, [1], True);  pow_8 = None
        sqrt_5: "f32[2, 1, 64, 64]" = torch.ops.aten.sqrt.default(sum_6);  sum_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_5: "f32[2, 1, 64, 64]" = torch.ops.aten.add.Tensor(sqrt_5, 1e-10);  sqrt_5 = None
        div_7: "f32[2, 256, 64, 64]" = torch.ops.aten.div.Tensor(relu_19, add_5);  relu_19 = add_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        sub_4: "f32[2, 256, 64, 64]" = torch.ops.aten.sub.Tensor(div_6, div_7);  div_7 = None
        pow_9: "f32[2, 256, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(sub_4, 2)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_10: "f32[2, 512, 32, 32]" = torch.ops.aten.pow.Tensor_Scalar(relu_9, 2)
        sum_7: "f32[2, 1, 32, 32]" = torch.ops.aten.sum.dim_IntList(pow_10, [1], True);  pow_10 = None
        sqrt_6: "f32[2, 1, 32, 32]" = torch.ops.aten.sqrt.default(sum_7);  sum_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_6: "f32[2, 1, 32, 32]" = torch.ops.aten.add.Tensor(sqrt_6, 1e-10)
        div_8: "f32[2, 512, 32, 32]" = torch.ops.aten.div.Tensor(relu_9, add_6)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_11: "f32[2, 512, 32, 32]" = torch.ops.aten.pow.Tensor_Scalar(relu_22, 2)
        sum_8: "f32[2, 1, 32, 32]" = torch.ops.aten.sum.dim_IntList(pow_11, [1], True);  pow_11 = None
        sqrt_7: "f32[2, 1, 32, 32]" = torch.ops.aten.sqrt.default(sum_8);  sum_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_7: "f32[2, 1, 32, 32]" = torch.ops.aten.add.Tensor(sqrt_7, 1e-10);  sqrt_7 = None
        div_9: "f32[2, 512, 32, 32]" = torch.ops.aten.div.Tensor(relu_22, add_7);  relu_22 = add_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        sub_5: "f32[2, 512, 32, 32]" = torch.ops.aten.sub.Tensor(div_8, div_9);  div_9 = None
        pow_12: "f32[2, 512, 32, 32]" = torch.ops.aten.pow.Tensor_Scalar(sub_5, 2)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_13: "f32[2, 512, 16, 16]" = torch.ops.aten.pow.Tensor_Scalar(relu_12, 2)
        sum_9: "f32[2, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(pow_13, [1], True);  pow_13 = None
        sqrt_8: "f32[2, 1, 16, 16]" = torch.ops.aten.sqrt.default(sum_9);  sum_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_8: "f32[2, 1, 16, 16]" = torch.ops.aten.add.Tensor(sqrt_8, 1e-10)
        div_10: "f32[2, 512, 16, 16]" = torch.ops.aten.div.Tensor(relu_12, add_8);  relu_12 = add_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:14 in normalize_tensor, code: norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        pow_14: "f32[2, 512, 16, 16]" = torch.ops.aten.pow.Tensor_Scalar(relu_25, 2)
        sum_10: "f32[2, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(pow_14, [1], True);  pow_14 = None
        sqrt_9: "f32[2, 1, 16, 16]" = torch.ops.aten.sqrt.default(sum_10);  sum_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        add_9: "f32[2, 1, 16, 16]" = torch.ops.aten.add.Tensor(sqrt_9, 1e-10);  sqrt_9 = None
        div_11: "f32[2, 512, 16, 16]" = torch.ops.aten.div.Tensor(relu_25, add_9);  relu_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        sub_6: "f32[2, 512, 16, 16]" = torch.ops.aten.sub.Tensor(div_10, div_11);  div_10 = div_11 = None
        pow_15: "f32[2, 512, 16, 16]" = torch.ops.aten.pow.Tensor_Scalar(sub_6, 2);  sub_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_26: "f32[2, 1, 256, 256]" = torch.ops.aten.convolution.default(pow_3, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        mean: "f32[2, 1, 1, 1]" = torch.ops.aten.mean.dim(convolution_26, [2, 3], True);  convolution_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_27: "f32[2, 1, 128, 128]" = torch.ops.aten.convolution.default(pow_6, primals_32, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        mean_1: "f32[2, 1, 1, 1]" = torch.ops.aten.mean.dim(convolution_27, [2, 3], True);  convolution_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_28: "f32[2, 1, 64, 64]" = torch.ops.aten.convolution.default(pow_9, primals_33, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        mean_2: "f32[2, 1, 1, 1]" = torch.ops.aten.mean.dim(convolution_28, [2, 3], True);  convolution_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_29: "f32[2, 1, 32, 32]" = torch.ops.aten.convolution.default(pow_12, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        mean_3: "f32[2, 1, 1, 1]" = torch.ops.aten.mean.dim(convolution_29, [2, 3], True);  convolution_29 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:167 in forward, code: return self.model(x)
        convolution_30: "f32[2, 1, 16, 16]" = torch.ops.aten.convolution.default(pow_15, primals_35, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:15 in spatial_average, code: return in_tens.mean([2,3],keepdim=keepdim)
        mean_4: "f32[2, 1, 1, 1]" = torch.ops.aten.mean.dim(convolution_30, [2, 3], True);  convolution_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:139 in forward, code: val += res[l]
        add_10: "f32[2, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean, 0);  mean = None
        add_11: "f32[2, 1, 1, 1]" = torch.ops.aten.add.Tensor(add_10, mean_1);  add_10 = mean_1 = None
        add_12: "f32[2, 1, 1, 1]" = torch.ops.aten.add.Tensor(add_11, mean_2);  add_11 = mean_2 = None
        add_13: "f32[2, 1, 1, 1]" = torch.ops.aten.add.Tensor(add_12, mean_3);  add_12 = mean_3 = None
        add_14: "f32[2, 1, 1, 1]" = torch.ops.aten.add.Tensor(add_13, mean_4);  add_13 = mean_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        pow_18: "f32[2, 512, 32, 32]" = torch.ops.aten.pow.Tensor_Scalar(sub_5, 1.0);  sub_5 = None
        mul_6: "f32[2, 512, 32, 32]" = torch.ops.aten.mul.Scalar(pow_18, 2.0);  pow_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        div_22: "f32[2, 512, 32, 32]" = torch.ops.aten.div.Tensor(div_8, add_6);  div_8 = add_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        pow_20: "f32[2, 256, 64, 64]" = torch.ops.aten.pow.Tensor_Scalar(sub_4, 1.0);  sub_4 = None
        mul_12: "f32[2, 256, 64, 64]" = torch.ops.aten.mul.Scalar(pow_20, 2.0);  pow_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        div_26: "f32[2, 256, 64, 64]" = torch.ops.aten.div.Tensor(div_6, add_4);  div_6 = add_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        pow_22: "f32[2, 128, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(sub_3, 1.0);  sub_3 = None
        mul_18: "f32[2, 128, 128, 128]" = torch.ops.aten.mul.Scalar(pow_22, 2.0);  pow_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        div_30: "f32[2, 128, 128, 128]" = torch.ops.aten.div.Tensor(div_4, add_2);  div_4 = add_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/lpips.py:124 in forward, code: diffs[kk] = (feats0[kk]-feats1[kk])**2
        pow_24: "f32[2, 64, 256, 256]" = torch.ops.aten.pow.Tensor_Scalar(sub_2, 1.0);  sub_2 = None
        mul_24: "f32[2, 64, 256, 256]" = torch.ops.aten.mul.Scalar(pow_24, 2.0);  pow_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/lpips/__init__.py:15 in normalize_tensor, code: return in_feat/(norm_factor+eps)
        div_34: "f32[2, 64, 256, 256]" = torch.ops.aten.div.Tensor(div_2, add);  div_2 = add = None
        return (add_14, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, div, relu, relu_1, getitem, getitem_1, relu_2, relu_3, getitem_2, getitem_3, relu_4, relu_5, relu_6, getitem_4, getitem_5, relu_7, relu_8, relu_9, getitem_6, getitem_7, relu_10, relu_11, convolution_12, convolution_25, sqrt, pow_3, sqrt_2, pow_6, sqrt_4, pow_9, sqrt_6, pow_12, sqrt_8, add_9, pow_15, mul_6, div_22, mul_12, div_26, mul_18, div_30, mul_24, div_34)
        