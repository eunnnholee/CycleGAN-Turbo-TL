class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 32, 32]", primals_2: "f32[4, 32, 32]", primals_3: "f32[4, 32, 32]", primals_4: "f32[4, 32, 32]", primals_5: "f32[4, 4, 1, 1]", primals_6: "f32[4]", primals_7: "f32[512, 4, 3, 3]", primals_8: "f32[512]", primals_9: "f32[4, 4, 3, 3]", primals_10: "f32[512, 4, 1, 1]", primals_11: "f32[512]", primals_12: "f32[512]", primals_13: "f32[512, 512, 3, 3]", primals_14: "f32[512]", primals_15: "f32[4, 512, 3, 3]", primals_16: "f32[512, 4, 1, 1]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[512, 512, 3, 3]", primals_20: "f32[512]", primals_21: "f32[4, 512, 3, 3]", primals_22: "f32[512, 4, 1, 1]", primals_23: "f32[512]", primals_24: "f32[512]", primals_25: "f32[512, 512, 3, 3]", primals_26: "f32[512]", primals_27: "f32[4, 512, 3, 3]", primals_28: "f32[512, 4, 1, 1]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512, 512, 3, 3]", primals_32: "f32[512]", primals_33: "f32[4, 512, 3, 3]", primals_34: "f32[512, 4, 1, 1]", primals_35: "f32[512]", primals_36: "f32[512]", primals_37: "f32[512, 512, 3, 3]", primals_38: "f32[512]", primals_39: "f32[4, 512, 3, 3]", primals_40: "f32[512, 4, 1, 1]", primals_41: "f32[512]", primals_42: "f32[512]", primals_43: "f32[512, 512, 3, 3]", primals_44: "f32[512]", primals_45: "f32[4, 512, 3, 3]", primals_46: "f32[512, 4, 1, 1]", primals_47: "f32[512, 512, 3, 3]", primals_48: "f32[512]", primals_49: "f32[4, 512, 3, 3]", primals_50: "f32[512, 4, 1, 1]", primals_51: "f32[512]", primals_52: "f32[512]", primals_53: "f32[512, 512, 3, 3]", primals_54: "f32[512]", primals_55: "f32[4, 512, 3, 3]", primals_56: "f32[512, 4, 1, 1]", primals_57: "f32[512]", primals_58: "f32[512]", primals_59: "f32[512, 512, 3, 3]", primals_60: "f32[512]", primals_61: "f32[4, 512, 3, 3]", primals_62: "f32[512, 4, 1, 1]", primals_63: "f32[512]", primals_64: "f32[512]", primals_65: "f32[512, 512, 3, 3]", primals_66: "f32[512]", primals_67: "f32[4, 512, 3, 3]", primals_68: "f32[512, 4, 1, 1]", primals_69: "f32[512]", primals_70: "f32[512]", primals_71: "f32[512, 512, 3, 3]", primals_72: "f32[512]", primals_73: "f32[4, 512, 3, 3]", primals_74: "f32[512, 4, 1, 1]", primals_75: "f32[512]", primals_76: "f32[512]", primals_77: "f32[512, 512, 3, 3]", primals_78: "f32[512]", primals_79: "f32[4, 512, 3, 3]", primals_80: "f32[512, 4, 1, 1]", primals_81: "f32[512]", primals_82: "f32[512]", primals_83: "f32[512, 512, 3, 3]", primals_84: "f32[512]", primals_85: "f32[4, 512, 3, 3]", primals_86: "f32[512, 4, 1, 1]", primals_87: "f32[512, 512, 3, 3]", primals_88: "f32[512]", primals_89: "f32[4, 512, 3, 3]", primals_90: "f32[512, 4, 1, 1]", primals_91: "f32[512]", primals_92: "f32[512]", primals_93: "f32[256, 512, 3, 3]", primals_94: "f32[256]", primals_95: "f32[4, 512, 3, 3]", primals_96: "f32[256, 4, 1, 1]", primals_97: "f32[256]", primals_98: "f32[256]", primals_99: "f32[256, 256, 3, 3]", primals_100: "f32[256]", primals_101: "f32[4, 256, 3, 3]", primals_102: "f32[256, 4, 1, 1]", primals_103: "f32[256, 512, 1, 1]", primals_104: "f32[256]", primals_105: "f32[4, 512, 1, 1]", primals_106: "f32[256, 4, 1, 1]", primals_107: "f32[256]", primals_108: "f32[256]", primals_109: "f32[256, 256, 3, 3]", primals_110: "f32[256]", primals_111: "f32[4, 256, 3, 3]", primals_112: "f32[256, 4, 1, 1]", primals_113: "f32[256]", primals_114: "f32[256]", primals_115: "f32[256, 256, 3, 3]", primals_116: "f32[256]", primals_117: "f32[4, 256, 3, 3]", primals_118: "f32[256, 4, 1, 1]", primals_119: "f32[256]", primals_120: "f32[256]", primals_121: "f32[256, 256, 3, 3]", primals_122: "f32[256]", primals_123: "f32[4, 256, 3, 3]", primals_124: "f32[256, 4, 1, 1]", primals_125: "f32[256]", primals_126: "f32[256]", primals_127: "f32[256, 256, 3, 3]", primals_128: "f32[256]", primals_129: "f32[4, 256, 3, 3]", primals_130: "f32[256, 4, 1, 1]", primals_131: "f32[256, 256, 3, 3]", primals_132: "f32[256]", primals_133: "f32[4, 256, 3, 3]", primals_134: "f32[256, 4, 1, 1]", primals_135: "f32[256]", primals_136: "f32[256]", primals_137: "f32[128, 256, 3, 3]", primals_138: "f32[128]", primals_139: "f32[4, 256, 3, 3]", primals_140: "f32[128, 4, 1, 1]", primals_141: "f32[128]", primals_142: "f32[128]", primals_143: "f32[128, 128, 3, 3]", primals_144: "f32[128]", primals_145: "f32[4, 128, 3, 3]", primals_146: "f32[128, 4, 1, 1]", primals_147: "f32[128, 256, 1, 1]", primals_148: "f32[128]", primals_149: "f32[4, 256, 1, 1]", primals_150: "f32[128, 4, 1, 1]", primals_151: "f32[128]", primals_152: "f32[128]", primals_153: "f32[128, 128, 3, 3]", primals_154: "f32[128]", primals_155: "f32[4, 128, 3, 3]", primals_156: "f32[128, 4, 1, 1]", primals_157: "f32[128]", primals_158: "f32[128]", primals_159: "f32[128, 128, 3, 3]", primals_160: "f32[128]", primals_161: "f32[4, 128, 3, 3]", primals_162: "f32[128, 4, 1, 1]", primals_163: "f32[128]", primals_164: "f32[128]", primals_165: "f32[128, 128, 3, 3]", primals_166: "f32[128]", primals_167: "f32[4, 128, 3, 3]", primals_168: "f32[128, 4, 1, 1]", primals_169: "f32[128]", primals_170: "f32[128]", primals_171: "f32[128, 128, 3, 3]", primals_172: "f32[128]", primals_173: "f32[4, 128, 3, 3]", primals_174: "f32[128, 4, 1, 1]", primals_175: "f32[512]", primals_176: "f32[512]", primals_177: "f32[512, 512, 3, 3]", primals_178: "f32[512]", primals_179: "f32[4, 512, 3, 3]", primals_180: "f32[512, 4, 1, 1]", primals_181: "f32[512]", primals_182: "f32[512]", primals_183: "f32[512, 512, 3, 3]", primals_184: "f32[512]", primals_185: "f32[4, 512, 3, 3]", primals_186: "f32[512, 4, 1, 1]", primals_187: "f32[512]", primals_188: "f32[512]", primals_189: "f32[512, 512]", primals_190: "f32[512]", primals_191: "f32[4, 512]", primals_192: "f32[512, 4]", primals_193: "f32[512, 512]", primals_194: "f32[512]", primals_195: "f32[4, 512]", primals_196: "f32[512, 4]", primals_197: "f32[512, 512]", primals_198: "f32[512]", primals_199: "f32[4, 512]", primals_200: "f32[512, 4]", primals_201: "f32[512, 512]", primals_202: "f32[512]", primals_203: "f32[4, 512]", primals_204: "f32[512, 4]", primals_205: "f32[512]", primals_206: "f32[512]", primals_207: "f32[512, 512, 3, 3]", primals_208: "f32[512]", primals_209: "f32[4, 512, 3, 3]", primals_210: "f32[512, 4, 1, 1]", primals_211: "f32[512]", primals_212: "f32[512]", primals_213: "f32[512, 512, 3, 3]", primals_214: "f32[512]", primals_215: "f32[4, 512, 3, 3]", primals_216: "f32[512, 4, 1, 1]", primals_217: "f32[4, 512, 32, 32]", primals_218: "f32[512, 512, 1, 1]", primals_219: "f32[4, 512, 1, 1]", primals_220: "f32[512, 4, 1, 1]", primals_221: "f32[4, 256, 64, 64]", primals_222: "f32[512, 256, 1, 1]", primals_223: "f32[4, 256, 1, 1]", primals_224: "f32[512, 4, 1, 1]", primals_225: "f32[4, 128, 128, 128]", primals_226: "f32[512, 128, 1, 1]", primals_227: "f32[4, 128, 1, 1]", primals_228: "f32[512, 4, 1, 1]", primals_229: "f32[4, 128, 256, 256]", primals_230: "f32[256, 128, 1, 1]", primals_231: "f32[4, 128, 1, 1]", primals_232: "f32[256, 4, 1, 1]", primals_233: "f32[128]", primals_234: "f32[128]", primals_235: "f32[3, 128, 3, 3]", primals_236: "f32[3]", primals_237: "f32[4, 128, 3, 3]", primals_238: "f32[3, 4, 1, 1]"):
         # File: /home/elicer/cyclegan-turbo/src/cyclegan_turbo.py:210 in torch_dynamo_resume_in_forward_with_networks_at_210, code: x_out = torch.stack([
        cat: "f32[16, 32, 32]" = torch.ops.aten.cat.default([primals_4, primals_3, primals_2, primals_1]);  primals_4 = primals_3 = primals_2 = primals_1 = None
        view: "f32[4, 4, 32, 32]" = torch.ops.aten.reshape.default(cat, [4, 4, 32, 32]);  cat = None
        
         # File: /home/elicer/cyclegan-turbo/src/cyclegan_turbo.py:44 in forward, code: x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        div: "f32[4, 4, 32, 32]" = torch.ops.aten.div.Tensor(view, 0.18215);  view = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/autoencoders/autoencoder_kl.py:296 in _decode, code: z = self.post_quant_conv(z)
        convolution: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(div, primals_5, primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_1: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution, primals_7, primals_8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_2: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(convolution, primals_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_3: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_2, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_3, 2.0);  convolution_3 = None
        add: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_1, mul);  convolution_1 = mul = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        view_1: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(add, [4, 32, 16, 1024])
        var_mean = torch.ops.aten.var_mean.correction(view_1, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[4, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[4, 32, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_1, getitem_1);  view_1 = None
        mul_1: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        view_2: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_1, [4, 512, 32, 32]);  mul_1 = None
        unsqueeze: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_176, 0)
        unsqueeze_1: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_175, 0)
        unsqueeze_4: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_2: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_2, unsqueeze_5);  view_2 = unsqueeze_5 = None
        add_2: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_2);  mul_2 = unsqueeze_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_2)
        mul_3: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_2, sigmoid);  add_2 = sigmoid = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_4: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_3, primals_177, primals_178, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_178 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_5: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_3, primals_179, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_6: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_5, primals_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_4: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_6, 2.0);  convolution_6 = None
        add_3: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_4, mul_4);  convolution_4 = mul_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_3: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(add_3, [4, 32, 16, 1024])
        var_mean_1 = torch.ops.aten.var_mean.correction(view_3, [2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[4, 32, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[4, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
        rsqrt_1: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_3, getitem_3);  view_3 = None
        mul_5: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        view_4: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_5, [4, 512, 32, 32]);  mul_5 = None
        unsqueeze_6: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_182, 0)
        unsqueeze_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_181, 0)
        unsqueeze_10: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_6: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_4, unsqueeze_11);  view_4 = unsqueeze_11 = None
        add_5: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_8);  mul_6 = unsqueeze_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_1: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_5)
        mul_7: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_5, sigmoid_1);  add_5 = sigmoid_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_7: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_7, primals_183, primals_184, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_184 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_8: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_7, primals_185, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_9: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_8, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_8: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_9, 2.0);  convolution_9 = None
        add_6: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_7, mul_8);  convolution_7 = mul_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_7: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(add, add_6);  add_6 = None
        div_1: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_7, 1);  add_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3246 in __call__, code: hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        view_5: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(div_1, [4, 512, 1024])
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3259 in __call__, code: hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        view_6: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(view_5, [4, 32, 16, 1024])
        var_mean_2 = torch.ops.aten.var_mean.correction(view_6, [2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[4, 32, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[4, 32, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
        rsqrt_2: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_2: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_6, getitem_5);  view_6 = None
        mul_9: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        view_7: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mul_9, [4, 512, 1024]);  mul_9 = None
        unsqueeze_12: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_188, 0);  primals_188 = None
        unsqueeze_13: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_187, 0)
        unsqueeze_15: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 2);  unsqueeze_14 = None
        mul_10: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_7, unsqueeze_15);  view_7 = unsqueeze_15 = None
        add_9: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_10, unsqueeze_13);  mul_10 = unsqueeze_13 = None
        squeeze_4: "f32[4, 32]" = torch.ops.aten.squeeze.dims(getitem_5, [2, 3]);  getitem_5 = None
        squeeze_5: "f32[4, 32]" = torch.ops.aten.squeeze.dims(rsqrt_2, [2, 3]);  rsqrt_2 = None
        permute_2: "f32[4, 1024, 512]" = torch.ops.aten.permute.default(add_9, [0, 2, 1]);  add_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_3: "f32[512, 512]" = torch.ops.aten.permute.default(primals_189, [1, 0])
        expand: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(permute_2, [4, 1024, 512])
        expand_1: "f32[4, 512, 512]" = torch.ops.aten.expand.default(permute_3, [4, 512, 512]);  permute_3 = None
        bmm: "f32[4, 1024, 512]" = torch.ops.aten.bmm.default(expand, expand_1);  expand_1 = None
        add_10: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(bmm, primals_190);  bmm = primals_190 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_4: "f32[512, 4]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
        clone_1: "f32[4, 1024, 512]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_11: "f32[4096, 512]" = torch.ops.aten.reshape.default(clone_1, [4096, 512]);  clone_1 = None
        mm: "f32[4096, 4]" = torch.ops.aten.mm.default(view_11, permute_4)
        permute_5: "f32[4, 512]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        mm_1: "f32[4096, 512]" = torch.ops.aten.mm.default(mm, permute_5)
        view_14: "f32[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_1, [4, 1024, 512]);  mm_1 = None
        mul_11: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_14, 2.0);  view_14 = None
        add_11: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_10, mul_11);  add_10 = mul_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_6: "f32[512, 512]" = torch.ops.aten.permute.default(primals_193, [1, 0])
        expand_3: "f32[4, 512, 512]" = torch.ops.aten.expand.default(permute_6, [4, 512, 512]);  permute_6 = None
        bmm_1: "f32[4, 1024, 512]" = torch.ops.aten.bmm.default(expand, expand_3);  expand_3 = None
        add_12: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(bmm_1, primals_194);  bmm_1 = primals_194 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_7: "f32[512, 4]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
        mm_2: "f32[4096, 4]" = torch.ops.aten.mm.default(view_11, permute_7)
        permute_8: "f32[4, 512]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        mm_3: "f32[4096, 512]" = torch.ops.aten.mm.default(mm_2, permute_8)
        view_21: "f32[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_3, [4, 1024, 512]);  mm_3 = None
        mul_12: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_21, 2.0);  view_21 = None
        add_13: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_12, mul_12);  add_12 = mul_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_197, [1, 0])
        expand_5: "f32[4, 512, 512]" = torch.ops.aten.expand.default(permute_9, [4, 512, 512]);  permute_9 = None
        bmm_2: "f32[4, 1024, 512]" = torch.ops.aten.bmm.default(expand, expand_5);  expand = expand_5 = None
        add_14: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(bmm_2, primals_198);  bmm_2 = primals_198 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_10: "f32[512, 4]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
        mm_4: "f32[4096, 4]" = torch.ops.aten.mm.default(view_11, permute_10)
        permute_11: "f32[4, 512]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        mm_5: "f32[4096, 512]" = torch.ops.aten.mm.default(mm_4, permute_11)
        view_28: "f32[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_5, [4, 1024, 512]);  mm_5 = None
        mul_13: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_28, 2.0);  view_28 = None
        add_15: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_14, mul_13);  add_14 = mul_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3286 in __call__, code: hidden_states = F.scaled_dot_product_attention(
        view_32: "f32[4, 1024, 1, 512]" = torch.ops.aten.reshape.default(add_11, [4, -1, 1, 512]);  add_11 = None
        permute_15: "f32[4, 1, 1024, 512]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        view_33: "f32[4, 1024, 1, 512]" = torch.ops.aten.reshape.default(add_13, [4, -1, 1, 512]);  add_13 = None
        permute_16: "f32[4, 1, 1024, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        view_34: "f32[4, 1024, 1, 512]" = torch.ops.aten.reshape.default(add_15, [4, -1, 1, 512]);  add_15 = None
        permute_17: "f32[4, 1, 1024, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_15, permute_16, permute_17, None, True)
        getitem_6: "f32[4, 1, 1024, 512]" = _scaled_dot_product_efficient_attention[0]
        getitem_7: "f32[4, 1, 1024]" = _scaled_dot_product_efficient_attention[1]
        getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_9: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3290 in __call__, code: hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        permute_18: "f32[4, 1024, 1, 512]" = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3])
        view_35: "f32[4, 1024, 512]" = torch.ops.aten.reshape.default(permute_18, [4, -1, 512]);  permute_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_36: "f32[4096, 512]" = torch.ops.aten.reshape.default(view_35, [4096, 512]);  view_35 = None
        permute_19: "f32[512, 512]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[4096, 512]" = torch.ops.aten.mm.default(view_36, permute_19)
        add_tensor: "f32[4096, 512]" = torch.ops.aten.add.Tensor(mm_default, primals_202);  mm_default = primals_202 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        view_37: "f32[4, 1024, 512]" = torch.ops.aten.reshape.default(add_tensor, [4, 1024, 512]);  add_tensor = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_20: "f32[512, 4]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        mm_6: "f32[4096, 4]" = torch.ops.aten.mm.default(view_36, permute_20);  view_36 = None
        permute_21: "f32[4, 512]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
        mm_7: "f32[4096, 512]" = torch.ops.aten.mm.default(mm_6, permute_21)
        view_41: "f32[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_7, [4, 1024, 512]);  mm_7 = None
        mul_14: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(view_41, 2.0);  view_41 = None
        add_16: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(view_37, mul_14);  view_37 = mul_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3299 in __call__, code: hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        permute_22: "f32[4, 512, 1024]" = torch.ops.aten.permute.default(add_16, [0, 2, 1]);  add_16 = None
        view_45: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(permute_22, [4, 512, 32, 32]);  permute_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3302 in __call__, code: hidden_states = hidden_states + residual
        add_17: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(view_45, div_1);  view_45 = div_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/attention_processor.py:3304 in __call__, code: hidden_states = hidden_states / attn.rescale_output_factor
        div_2: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_17, 1);  add_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_5: "f32[4, 512, 32, 32]" = torch.ops.aten.clone.default(div_2, memory_format = torch.contiguous_format)
        view_46: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(clone_5, [4, 32, 16, 1024])
        var_mean_3 = torch.ops.aten.var_mean.correction(view_46, [2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[4, 32, 1, 1]" = var_mean_3[0]
        getitem_11: "f32[4, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_18: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
        rsqrt_3: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_3: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_46, getitem_11);  view_46 = None
        mul_15: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        view_47: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_15, [4, 512, 32, 32]);  mul_15 = None
        unsqueeze_16: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_206, 0)
        unsqueeze_17: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 2);  unsqueeze_16 = None
        unsqueeze_18: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        unsqueeze_19: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_205, 0)
        unsqueeze_20: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 2);  unsqueeze_19 = None
        unsqueeze_21: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 3);  unsqueeze_20 = None
        mul_16: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_47, unsqueeze_21);  view_47 = unsqueeze_21 = None
        add_19: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_16, unsqueeze_18);  mul_16 = unsqueeze_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_2: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_19)
        mul_17: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_19, sigmoid_2);  add_19 = sigmoid_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_10: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_17, primals_207, primals_208, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_208 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_11: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_17, primals_209, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_12: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_11, primals_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_18: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_12, 2.0);  convolution_12 = None
        add_20: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_10, mul_18);  convolution_10 = mul_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:351 in forward, code: hidden_states = self.norm2(hidden_states)
        view_48: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(add_20, [4, 32, 16, 1024])
        var_mean_4 = torch.ops.aten.var_mean.correction(view_48, [2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[4, 32, 1, 1]" = var_mean_4[0]
        getitem_13: "f32[4, 32, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
        rsqrt_4: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_48, getitem_13);  view_48 = None
        mul_19: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        view_49: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_19, [4, 512, 32, 32]);  mul_19 = None
        unsqueeze_22: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_212, 0)
        unsqueeze_23: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 2);  unsqueeze_22 = None
        unsqueeze_24: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 3);  unsqueeze_23 = None
        unsqueeze_25: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_211, 0)
        unsqueeze_26: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 2);  unsqueeze_25 = None
        unsqueeze_27: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 3);  unsqueeze_26 = None
        mul_20: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_49, unsqueeze_27);  view_49 = unsqueeze_27 = None
        add_22: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_24);  mul_20 = unsqueeze_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_3: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_22)
        mul_21: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_22, sigmoid_3);  add_22 = sigmoid_3 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_13: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_21, primals_213, primals_214, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_214 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_14: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_21, primals_215, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_15: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_14, primals_216, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_22: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_15, 2.0);  convolution_15 = None
        add_23: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_13, mul_22);  convolution_13 = mul_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_24: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(div_2, add_23);  div_2 = add_23 = None
        div_3: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_24, 1);  add_24 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:40 in my_vae_decoder_fwd, code: skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
        mul_23: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(primals_217, 1);  primals_217 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_16: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_23, primals_218, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_17: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_23, primals_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_18: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_17, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_24: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_18, 2.0);  convolution_18 = None
        add_25: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_16, mul_24);  convolution_16 = mul_24 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:42 in my_vae_decoder_fwd, code: sample = sample + skip_in
        add_26: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(div_3, add_25);  div_3 = add_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_7: "f32[4, 512, 32, 32]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
        view_50: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(clone_7, [4, 32, 16, 1024])
        var_mean_5 = torch.ops.aten.var_mean.correction(view_50, [2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[4, 32, 1, 1]" = var_mean_5[0]
        getitem_15: "f32[4, 32, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_27: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_5: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_50, getitem_15);  view_50 = None
        mul_25: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        view_51: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_25, [4, 512, 32, 32]);  mul_25 = None
        unsqueeze_28: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_12, 0)
        unsqueeze_29: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 2);  unsqueeze_28 = None
        unsqueeze_30: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 3);  unsqueeze_29 = None
        unsqueeze_31: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_11, 0)
        unsqueeze_32: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 2);  unsqueeze_31 = None
        unsqueeze_33: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 3);  unsqueeze_32 = None
        mul_26: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_51, unsqueeze_33);  view_51 = unsqueeze_33 = None
        add_28: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_30);  mul_26 = unsqueeze_30 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_4: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_28)
        mul_27: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_28, sigmoid_4);  add_28 = sigmoid_4 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_19: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_27, primals_13, primals_14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_20: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_27, primals_15, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_21: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_20, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_28: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_21, 2.0);  convolution_21 = None
        add_29: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_19, mul_28);  convolution_19 = mul_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_52: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(add_29, [4, 32, 16, 1024])
        var_mean_6 = torch.ops.aten.var_mean.correction(view_52, [2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[4, 32, 1, 1]" = var_mean_6[0]
        getitem_17: "f32[4, 32, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_30: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_6: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_6: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_52, getitem_17);  view_52 = None
        mul_29: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        view_53: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_29, [4, 512, 32, 32]);  mul_29 = None
        unsqueeze_34: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_18, 0)
        unsqueeze_35: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 2);  unsqueeze_34 = None
        unsqueeze_36: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, 3);  unsqueeze_35 = None
        unsqueeze_37: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_17, 0)
        unsqueeze_38: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 2);  unsqueeze_37 = None
        unsqueeze_39: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 3);  unsqueeze_38 = None
        mul_30: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_53, unsqueeze_39);  view_53 = unsqueeze_39 = None
        add_31: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_30, unsqueeze_36);  mul_30 = unsqueeze_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_5: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_31)
        mul_31: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_31, sigmoid_5);  add_31 = sigmoid_5 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_22: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_31, primals_19, primals_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_23: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_31, primals_21, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_24: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_23, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_32: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_24, 2.0);  convolution_24 = None
        add_32: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_22, mul_32);  convolution_22 = mul_32 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_33: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_26, add_32);  add_26 = add_32 = None
        div_4: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_33, 1.0);  add_33 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_9: "f32[4, 512, 32, 32]" = torch.ops.aten.clone.default(div_4, memory_format = torch.contiguous_format)
        view_54: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(clone_9, [4, 32, 16, 1024])
        var_mean_7 = torch.ops.aten.var_mean.correction(view_54, [2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[4, 32, 1, 1]" = var_mean_7[0]
        getitem_19: "f32[4, 32, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_34: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
        rsqrt_7: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_7: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_54, getitem_19);  view_54 = None
        mul_33: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        view_55: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_33, [4, 512, 32, 32]);  mul_33 = None
        unsqueeze_40: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_24, 0)
        unsqueeze_41: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 2);  unsqueeze_40 = None
        unsqueeze_42: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, 3);  unsqueeze_41 = None
        unsqueeze_43: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_23, 0)
        unsqueeze_44: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 2);  unsqueeze_43 = None
        unsqueeze_45: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, 3);  unsqueeze_44 = None
        mul_34: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_55, unsqueeze_45);  view_55 = unsqueeze_45 = None
        add_35: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_42);  mul_34 = unsqueeze_42 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_6: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_35)
        mul_35: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_35, sigmoid_6);  add_35 = sigmoid_6 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_25: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_35, primals_25, primals_26, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_26: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_35, primals_27, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_27: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_26, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_36: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_27, 2.0);  convolution_27 = None
        add_36: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_25, mul_36);  convolution_25 = mul_36 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_56: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(add_36, [4, 32, 16, 1024])
        var_mean_8 = torch.ops.aten.var_mean.correction(view_56, [2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[4, 32, 1, 1]" = var_mean_8[0]
        getitem_21: "f32[4, 32, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_37: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
        rsqrt_8: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_8: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_56, getitem_21);  view_56 = None
        mul_37: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        view_57: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_37, [4, 512, 32, 32]);  mul_37 = None
        unsqueeze_46: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_30, 0)
        unsqueeze_47: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None
        unsqueeze_48: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_47, 3);  unsqueeze_47 = None
        unsqueeze_49: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_29, 0)
        unsqueeze_50: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 2);  unsqueeze_49 = None
        unsqueeze_51: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 3);  unsqueeze_50 = None
        mul_38: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_57, unsqueeze_51);  view_57 = unsqueeze_51 = None
        add_38: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_48);  mul_38 = unsqueeze_48 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_7: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_38)
        mul_39: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_38, sigmoid_7);  add_38 = sigmoid_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_28: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_39, primals_31, primals_32, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_32 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_29: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_39, primals_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_30: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_29, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_40: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_30, 2.0);  convolution_30 = None
        add_39: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_28, mul_40);  convolution_28 = mul_40 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_40: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(div_4, add_39);  div_4 = add_39 = None
        div_5: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_40, 1.0);  add_40 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_11: "f32[4, 512, 32, 32]" = torch.ops.aten.clone.default(div_5, memory_format = torch.contiguous_format)
        view_58: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(clone_11, [4, 32, 16, 1024])
        var_mean_9 = torch.ops.aten.var_mean.correction(view_58, [2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[4, 32, 1, 1]" = var_mean_9[0]
        getitem_23: "f32[4, 32, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_41: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_9: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_9: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_58, getitem_23);  view_58 = None
        mul_41: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        view_59: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_41, [4, 512, 32, 32]);  mul_41 = None
        unsqueeze_52: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_36, 0)
        unsqueeze_53: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 2);  unsqueeze_52 = None
        unsqueeze_54: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 3);  unsqueeze_53 = None
        unsqueeze_55: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_35, 0)
        unsqueeze_56: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 2);  unsqueeze_55 = None
        unsqueeze_57: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, 3);  unsqueeze_56 = None
        mul_42: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_59, unsqueeze_57);  view_59 = unsqueeze_57 = None
        add_42: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_54);  mul_42 = unsqueeze_54 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_8: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_42)
        mul_43: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_42, sigmoid_8);  add_42 = sigmoid_8 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_31: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_43, primals_37, primals_38, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_38 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_32: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_43, primals_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_33: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_32, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_44: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_33, 2.0);  convolution_33 = None
        add_43: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_31, mul_44);  convolution_31 = mul_44 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_60: "f32[4, 32, 16, 1024]" = torch.ops.aten.reshape.default(add_43, [4, 32, 16, 1024])
        var_mean_10 = torch.ops.aten.var_mean.correction(view_60, [2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[4, 32, 1, 1]" = var_mean_10[0]
        getitem_25: "f32[4, 32, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_44: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_10: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_10: "f32[4, 32, 16, 1024]" = torch.ops.aten.sub.Tensor(view_60, getitem_25);  view_60 = None
        mul_45: "f32[4, 32, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        view_61: "f32[4, 512, 32, 32]" = torch.ops.aten.reshape.default(mul_45, [4, 512, 32, 32]);  mul_45 = None
        unsqueeze_58: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_42, 0)
        unsqueeze_59: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 2);  unsqueeze_58 = None
        unsqueeze_60: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 3);  unsqueeze_59 = None
        unsqueeze_61: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_41, 0)
        unsqueeze_62: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None
        unsqueeze_63: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, 3);  unsqueeze_62 = None
        mul_46: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(view_61, unsqueeze_63);  view_61 = unsqueeze_63 = None
        add_45: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_46, unsqueeze_60);  mul_46 = unsqueeze_60 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_9: "f32[4, 512, 32, 32]" = torch.ops.aten.sigmoid.default(add_45)
        mul_47: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(add_45, sigmoid_9);  add_45 = sigmoid_9 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_34: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(mul_47, primals_43, primals_44, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_44 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_35: "f32[4, 4, 32, 32]" = torch.ops.aten.convolution.default(mul_47, primals_45, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_36: "f32[4, 512, 32, 32]" = torch.ops.aten.convolution.default(convolution_35, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_48: "f32[4, 512, 32, 32]" = torch.ops.aten.mul.Tensor(convolution_36, 2.0);  convolution_36 = None
        add_46: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(convolution_34, mul_48);  convolution_34 = mul_48 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_47: "f32[4, 512, 32, 32]" = torch.ops.aten.add.Tensor(div_5, add_46);  div_5 = add_46 = None
        div_6: "f32[4, 512, 32, 32]" = torch.ops.aten.div.Tensor(add_47, 1.0);  add_47 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/upsampling.py:177 in forward, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        iota: "i64[64]" = torch.ops.prims.iota.default(64, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_49: "i64[64]" = torch.ops.aten.mul.Tensor(iota, 1);  iota = None
        add_48: "i64[64]" = torch.ops.aten.add.Tensor(mul_49, 0);  mul_49 = None
        convert_element_type: "f32[64]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        add_49: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type, 0.0);  convert_element_type = None
        mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        convert_element_type_1: "i64[64]" = torch.ops.prims.convert_element_type.default(mul_50, torch.int64);  mul_50 = None
        unsqueeze_64: "i64[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1)
        _unsafe_index: "f32[4, 512, 64, 64]" = torch.ops.aten._unsafe_index.Tensor(div_6, [None, None, unsqueeze_64, convert_element_type_1]);  div_6 = unsqueeze_64 = None
        clone_13: "f32[4, 512, 64, 64]" = torch.ops.aten.clone.default(_unsafe_index, memory_format = torch.channels_last);  _unsafe_index = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_37: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(clone_13, primals_47, primals_48, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_48 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_38: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(clone_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_39: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_38, primals_50, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_53: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_39, 2.0);  convolution_39 = None
        add_52: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_37, mul_53);  convolution_37 = mul_53 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:40 in my_vae_decoder_fwd, code: skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
        mul_54: "f32[4, 256, 64, 64]" = torch.ops.aten.mul.Tensor(primals_221, 1);  primals_221 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_40: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_54, primals_222, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_41: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_54, primals_223, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_42: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_41, primals_224, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_55: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_42, 2.0);  convolution_42 = None
        add_53: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_40, mul_55);  convolution_40 = mul_55 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:42 in my_vae_decoder_fwd, code: sample = sample + skip_in
        add_54: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(add_52, add_53);  add_52 = add_53 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_14: "f32[4, 512, 64, 64]" = torch.ops.aten.clone.default(add_54, memory_format = torch.contiguous_format)
        view_62: "f32[4, 32, 16, 4096]" = torch.ops.aten.reshape.default(clone_14, [4, 32, 16, 4096])
        var_mean_11 = torch.ops.aten.var_mean.correction(view_62, [2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[4, 32, 1, 1]" = var_mean_11[0]
        getitem_27: "f32[4, 32, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_55: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
        rsqrt_11: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_11: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_62, getitem_27);  view_62 = None
        mul_56: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        view_63: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(mul_56, [4, 512, 64, 64]);  mul_56 = None
        unsqueeze_65: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_52, 0)
        unsqueeze_66: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 2);  unsqueeze_65 = None
        unsqueeze_67: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, 3);  unsqueeze_66 = None
        unsqueeze_68: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_51, 0)
        unsqueeze_69: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, 2);  unsqueeze_68 = None
        unsqueeze_70: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 3);  unsqueeze_69 = None
        mul_57: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_63, unsqueeze_70);  view_63 = unsqueeze_70 = None
        add_56: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_67);  mul_57 = unsqueeze_67 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_10: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_56)
        mul_58: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_10);  add_56 = sigmoid_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_43: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_58, primals_53, primals_54, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_54 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_44: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_58, primals_55, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_45: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_44, primals_56, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_59: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_45, 2.0);  convolution_45 = None
        add_57: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_43, mul_59);  convolution_43 = mul_59 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_64: "f32[4, 32, 16, 4096]" = torch.ops.aten.reshape.default(add_57, [4, 32, 16, 4096])
        var_mean_12 = torch.ops.aten.var_mean.correction(view_64, [2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[4, 32, 1, 1]" = var_mean_12[0]
        getitem_29: "f32[4, 32, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_58: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
        rsqrt_12: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_12: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_64, getitem_29);  view_64 = None
        mul_60: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        view_65: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(mul_60, [4, 512, 64, 64]);  mul_60 = None
        unsqueeze_71: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_58, 0)
        unsqueeze_72: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
        unsqueeze_73: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
        unsqueeze_74: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_57, 0)
        unsqueeze_75: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, 2);  unsqueeze_74 = None
        unsqueeze_76: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 3);  unsqueeze_75 = None
        mul_61: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_65, unsqueeze_76);  view_65 = unsqueeze_76 = None
        add_59: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_61, unsqueeze_73);  mul_61 = unsqueeze_73 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_11: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_59)
        mul_62: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_59, sigmoid_11);  add_59 = sigmoid_11 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_46: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_62, primals_59, primals_60, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_60 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_47: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_62, primals_61, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_48: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_47, primals_62, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_63: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_48, 2.0);  convolution_48 = None
        add_60: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_46, mul_63);  convolution_46 = mul_63 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_61: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(add_54, add_60);  add_54 = add_60 = None
        div_7: "f32[4, 512, 64, 64]" = torch.ops.aten.div.Tensor(add_61, 1.0);  add_61 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_16: "f32[4, 512, 64, 64]" = torch.ops.aten.clone.default(div_7, memory_format = torch.contiguous_format)
        view_66: "f32[4, 32, 16, 4096]" = torch.ops.aten.reshape.default(clone_16, [4, 32, 16, 4096])
        var_mean_13 = torch.ops.aten.var_mean.correction(view_66, [2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[4, 32, 1, 1]" = var_mean_13[0]
        getitem_31: "f32[4, 32, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_62: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_13: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_13: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_66, getitem_31);  view_66 = None
        mul_64: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        view_67: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(mul_64, [4, 512, 64, 64]);  mul_64 = None
        unsqueeze_77: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_64, 0)
        unsqueeze_78: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None
        unsqueeze_79: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 3);  unsqueeze_78 = None
        unsqueeze_80: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_63, 0)
        unsqueeze_81: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 2);  unsqueeze_80 = None
        unsqueeze_82: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 3);  unsqueeze_81 = None
        mul_65: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_67, unsqueeze_82);  view_67 = unsqueeze_82 = None
        add_63: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_79);  mul_65 = unsqueeze_79 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_12: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_63)
        mul_66: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_63, sigmoid_12);  add_63 = sigmoid_12 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_49: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_66, primals_65, primals_66, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_66 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_50: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_66, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_51: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_50, primals_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_67: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_51, 2.0);  convolution_51 = None
        add_64: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_49, mul_67);  convolution_49 = mul_67 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_68: "f32[4, 32, 16, 4096]" = torch.ops.aten.reshape.default(add_64, [4, 32, 16, 4096])
        var_mean_14 = torch.ops.aten.var_mean.correction(view_68, [2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[4, 32, 1, 1]" = var_mean_14[0]
        getitem_33: "f32[4, 32, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_65: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_14: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_14: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_68, getitem_33);  view_68 = None
        mul_68: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        view_69: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(mul_68, [4, 512, 64, 64]);  mul_68 = None
        unsqueeze_83: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_70, 0)
        unsqueeze_84: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
        unsqueeze_85: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 3);  unsqueeze_84 = None
        unsqueeze_86: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_69, 0)
        unsqueeze_87: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, 2);  unsqueeze_86 = None
        unsqueeze_88: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 3);  unsqueeze_87 = None
        mul_69: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_69, unsqueeze_88);  view_69 = unsqueeze_88 = None
        add_66: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_85);  mul_69 = unsqueeze_85 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_13: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_66)
        mul_70: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_66, sigmoid_13);  add_66 = sigmoid_13 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_52: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_70, primals_71, primals_72, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_72 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_53: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_70, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_54: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_53, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_71: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_54, 2.0);  convolution_54 = None
        add_67: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_52, mul_71);  convolution_52 = mul_71 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_68: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(div_7, add_67);  div_7 = add_67 = None
        div_8: "f32[4, 512, 64, 64]" = torch.ops.aten.div.Tensor(add_68, 1.0);  add_68 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_18: "f32[4, 512, 64, 64]" = torch.ops.aten.clone.default(div_8, memory_format = torch.contiguous_format)
        view_70: "f32[4, 32, 16, 4096]" = torch.ops.aten.reshape.default(clone_18, [4, 32, 16, 4096])
        var_mean_15 = torch.ops.aten.var_mean.correction(view_70, [2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[4, 32, 1, 1]" = var_mean_15[0]
        getitem_35: "f32[4, 32, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_69: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
        rsqrt_15: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_15: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_70, getitem_35);  view_70 = None
        mul_72: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        view_71: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(mul_72, [4, 512, 64, 64]);  mul_72 = None
        unsqueeze_89: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_76, 0)
        unsqueeze_90: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        unsqueeze_91: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 3);  unsqueeze_90 = None
        unsqueeze_92: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_75, 0)
        unsqueeze_93: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
        unsqueeze_94: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None
        mul_73: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_71, unsqueeze_94);  view_71 = unsqueeze_94 = None
        add_70: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_73, unsqueeze_91);  mul_73 = unsqueeze_91 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_14: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_70)
        mul_74: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_70, sigmoid_14);  add_70 = sigmoid_14 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_55: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_74, primals_77, primals_78, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_78 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_56: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_74, primals_79, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_57: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_56, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_75: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_57, 2.0);  convolution_57 = None
        add_71: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_55, mul_75);  convolution_55 = mul_75 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_72: "f32[4, 32, 16, 4096]" = torch.ops.aten.reshape.default(add_71, [4, 32, 16, 4096])
        var_mean_16 = torch.ops.aten.var_mean.correction(view_72, [2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[4, 32, 1, 1]" = var_mean_16[0]
        getitem_37: "f32[4, 32, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_72: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_16: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_16: "f32[4, 32, 16, 4096]" = torch.ops.aten.sub.Tensor(view_72, getitem_37);  view_72 = None
        mul_76: "f32[4, 32, 16, 4096]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        view_73: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(mul_76, [4, 512, 64, 64]);  mul_76 = None
        unsqueeze_95: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_82, 0)
        unsqueeze_96: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        unsqueeze_97: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
        unsqueeze_98: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_81, 0)
        unsqueeze_99: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
        unsqueeze_100: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
        mul_77: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(view_73, unsqueeze_100);  view_73 = unsqueeze_100 = None
        add_73: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_97);  mul_77 = unsqueeze_97 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_15: "f32[4, 512, 64, 64]" = torch.ops.aten.sigmoid.default(add_73)
        mul_78: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(add_73, sigmoid_15);  add_73 = sigmoid_15 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_58: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(mul_78, primals_83, primals_84, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_84 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_59: "f32[4, 4, 64, 64]" = torch.ops.aten.convolution.default(mul_78, primals_85, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_60: "f32[4, 512, 64, 64]" = torch.ops.aten.convolution.default(convolution_59, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_79: "f32[4, 512, 64, 64]" = torch.ops.aten.mul.Tensor(convolution_60, 2.0);  convolution_60 = None
        add_74: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(convolution_58, mul_79);  convolution_58 = mul_79 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_75: "f32[4, 512, 64, 64]" = torch.ops.aten.add.Tensor(div_8, add_74);  div_8 = add_74 = None
        div_9: "f32[4, 512, 64, 64]" = torch.ops.aten.div.Tensor(add_75, 1.0);  add_75 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/upsampling.py:177 in forward, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_80: "i64[128]" = torch.ops.aten.mul.Tensor(iota_2, 1);  iota_2 = None
        add_76: "i64[128]" = torch.ops.aten.add.Tensor(mul_80, 0);  mul_80 = None
        convert_element_type_4: "f32[128]" = torch.ops.prims.convert_element_type.default(add_76, torch.float32);  add_76 = None
        add_77: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_4, 0.0);  convert_element_type_4 = None
        mul_81: "f32[128]" = torch.ops.aten.mul.Tensor(add_77, 0.5);  add_77 = None
        convert_element_type_5: "i64[128]" = torch.ops.prims.convert_element_type.default(mul_81, torch.int64);  mul_81 = None
        unsqueeze_101: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, -1)
        _unsafe_index_1: "f32[4, 512, 128, 128]" = torch.ops.aten._unsafe_index.Tensor(div_9, [None, None, unsqueeze_101, convert_element_type_5]);  div_9 = unsqueeze_101 = None
        clone_20: "f32[4, 512, 128, 128]" = torch.ops.aten.clone.default(_unsafe_index_1, memory_format = torch.channels_last);  _unsafe_index_1 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_61: "f32[4, 512, 128, 128]" = torch.ops.aten.convolution.default(clone_20, primals_87, primals_88, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_88 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_62: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(clone_20, primals_89, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_63: "f32[4, 512, 128, 128]" = torch.ops.aten.convolution.default(convolution_62, primals_90, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_84: "f32[4, 512, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_63, 2.0);  convolution_63 = None
        add_80: "f32[4, 512, 128, 128]" = torch.ops.aten.add.Tensor(convolution_61, mul_84);  convolution_61 = mul_84 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:40 in my_vae_decoder_fwd, code: skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
        mul_85: "f32[4, 128, 128, 128]" = torch.ops.aten.mul.Tensor(primals_225, 1);  primals_225 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_64: "f32[4, 512, 128, 128]" = torch.ops.aten.convolution.default(mul_85, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_65: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_85, primals_227, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_66: "f32[4, 512, 128, 128]" = torch.ops.aten.convolution.default(convolution_65, primals_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_86: "f32[4, 512, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_66, 2.0);  convolution_66 = None
        add_81: "f32[4, 512, 128, 128]" = torch.ops.aten.add.Tensor(convolution_64, mul_86);  convolution_64 = mul_86 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:42 in my_vae_decoder_fwd, code: sample = sample + skip_in
        add_82: "f32[4, 512, 128, 128]" = torch.ops.aten.add.Tensor(add_80, add_81);  add_80 = add_81 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_21: "f32[4, 512, 128, 128]" = torch.ops.aten.clone.default(add_82, memory_format = torch.contiguous_format)
        view_74: "f32[4, 32, 16, 16384]" = torch.ops.aten.reshape.default(clone_21, [4, 32, 16, 16384]);  clone_21 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(view_74, [2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[4, 32, 1, 1]" = var_mean_17[0]
        getitem_39: "f32[4, 32, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_83: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
        rsqrt_17: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_17: "f32[4, 32, 16, 16384]" = torch.ops.aten.sub.Tensor(view_74, getitem_39);  view_74 = None
        mul_87: "f32[4, 32, 16, 16384]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        view_75: "f32[4, 512, 128, 128]" = torch.ops.aten.reshape.default(mul_87, [4, 512, 128, 128]);  mul_87 = None
        unsqueeze_102: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_92, 0)
        unsqueeze_103: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
        unsqueeze_104: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
        unsqueeze_105: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_91, 0)
        unsqueeze_106: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_107: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
        mul_88: "f32[4, 512, 128, 128]" = torch.ops.aten.mul.Tensor(view_75, unsqueeze_107);  view_75 = unsqueeze_107 = None
        add_84: "f32[4, 512, 128, 128]" = torch.ops.aten.add.Tensor(mul_88, unsqueeze_104);  mul_88 = unsqueeze_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_16: "f32[4, 512, 128, 128]" = torch.ops.aten.sigmoid.default(add_84)
        mul_89: "f32[4, 512, 128, 128]" = torch.ops.aten.mul.Tensor(add_84, sigmoid_16);  add_84 = sigmoid_16 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_67: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_89, primals_93, primals_94, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_94 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_68: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_89, primals_95, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_69: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_68, primals_96, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_90: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_69, 2.0);  convolution_69 = None
        add_85: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_67, mul_90);  convolution_67 = mul_90 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_76: "f32[4, 32, 8, 16384]" = torch.ops.aten.reshape.default(add_85, [4, 32, 8, 16384])
        var_mean_18 = torch.ops.aten.var_mean.correction(view_76, [2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[4, 32, 1, 1]" = var_mean_18[0]
        getitem_41: "f32[4, 32, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_86: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
        rsqrt_18: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_18: "f32[4, 32, 8, 16384]" = torch.ops.aten.sub.Tensor(view_76, getitem_41);  view_76 = None
        mul_91: "f32[4, 32, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        view_77: "f32[4, 256, 128, 128]" = torch.ops.aten.reshape.default(mul_91, [4, 256, 128, 128]);  mul_91 = None
        unsqueeze_108: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_98, 0)
        unsqueeze_109: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        unsqueeze_110: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
        unsqueeze_111: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_97, 0)
        unsqueeze_112: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        unsqueeze_113: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
        mul_92: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(view_77, unsqueeze_113);  view_77 = unsqueeze_113 = None
        add_87: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_110);  mul_92 = unsqueeze_110 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_17: "f32[4, 256, 128, 128]" = torch.ops.aten.sigmoid.default(add_87)
        mul_93: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(add_87, sigmoid_17);  add_87 = sigmoid_17 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_70: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_93, primals_99, primals_100, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_100 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_71: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_93, primals_101, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_72: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_71, primals_102, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_94: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_72, 2.0);  convolution_72 = None
        add_88: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_70, mul_94);  convolution_70 = mul_94 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_73: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(add_82, primals_103, primals_104, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_74: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(add_82, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_75: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_74, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_95: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_75, 2.0);  convolution_75 = None
        add_89: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_73, mul_95);  convolution_73 = mul_95 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_90: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(add_89, add_88);  add_89 = add_88 = None
        div_10: "f32[4, 256, 128, 128]" = torch.ops.aten.div.Tensor(add_90, 1.0);  add_90 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_23: "f32[4, 256, 128, 128]" = torch.ops.aten.clone.default(div_10, memory_format = torch.contiguous_format)
        view_78: "f32[4, 32, 8, 16384]" = torch.ops.aten.reshape.default(clone_23, [4, 32, 8, 16384])
        var_mean_19 = torch.ops.aten.var_mean.correction(view_78, [2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[4, 32, 1, 1]" = var_mean_19[0]
        getitem_43: "f32[4, 32, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_91: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
        rsqrt_19: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_19: "f32[4, 32, 8, 16384]" = torch.ops.aten.sub.Tensor(view_78, getitem_43);  view_78 = None
        mul_96: "f32[4, 32, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        view_79: "f32[4, 256, 128, 128]" = torch.ops.aten.reshape.default(mul_96, [4, 256, 128, 128]);  mul_96 = None
        unsqueeze_114: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_108, 0)
        unsqueeze_115: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
        unsqueeze_116: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
        unsqueeze_117: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_107, 0)
        unsqueeze_118: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
        unsqueeze_119: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
        mul_97: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(view_79, unsqueeze_119);  view_79 = unsqueeze_119 = None
        add_92: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_116);  mul_97 = unsqueeze_116 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_18: "f32[4, 256, 128, 128]" = torch.ops.aten.sigmoid.default(add_92)
        mul_98: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(add_92, sigmoid_18);  add_92 = sigmoid_18 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_76: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_98, primals_109, primals_110, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_110 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_77: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_98, primals_111, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_78: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_77, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_99: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_78, 2.0);  convolution_78 = None
        add_93: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_76, mul_99);  convolution_76 = mul_99 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_80: "f32[4, 32, 8, 16384]" = torch.ops.aten.reshape.default(add_93, [4, 32, 8, 16384])
        var_mean_20 = torch.ops.aten.var_mean.correction(view_80, [2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[4, 32, 1, 1]" = var_mean_20[0]
        getitem_45: "f32[4, 32, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_94: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
        rsqrt_20: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_20: "f32[4, 32, 8, 16384]" = torch.ops.aten.sub.Tensor(view_80, getitem_45);  view_80 = None
        mul_100: "f32[4, 32, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        view_81: "f32[4, 256, 128, 128]" = torch.ops.aten.reshape.default(mul_100, [4, 256, 128, 128]);  mul_100 = None
        unsqueeze_120: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_114, 0)
        unsqueeze_121: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 2);  unsqueeze_120 = None
        unsqueeze_122: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
        unsqueeze_123: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_113, 0)
        unsqueeze_124: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
        unsqueeze_125: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 3);  unsqueeze_124 = None
        mul_101: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(view_81, unsqueeze_125);  view_81 = unsqueeze_125 = None
        add_95: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_122);  mul_101 = unsqueeze_122 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_19: "f32[4, 256, 128, 128]" = torch.ops.aten.sigmoid.default(add_95)
        mul_102: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(add_95, sigmoid_19);  add_95 = sigmoid_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_79: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_102, primals_115, primals_116, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_116 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_80: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_102, primals_117, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_81: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_80, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_103: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_81, 2.0);  convolution_81 = None
        add_96: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_79, mul_103);  convolution_79 = mul_103 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_97: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(div_10, add_96);  div_10 = add_96 = None
        div_11: "f32[4, 256, 128, 128]" = torch.ops.aten.div.Tensor(add_97, 1.0);  add_97 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_25: "f32[4, 256, 128, 128]" = torch.ops.aten.clone.default(div_11, memory_format = torch.contiguous_format)
        view_82: "f32[4, 32, 8, 16384]" = torch.ops.aten.reshape.default(clone_25, [4, 32, 8, 16384])
        var_mean_21 = torch.ops.aten.var_mean.correction(view_82, [2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[4, 32, 1, 1]" = var_mean_21[0]
        getitem_47: "f32[4, 32, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_98: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_21: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_21: "f32[4, 32, 8, 16384]" = torch.ops.aten.sub.Tensor(view_82, getitem_47);  view_82 = None
        mul_104: "f32[4, 32, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        view_83: "f32[4, 256, 128, 128]" = torch.ops.aten.reshape.default(mul_104, [4, 256, 128, 128]);  mul_104 = None
        unsqueeze_126: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_120, 0)
        unsqueeze_127: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 2);  unsqueeze_126 = None
        unsqueeze_128: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 3);  unsqueeze_127 = None
        unsqueeze_129: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_119, 0)
        unsqueeze_130: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
        unsqueeze_131: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
        mul_105: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(view_83, unsqueeze_131);  view_83 = unsqueeze_131 = None
        add_99: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(mul_105, unsqueeze_128);  mul_105 = unsqueeze_128 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_20: "f32[4, 256, 128, 128]" = torch.ops.aten.sigmoid.default(add_99)
        mul_106: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(add_99, sigmoid_20);  add_99 = sigmoid_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_82: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_106, primals_121, primals_122, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_122 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_83: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_106, primals_123, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_84: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_83, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_107: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_84, 2.0);  convolution_84 = None
        add_100: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_82, mul_107);  convolution_82 = mul_107 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_84: "f32[4, 32, 8, 16384]" = torch.ops.aten.reshape.default(add_100, [4, 32, 8, 16384])
        var_mean_22 = torch.ops.aten.var_mean.correction(view_84, [2, 3], correction = 0, keepdim = True)
        getitem_48: "f32[4, 32, 1, 1]" = var_mean_22[0]
        getitem_49: "f32[4, 32, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_101: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_22: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_22: "f32[4, 32, 8, 16384]" = torch.ops.aten.sub.Tensor(view_84, getitem_49);  view_84 = None
        mul_108: "f32[4, 32, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        view_85: "f32[4, 256, 128, 128]" = torch.ops.aten.reshape.default(mul_108, [4, 256, 128, 128]);  mul_108 = None
        unsqueeze_132: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_126, 0)
        unsqueeze_133: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
        unsqueeze_134: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 3);  unsqueeze_133 = None
        unsqueeze_135: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_125, 0)
        unsqueeze_136: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
        unsqueeze_137: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 3);  unsqueeze_136 = None
        mul_109: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(view_85, unsqueeze_137);  view_85 = unsqueeze_137 = None
        add_102: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(mul_109, unsqueeze_134);  mul_109 = unsqueeze_134 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_21: "f32[4, 256, 128, 128]" = torch.ops.aten.sigmoid.default(add_102)
        mul_110: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(add_102, sigmoid_21);  add_102 = sigmoid_21 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_85: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(mul_110, primals_127, primals_128, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_128 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_86: "f32[4, 4, 128, 128]" = torch.ops.aten.convolution.default(mul_110, primals_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_87: "f32[4, 256, 128, 128]" = torch.ops.aten.convolution.default(convolution_86, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_111: "f32[4, 256, 128, 128]" = torch.ops.aten.mul.Tensor(convolution_87, 2.0);  convolution_87 = None
        add_103: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(convolution_85, mul_111);  convolution_85 = mul_111 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_104: "f32[4, 256, 128, 128]" = torch.ops.aten.add.Tensor(div_11, add_103);  div_11 = add_103 = None
        div_12: "f32[4, 256, 128, 128]" = torch.ops.aten.div.Tensor(add_104, 1.0);  add_104 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/upsampling.py:177 in forward, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        iota_4: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        mul_112: "i64[256]" = torch.ops.aten.mul.Tensor(iota_4, 1);  iota_4 = None
        add_105: "i64[256]" = torch.ops.aten.add.Tensor(mul_112, 0);  mul_112 = None
        convert_element_type_8: "f32[256]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        add_106: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_8, 0.0);  convert_element_type_8 = None
        mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
        convert_element_type_9: "i64[256]" = torch.ops.prims.convert_element_type.default(mul_113, torch.int64);  mul_113 = None
        unsqueeze_138: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_9, -1)
        _unsafe_index_2: "f32[4, 256, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(div_12, [None, None, unsqueeze_138, convert_element_type_9]);  div_12 = unsqueeze_138 = None
        clone_27: "f32[4, 256, 256, 256]" = torch.ops.aten.clone.default(_unsafe_index_2, memory_format = torch.channels_last);  _unsafe_index_2 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_88: "f32[4, 256, 256, 256]" = torch.ops.aten.convolution.default(clone_27, primals_131, primals_132, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_132 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_89: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(clone_27, primals_133, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_90: "f32[4, 256, 256, 256]" = torch.ops.aten.convolution.default(convolution_89, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_116: "f32[4, 256, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_90, 2.0);  convolution_90 = None
        add_109: "f32[4, 256, 256, 256]" = torch.ops.aten.add.Tensor(convolution_88, mul_116);  convolution_88 = mul_116 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:40 in my_vae_decoder_fwd, code: skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
        mul_117: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(primals_229, 1);  primals_229 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_91: "f32[4, 256, 256, 256]" = torch.ops.aten.convolution.default(mul_117, primals_230, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_92: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_117, primals_231, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_93: "f32[4, 256, 256, 256]" = torch.ops.aten.convolution.default(convolution_92, primals_232, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_118: "f32[4, 256, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_93, 2.0);  convolution_93 = None
        add_110: "f32[4, 256, 256, 256]" = torch.ops.aten.add.Tensor(convolution_91, mul_118);  convolution_91 = mul_118 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:42 in my_vae_decoder_fwd, code: sample = sample + skip_in
        add_111: "f32[4, 256, 256, 256]" = torch.ops.aten.add.Tensor(add_109, add_110);  add_109 = add_110 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_28: "f32[4, 256, 256, 256]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
        view_86: "f32[4, 32, 8, 65536]" = torch.ops.aten.reshape.default(clone_28, [4, 32, 8, 65536]);  clone_28 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(view_86, [2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[4, 32, 1, 1]" = var_mean_23[0]
        getitem_51: "f32[4, 32, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_112: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
        rsqrt_23: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_23: "f32[4, 32, 8, 65536]" = torch.ops.aten.sub.Tensor(view_86, getitem_51);  view_86 = None
        mul_119: "f32[4, 32, 8, 65536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        view_87: "f32[4, 256, 256, 256]" = torch.ops.aten.reshape.default(mul_119, [4, 256, 256, 256]);  mul_119 = None
        unsqueeze_139: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_136, 0)
        unsqueeze_140: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        unsqueeze_142: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_135, 0)
        unsqueeze_143: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        mul_120: "f32[4, 256, 256, 256]" = torch.ops.aten.mul.Tensor(view_87, unsqueeze_144);  view_87 = unsqueeze_144 = None
        add_113: "f32[4, 256, 256, 256]" = torch.ops.aten.add.Tensor(mul_120, unsqueeze_141);  mul_120 = unsqueeze_141 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_22: "f32[4, 256, 256, 256]" = torch.ops.aten.sigmoid.default(add_113)
        mul_121: "f32[4, 256, 256, 256]" = torch.ops.aten.mul.Tensor(add_113, sigmoid_22);  add_113 = sigmoid_22 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_94: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_121, primals_137, primals_138, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_138 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_95: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_121, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_96: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_95, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_122: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_96, 2.0);  convolution_96 = None
        add_114: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_94, mul_122);  convolution_94 = mul_122 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_88: "f32[4, 32, 4, 65536]" = torch.ops.aten.reshape.default(add_114, [4, 32, 4, 65536])
        var_mean_24 = torch.ops.aten.var_mean.correction(view_88, [2, 3], correction = 0, keepdim = True)
        getitem_52: "f32[4, 32, 1, 1]" = var_mean_24[0]
        getitem_53: "f32[4, 32, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_115: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
        rsqrt_24: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_24: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_88, getitem_53);  view_88 = None
        mul_123: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        view_89: "f32[4, 128, 256, 256]" = torch.ops.aten.reshape.default(mul_123, [4, 128, 256, 256]);  mul_123 = None
        unsqueeze_145: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_142, 0)
        unsqueeze_146: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        unsqueeze_148: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_141, 0)
        unsqueeze_149: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 2);  unsqueeze_148 = None
        unsqueeze_150: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 3);  unsqueeze_149 = None
        mul_124: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_89, unsqueeze_150);  view_89 = unsqueeze_150 = None
        add_116: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_124, unsqueeze_147);  mul_124 = unsqueeze_147 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_23: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_116)
        mul_125: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_116, sigmoid_23);  add_116 = sigmoid_23 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_97: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_125, primals_143, primals_144, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_144 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_98: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_125, primals_145, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_99: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_98, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_126: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_99, 2.0);  convolution_99 = None
        add_117: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_97, mul_126);  convolution_97 = mul_126 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_100: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(add_111, primals_147, primals_148, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_148 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_101: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(add_111, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        convolution_102: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_101, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_127: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_102, 2.0);  convolution_102 = None
        add_118: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_100, mul_127);  convolution_100 = mul_127 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_119: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(add_118, add_117);  add_118 = add_117 = None
        div_13: "f32[4, 128, 256, 256]" = torch.ops.aten.div.Tensor(add_119, 1.0);  add_119 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_30: "f32[4, 128, 256, 256]" = torch.ops.aten.clone.default(div_13, memory_format = torch.contiguous_format)
        view_90: "f32[4, 32, 4, 65536]" = torch.ops.aten.reshape.default(clone_30, [4, 32, 4, 65536])
        var_mean_25 = torch.ops.aten.var_mean.correction(view_90, [2, 3], correction = 0, keepdim = True)
        getitem_54: "f32[4, 32, 1, 1]" = var_mean_25[0]
        getitem_55: "f32[4, 32, 1, 1]" = var_mean_25[1];  var_mean_25 = None
        add_120: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_25: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_25: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_90, getitem_55);  view_90 = None
        mul_128: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        view_91: "f32[4, 128, 256, 256]" = torch.ops.aten.reshape.default(mul_128, [4, 128, 256, 256]);  mul_128 = None
        unsqueeze_151: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_152, 0)
        unsqueeze_152: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        unsqueeze_154: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_151, 0)
        unsqueeze_155: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
        mul_129: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_91, unsqueeze_156);  view_91 = unsqueeze_156 = None
        add_121: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_153);  mul_129 = unsqueeze_153 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_24: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_121)
        mul_130: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_121, sigmoid_24);  add_121 = sigmoid_24 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_103: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_130, primals_153, primals_154, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_154 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_104: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_130, primals_155, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_105: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_104, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_131: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_105, 2.0);  convolution_105 = None
        add_122: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_103, mul_131);  convolution_103 = mul_131 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_92: "f32[4, 32, 4, 65536]" = torch.ops.aten.reshape.default(add_122, [4, 32, 4, 65536])
        var_mean_26 = torch.ops.aten.var_mean.correction(view_92, [2, 3], correction = 0, keepdim = True)
        getitem_56: "f32[4, 32, 1, 1]" = var_mean_26[0]
        getitem_57: "f32[4, 32, 1, 1]" = var_mean_26[1];  var_mean_26 = None
        add_123: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_26: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_26: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_92, getitem_57);  view_92 = None
        mul_132: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        view_93: "f32[4, 128, 256, 256]" = torch.ops.aten.reshape.default(mul_132, [4, 128, 256, 256]);  mul_132 = None
        unsqueeze_157: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_158, 0)
        unsqueeze_158: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        unsqueeze_160: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_157, 0)
        unsqueeze_161: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 2);  unsqueeze_160 = None
        unsqueeze_162: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 3);  unsqueeze_161 = None
        mul_133: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_93, unsqueeze_162);  view_93 = unsqueeze_162 = None
        add_124: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_133, unsqueeze_159);  mul_133 = unsqueeze_159 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_25: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_124)
        mul_134: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_124, sigmoid_25);  add_124 = sigmoid_25 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_106: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_134, primals_159, primals_160, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_160 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_107: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_134, primals_161, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_108: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_107, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_135: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_108, 2.0);  convolution_108 = None
        add_125: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_106, mul_135);  convolution_106 = mul_135 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_126: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(div_13, add_125);  div_13 = add_125 = None
        div_14: "f32[4, 128, 256, 256]" = torch.ops.aten.div.Tensor(add_126, 1.0);  add_126 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:327 in forward, code: hidden_states = self.norm1(hidden_states)
        clone_32: "f32[4, 128, 256, 256]" = torch.ops.aten.clone.default(div_14, memory_format = torch.contiguous_format)
        view_94: "f32[4, 32, 4, 65536]" = torch.ops.aten.reshape.default(clone_32, [4, 32, 4, 65536])
        var_mean_27 = torch.ops.aten.var_mean.correction(view_94, [2, 3], correction = 0, keepdim = True)
        getitem_58: "f32[4, 32, 1, 1]" = var_mean_27[0]
        getitem_59: "f32[4, 32, 1, 1]" = var_mean_27[1];  var_mean_27 = None
        add_127: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
        rsqrt_27: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_27: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_94, getitem_59);  view_94 = None
        mul_136: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        view_95: "f32[4, 128, 256, 256]" = torch.ops.aten.reshape.default(mul_136, [4, 128, 256, 256]);  mul_136 = None
        unsqueeze_163: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_164, 0)
        unsqueeze_164: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        unsqueeze_166: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_163, 0)
        unsqueeze_167: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        mul_137: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_95, unsqueeze_168);  view_95 = unsqueeze_168 = None
        add_128: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_165);  mul_137 = unsqueeze_165 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:328 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_26: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_128)
        mul_138: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_128, sigmoid_26);  add_128 = sigmoid_26 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_109: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_138, primals_165, primals_166, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_166 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_110: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_138, primals_167, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_111: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_110, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_139: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_111, 2.0);  convolution_111 = None
        add_129: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_109, mul_139);  convolution_109 = mul_139 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:361 in forward, code: hidden_states = self.norm2(hidden_states)
        view_96: "f32[4, 32, 4, 65536]" = torch.ops.aten.reshape.default(add_129, [4, 32, 4, 65536])
        var_mean_28 = torch.ops.aten.var_mean.correction(view_96, [2, 3], correction = 0, keepdim = True)
        getitem_60: "f32[4, 32, 1, 1]" = var_mean_28[0]
        getitem_61: "f32[4, 32, 1, 1]" = var_mean_28[1];  var_mean_28 = None
        add_130: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
        rsqrt_28: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_28: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_96, getitem_61);  view_96 = None
        mul_140: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        view_97: "f32[4, 128, 256, 256]" = torch.ops.aten.reshape.default(mul_140, [4, 128, 256, 256]);  mul_140 = None
        unsqueeze_169: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_170, 0)
        unsqueeze_170: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        unsqueeze_172: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_169, 0)
        unsqueeze_173: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 2);  unsqueeze_172 = None
        unsqueeze_174: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 3);  unsqueeze_173 = None
        mul_141: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_97, unsqueeze_174);  view_97 = unsqueeze_174 = None
        add_131: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_141, unsqueeze_171);  mul_141 = unsqueeze_171 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:363 in forward, code: hidden_states = self.nonlinearity(hidden_states)
        sigmoid_27: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_131)
        mul_142: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_131, sigmoid_27);  add_131 = sigmoid_27 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_112: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(mul_142, primals_171, primals_172, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_172 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_113: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_142, primals_173, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_114: "f32[4, 128, 256, 256]" = torch.ops.aten.convolution.default(convolution_113, primals_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_143: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_114, 2.0);  convolution_114 = None
        add_132: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(convolution_112, mul_143);  convolution_112 = mul_143 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/diffusers/models/resnet.py:371 in forward, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_133: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(div_14, add_132);  div_14 = add_132 = None
        div_15: "f32[4, 128, 256, 256]" = torch.ops.aten.div.Tensor(add_133, 1.0);  add_133 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:49 in my_vae_decoder_fwd, code: sample = self.conv_norm_out(sample)
        clone_34: "f32[4, 128, 256, 256]" = torch.ops.aten.clone.default(div_15, memory_format = torch.contiguous_format);  div_15 = None
        view_98: "f32[4, 32, 4, 65536]" = torch.ops.aten.reshape.default(clone_34, [4, 32, 4, 65536])
        var_mean_29 = torch.ops.aten.var_mean.correction(view_98, [2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[4, 32, 1, 1]" = var_mean_29[0]
        getitem_63: "f32[4, 32, 1, 1]" = var_mean_29[1];  var_mean_29 = None
        add_134: "f32[4, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_29: "f32[4, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_29: "f32[4, 32, 4, 65536]" = torch.ops.aten.sub.Tensor(view_98, getitem_63);  view_98 = None
        mul_144: "f32[4, 32, 4, 65536]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        view_99: "f32[4, 128, 256, 256]" = torch.ops.aten.reshape.default(mul_144, [4, 128, 256, 256]);  mul_144 = None
        unsqueeze_175: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_234, 0)
        unsqueeze_176: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        unsqueeze_177: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
        unsqueeze_178: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_233, 0)
        unsqueeze_179: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
        unsqueeze_180: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
        mul_145: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(view_99, unsqueeze_180);  view_99 = unsqueeze_180 = None
        add_135: "f32[4, 128, 256, 256]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_177);  mul_145 = unsqueeze_177 = None
        
         # File: /home/elicer/cyclegan-turbo/src/model.py:52 in my_vae_decoder_fwd, code: sample = self.conv_act(sample)
        sigmoid_28: "f32[4, 128, 256, 256]" = torch.ops.aten.sigmoid.default(add_135)
        mul_146: "f32[4, 128, 256, 256]" = torch.ops.aten.mul.Tensor(add_135, sigmoid_28);  add_135 = sigmoid_28 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:654 in forward, code: result = self.base_layer(x, *args, **kwargs)
        convolution_115: "f32[4, 3, 256, 256]" = torch.ops.aten.convolution.default(mul_146, primals_235, primals_236, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_236 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:663 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        convolution_116: "f32[4, 4, 256, 256]" = torch.ops.aten.convolution.default(mul_146, primals_237, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        convolution_117: "f32[4, 3, 256, 256]" = torch.ops.aten.convolution.default(convolution_116, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        mul_147: "f32[4, 3, 256, 256]" = torch.ops.aten.mul.Tensor(convolution_117, 2.0);  convolution_117 = None
        add_136: "f32[4, 3, 256, 256]" = torch.ops.aten.add.Tensor(convolution_115, mul_147);  convolution_115 = mul_147 = None
        
         # File: /home/elicer/cyclegan-turbo/src/cyclegan_turbo.py:44 in forward, code: x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        clamp_min: "f32[4, 3, 256, 256]" = torch.ops.aten.clamp_min.default(add_136, -1)
        clamp_max: "f32[4, 3, 256, 256]" = torch.ops.aten.clamp_max.default(clamp_min, 1);  clamp_min = None
        ge: "b8[4, 3, 256, 256]" = torch.ops.aten.ge.Scalar(add_136, -1)
        le: "b8[4, 3, 256, 256]" = torch.ops.aten.le.Scalar(add_136, 1);  add_136 = None
        logical_and: "b8[4, 3, 256, 256]" = torch.ops.aten.logical_and.default(ge, le);  ge = le = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_53: "f32[512, 4]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_57: "f32[4, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:364 in forward, code: result = self.base_layer(x, *args, **kwargs)
        permute_59: "f32[512, 512]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_66: "f32[512, 4]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_70: "f32[4, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_75: "f32[512, 4]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_79: "f32[4, 512]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        
         # File: /home/elicer/.local/lib/python3.10/site-packages/peft/tuners/lora/layer.py:373 in forward, code: result += lora_B(lora_A(dropout(x))) * scaling
        permute_84: "f32[512, 4]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_88: "f32[4, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        return (clamp_max, primals_5, primals_7, primals_9, primals_10, primals_11, primals_12, primals_13, primals_15, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_61, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, primals_73, primals_74, primals_75, primals_76, primals_77, primals_79, primals_80, primals_81, primals_82, primals_83, primals_85, primals_86, primals_87, primals_89, primals_90, primals_91, primals_92, primals_93, primals_95, primals_96, primals_97, primals_98, primals_99, primals_101, primals_102, primals_103, primals_105, primals_106, primals_107, primals_108, primals_109, primals_111, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_133, primals_134, primals_135, primals_136, primals_137, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_173, primals_174, primals_175, primals_176, primals_177, primals_179, primals_180, primals_181, primals_182, primals_183, primals_185, primals_186, primals_187, primals_189, primals_193, primals_197, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, div, convolution, convolution_2, add, getitem_1, rsqrt, mul_3, convolution_5, add_3, getitem_3, rsqrt_1, mul_7, convolution_8, view_5, squeeze_4, squeeze_5, view_11, mm, mm_2, mm_4, permute_15, permute_16, permute_17, getitem_6, getitem_7, getitem_8, getitem_9, mm_6, clone_5, getitem_11, rsqrt_3, mul_17, convolution_11, add_20, getitem_13, rsqrt_4, mul_21, convolution_14, mul_23, convolution_17, clone_7, getitem_15, rsqrt_5, mul_27, convolution_20, add_29, getitem_17, rsqrt_6, mul_31, convolution_23, clone_9, getitem_19, rsqrt_7, mul_35, convolution_26, add_36, getitem_21, rsqrt_8, mul_39, convolution_29, clone_11, getitem_23, rsqrt_9, mul_43, convolution_32, add_43, getitem_25, rsqrt_10, mul_47, convolution_35, convert_element_type_1, clone_13, convolution_38, mul_54, convolution_41, clone_14, getitem_27, rsqrt_11, mul_58, convolution_44, add_57, getitem_29, rsqrt_12, mul_62, convolution_47, clone_16, getitem_31, rsqrt_13, mul_66, convolution_50, add_64, getitem_33, rsqrt_14, mul_70, convolution_53, clone_18, getitem_35, rsqrt_15, mul_74, convolution_56, add_71, getitem_37, rsqrt_16, mul_78, convolution_59, convert_element_type_5, clone_20, convolution_62, mul_85, convolution_65, add_82, getitem_39, rsqrt_17, mul_89, convolution_68, add_85, getitem_41, rsqrt_18, mul_93, convolution_71, convolution_74, clone_23, getitem_43, rsqrt_19, mul_98, convolution_77, add_93, getitem_45, rsqrt_20, mul_102, convolution_80, clone_25, getitem_47, rsqrt_21, mul_106, convolution_83, add_100, getitem_49, rsqrt_22, mul_110, convolution_86, convert_element_type_9, clone_27, convolution_89, mul_117, convolution_92, add_111, getitem_51, rsqrt_23, mul_121, convolution_95, add_114, getitem_53, rsqrt_24, mul_125, convolution_98, convolution_101, clone_30, getitem_55, rsqrt_25, mul_130, convolution_104, add_122, getitem_57, rsqrt_26, mul_134, convolution_107, clone_32, getitem_59, rsqrt_27, mul_138, convolution_110, add_129, getitem_61, rsqrt_28, mul_142, convolution_113, clone_34, getitem_63, rsqrt_29, mul_146, convolution_116, logical_and, permute_53, permute_57, permute_59, permute_66, permute_70, permute_75, permute_79, permute_84, permute_88)
        