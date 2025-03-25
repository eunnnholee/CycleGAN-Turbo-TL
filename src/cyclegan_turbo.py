import os
import sys
import copy
import torch
import torch.nn as nn
# from transformers import AutoTokenizer, CLIPTextModel  # 제거: 텍스트 관련 코드 제거
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def initialize_vae(rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", 
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


# class CycleGAN_Turbo(torch.nn.Module):
#     def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
#         super().__init__()
#         # 텍스트 관련 초기화 제거:
#         # self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
#         # self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        
#         self.sched = make_1step_sched()
#         vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
#         unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
#         vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
#         vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
#         # add the skip connection convs
#         vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
#         vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
#         vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
#         vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
#         vae.decoder.ignore_skip = False
#         self.unet, self.vae = unet, vae
        
#         # pretrained_name에 따른 체크포인트 로드 부분에서 캡션 관련 코드는 제거합니다.
#         if pretrained_name == "day_to_night":
#             url = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
#             self.load_ckpt_from_url(url, ckpt_folder)
#             self.timesteps = torch.tensor([999], device="cuda").long()
#             self.direction = "a2b"
#         elif pretrained_name == "night_to_day":
#             url = "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl"
#             self.load_ckpt_from_url(url, ckpt_folder)
#             self.timesteps = torch.tensor([999], device="cuda").long()
#             self.direction = "b2a"
#         elif pretrained_name == "clear_to_rainy":
#             url = "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl"
#             self.load_ckpt_from_url(url, ckpt_folder)
#             self.timesteps = torch.tensor([999], device="cuda").long()
#             self.direction = "a2b"
#         elif pretrained_name == "rainy_to_clear":
#             url = "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl"
#             self.load_ckpt_from_url(url, ckpt_folder)
#             self.timesteps = torch.tensor([999], device="cuda").long()
#             self.direction = "b2a"
#         elif pretrained_path is not None:
#             sd = torch.load(pretrained_path)
#             self.load_ckpt_from_state_dict(sd)
#             self.timesteps = torch.tensor([999], device="cuda").long()
#             self.direction = None

#         self.vae_enc.cuda()
#         self.vae_dec.cuda()
#         self.unet.cuda()

## 위 코드에서 변경함. 
class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        
        self.sched = make_1step_sched()

        # Load base SD-Turbo components
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        # Patch VAE forward
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        # Add skip connections
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False

        self.unet, self.vae = unet, vae

        # Load pretrained LoRA checkpoints if specified
        if pretrained_name in ["day_to_night", "night_to_day", "clear_to_rainy", "rainy_to_clear"]:
            url_map = {
                "day_to_night": "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl",
                "night_to_day": "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl",
                "clear_to_rainy": "https://www.cs.cmu.edu/~img2img-turbo/models/clear2rainy.pkl",
                "rainy_to_clear": "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl",
            }
            url = url_map[pretrained_name]
            self.load_ckpt_from_url(url, ckpt_folder)
            self.direction = "a2b" if pretrained_name in ["day_to_night", "clear_to_rainy"] else "b2a"

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.direction = None

        else:
            # No pretrained checkpoint used
            self.direction = None
            # Ensure VAE has required attributes even without pretrained checkpoint
            self.vae.decoder.gamma = 1
            self.vae.decoder.ignore_skip = False
            self.vae_b2a = copy.deepcopy(self.vae)
            self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
            self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)


        # Always set timesteps
        self.timesteps = torch.tensor([999], device="cuda").long()

        # Move models to GPU
        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()
        print(f"[INFO] Loaded model with pretrained_name={pretrained_name}")


    # def load_ckpt_from_state_dict(self, sd):
    #     lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
    #     lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
    #     lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
    #     self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    #     self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    #     self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
    #     for n, p in self.unet.named_parameters():
    #         name_sd = n.replace(".default_encoder.weight", ".weight")
    #         if "lora" in n and "default_encoder" in n:
    #             p.data.copy_(sd["sd_encoder"][name_sd])
    #     for n, p in self.unet.named_parameters():
    #         name_sd = n.replace(".default_decoder.weight", ".weight")
    #         if "lora" in n and "default_decoder" in n:
    #             p.data.copy_(sd["sd_decoder"][name_sd])
    #     for n, p in self.unet.named_parameters():
    #         name_sd = n.replace(".default_others.weight", ".weight")
    #         if "lora" in n and "default_others" in n:
    #             p.data.copy_(sd["sd_other"][name_sd])
    #     self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

    #     vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
    #     self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    #     self.vae.decoder.gamma = 1
    #     self.vae_b2a = copy.deepcopy(self.vae)
    #     self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
    #     self.vae_enc.load_state_dict(sd["sd_vae_enc"])
    #     self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
    #     self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    def load_ckpt_from_state_dict(self, sd):
        # UNet LoRA 어댑터 구성
        lora_conf_encoder = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_encoder"],
            lora_alpha=sd["rank_unet"]
        )
        lora_conf_decoder = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_decoder"],
            lora_alpha=sd["rank_unet"]
        )
        lora_conf_others = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_modules_others"],
            lora_alpha=sd["rank_unet"]
        )
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")

        # 어댑터 태그(예: ".default_encoder")를 제거하고 "_orig_mod." 접두사를 붙여 체크포인트 키와 맞춥니다.
        def copy_adapter_params(named_params, checkpoint, adapter_tag):
            for name, param in named_params:
                if adapter_tag in name and "lora" in name:
                    # 모델에서 저장 시 어댑터 태그가 추가되었으므로 이를 제거합니다.
                    key_name = name.replace(f".{adapter_tag}", "")
                    # 체크포인트의 키는 모두 "_orig_mod."로 시작하므로,
                    # 만약 key_name에 접두사가 없다면 추가합니다.
                    if not key_name.startswith("_orig_mod."):
                        key_name = "_orig_mod." + key_name
                    if key_name in checkpoint:
                        param.data.copy_(checkpoint[key_name])
                    else:
                        raise KeyError(
                            f"필수 키 '{name}' 또는 변환된 '{key_name}'가 체크포인트에 존재하지 않습니다 (adapter: {adapter_tag})."
                        )

        copy_adapter_params(self.unet.named_parameters(), sd["sd_encoder"], "default_encoder")
        copy_adapter_params(self.unet.named_parameters(), sd["sd_decoder"], "default_decoder")
        copy_adapter_params(self.unet.named_parameters(), sd["sd_other"], "default_others")
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # VAE의 경우, 체크포인트에 저장된 키는 "_orig_mod.vae...." 형태이므로 이를 제거해줍니다.
        def strip_orig_mod(state_dict):
            new_sd = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_sd[k[len("_orig_mod."):]] = v
                else:
                    new_sd[k] = v
            return new_sd

        # VAE 관련 LoRA 어댑터 구성 및 파라미터 로드
        vae_lora_config = LoraConfig(
            r=sd["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"]
        )
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        if "sd_vae_enc" not in sd:
            raise KeyError("필수 키 'sd_vae_enc'가 체크포인트에 존재하지 않습니다.")
        # 체크포인트의 VAE 인코더 state dict에서 "_orig_mod." 접두사를 제거한 후 로드
        self.vae_enc.load_state_dict(strip_orig_mod(sd["sd_vae_enc"]))
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        if "sd_vae_dec" not in sd:
            raise KeyError("필수 키 'sd_vae_dec'가 체크포인트에 존재하지 않습니다.")
        self.vae_dec.load_state_dict(strip_orig_mod(sd["sd_vae_dec"]))






    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        # text_emb는 이제 더미값(dummy)으로 대체됩니다.
        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb).sample
        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out, direction=direction)
        return x_out_decoded

    # @staticmethod
    # def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb):
    #     B = x.shape[0]
    #     assert direction in ["a2b", "b2a"]
        
    #     x_enc = vae_enc(x, direction=direction).to(x.dtype)

    #     # 배치 크기와 맞추기 ## 배치사이즈 1 이상으로 하면 오류 발생해서 수정함(update_03.22)
    #     if timesteps.dim() == 0 or timesteps.shape[0] == 1:
    #         timesteps = timesteps.expand(B)

    #     model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb).sample

    #     x_out = torch.stack([
    #         sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample
    #         for i in range(B)
    #     ])

    #     x_out_decoded = vae_dec(x_out, direction=direction)
    #     return x_out_decoded

    
    # ## 배치 사이즈 관련해서 위 코드 수정함. ## update(03.22)
    # @staticmethod
    # def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb):
    #     B = x.shape[0]
    #     assert direction in ["a2b", "b2a"]

    #     # 인코딩
    #     x_enc = vae_enc(x, direction=direction).to(x.dtype)

    #     # timesteps 배치 크기에 맞게 확장 (버그 수정)
    #     if timesteps.dim() == 0:
    #         timesteps = timesteps.expand(B)
    #     elif timesteps.dim() == 1 and timesteps.shape[0] == 1:
    #         timesteps = timesteps.expand(B)
    #     elif timesteps.shape[0] != B:
    #         raise ValueError(f"timesteps shape mismatch: expected ({B},), got {timesteps.shape}")

    #     # 예측
    #     model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb).sample

    #     # 샘플링을 배치 크기만큼 반복 수행
    #     x_out = torch.stack([
    #         sched.step(
    #             model_pred[i], timesteps[i], x_enc[i], return_dict=True
    #         ).prev_sample for i in range(B)
    #     ])

    #     # 디코딩
    #     x_out_decoded = vae_dec(x_out, direction=direction)
    #     return x_out_decoded

    def get_traininable_params(self, unet, vae_a2b, vae_b2a):
        # add all unet parameters
        params_gen = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)
        unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
        for n,p in unet.named_parameters():
            if "lora" in n and "default" in n:
                assert p.requires_grad
                params_gen.append(p)
        
        # add all vae_a2b parameters
        for n,p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        for n,p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
        return params_gen

    def forward(self, x_t, direction=None):
        # direction은 기존 값(예: "a2b" 또는 "b2a")를 사용합니다.
        if direction is None:
            assert self.direction is not None
            direction = self.direction
        # 텍스트(캡션) 관련 처리를 모두 제거하고,
        # UNet의 text 조건 입력에 대해 고정된 더미 텍스트 임베딩을 생성합니다.
        batch_size = x_t.shape[0]
        token_length = 77  # 일반적인 토큰 길이
        hidden_dim = self.unet.config.cross_attention_dim  # UNet 설정에서 추출
        dummy_text_emb = torch.zeros((batch_size, token_length, hidden_dim), device=x_t.device)
        return self.forward_with_networks(x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, dummy_text_emb)
