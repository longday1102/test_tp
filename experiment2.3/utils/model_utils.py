from transformers import AutoConfig,  AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils.bitsandbytes import replace_with_bnb_linear
from peft import LoraConfig, get_peft_model
import accelerate
import torch
import tensor_parallel as tp
from huggingface_hub import hf_hub_download

import json

def load_model(model_checkpoint,
               quantize_mode: bool = False,
               lora_mode: bool = False,
               half_precision_mode: bool = False,):

    with accelerate.init_empty_weights():
        if half_precision_mode == True:
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_checkpoint,
                                                                                trust_remote_code = True),
                                                    trust_remote_code = True).half()
        else:
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_checkpoint,
                                                                                trust_remote_code = True),
                                                    trust_remote_code = True)
            
    model = tp.TensorParallelPreTrainedModel(model, sharded = False)
    
    if quantize_mode == True:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
        )
        model = replace_with_bnb_linear(
            model,
            quantization_config = bnb_config,
        )
        model.is_loaded_in_4bit = True
    
    device_map = tp.infer_sharded_device_map(model)
    with open(hf_hub_download(model_checkpoint, "pytorch_model.bin.index.json"), "r") as index_file:
        shard_filenames = set(json.load(index_file)["weight_map"].values())
    
    for shard_filename in sorted(shard_filenames):
        shard_path = hf_hub_download(model_checkpoint, shard_filename)
        converted_state_dict = tp.convert_state_dict(
            torch.load(shard_path),
            model.tensor_parallel_config,
            world_size = 2,
            for_pretrained = True,
        )
        
        for param_name, param in converted_state_dict.items():
            module_name = param_name

            while len(module_name) > 0 and module_name not in device_map:
                module_name = ".".join(module_name.split(".")[:-1])
            param_device = device_map[module_name]

            accelerate.utils.set_module_tensor_to_device(model, param_name, param_device, value = param)
            converted_state_dict[param_name] = None
        del converted_state_dict
        
    if lora_mode == True:
        lora_config = LoraConfig(
            r = 8,
            lora_alpha = 32,
            lora_dropout = 0.05,
            bias = "none",
            task_type = "CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    return model
