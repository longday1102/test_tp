from transformers import AutoTokenizer
import torch

from bitsandbytes.optim import AdamW8bit

import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from prompt import Prompter
from utils.dataset_utils import PrepareDataset
from utils.model_utils import load_model
import tensor_parallel as tp
from datasets import load_dataset

from tqdm.auto import tqdm
import math
import os

from huggingface_hub import login

login(token = "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")

def main():
    
    # LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    # WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    # WORLD_RANK = int(os.environ['RANK'])
    # dist.init_process_group(backend='nccl', rank = LOCAL_RANK, world_size = WORLD_SIZE)

    # MODEL LIST
    model_list = {
        "llama-2": [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
        "bloom": [
            "bigscience/bloom-7b1",
            "bigscience/bloom",
        ],
        "polyLM": "DAMO-NLP-MT/polylm-13b",
    }
    # TOKENIZER
    # model_checkpoint = model_list["llama-2"][0]
    model_checkpoint = "longhoang06/my_model"
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
    if model_checkpoint in model_list["llama-2"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # DATASET
    dataset = load_dataset("MBZUAI/Bactrian-X", "vi", split = "train")
    splited_dataset = dataset.train_test_split(test_size = 0.1, seed = 42)
    train_data = splited_dataset["test"]
    valid_data = splited_dataset["test"]

    prompter = Prompter()
    prepare_dataset = PrepareDataset(prompter = prompter,
                                     tokenizer = tokenizer)
    
    train_data = train_data.shuffle().map(prepare_dataset.generate_and_tokenize_prompt, 
                                          num_proc = 13)
    valid_data = valid_data.map(prepare_dataset.generate_and_tokenize_prompt,
                                num_proc = 13)
    
    train_data = train_data.remove_columns(["instruction", "input", "id", "output"])
    valid_data = valid_data.remove_columns(["instruction", "input", "id", "output"])

    train_data.set_format("torch")
    valid_data.set_format("torch")

    train_dataloader = prepare_dataset.dataloader(train_data = train_data, batch_size = 1)

    # TRAINING
    model = load_model(model_checkpoint = model_checkpoint,
                       world_size = 2,
                       quantize_mode = False,
                       lora_mode = False,
                       half_precision_mode = False)

    num_epochs = 1
    total_steps = num_epochs * len(train_dataloader)
    optimizer = AdamW(model.parameters(), lr = 2e-5)
    lr_scheduler = CosineAnnealingLR(
        optimizer = optimizer,
        T_max = total_steps,
        eta_min = 1e-8,
        last_epoch = -1,
    )

    log_steps = 1
    scheduler_steps = 100
    gradient_accumulation_steps = 4
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader)):
            batch = {k:v.to("cuda:0") for k, v in batch.items()}
      
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            loss /= gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % scheduler_steps == 0:
                lr_scheduler.step()
            
            if (step + 1) % log_steps == 0:
                cur_loss = total_loss/(step + 1)
                print(f'Epoch: {epoch + 1} -- step: {step + 1} -- train_loss: {total_loss/(step + 1)} -- ppl: {math.exp(cur_loss)}')

if __name__ == "__main__":
    main()
