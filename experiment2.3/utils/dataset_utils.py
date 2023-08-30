from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

class PrepareDataset:
    def __init__(self,
                tokenizer,
                prompter,
                ):
        self.tokenizer = tokenizer
        self.prompter = prompter

    def tokenize(self,
                 prompt,
                 add_eos_token = True,
                 max_length = 512,
                 ):
        result = self.tokenizer(prompt,
                                truncation = True,
                                max_length = max_length,
                                padding = False,
                                return_tensors = None,
                                )
        if (   
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token
        ):
            
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()

        if "token_type_ids" in result.keys():
            result.pop("token_type_ids")
        return result
    
    def generate_and_tokenize_prompt(self, dataset):
        full_prompt = self.prompter.generate_prompt(dataset["instruction"],
                                                    dataset["input"],
                                                    dataset["output"])
        
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt
    
    def dataloader(self,
                   train_data,
                   batch_size: int,
                   valid_data = None
                   ):
    
        train_dataloader = DataLoader(dataset = train_data,
                                      batch_size = batch_size,
                                      collate_fn = DataCollatorForSeq2Seq(tokenizer = self.tokenizer,
                                                                          padding = True,
                                                                          return_tensors = "pt"),
                                      pin_memory = True) 
        if valid_data:
            valid_dataloader = DataLoader(dataset = train_data,
                                          batch_size = batch_size,
                                          collate_fn = DataCollatorForSeq2Seq(tokenizer = self.tokenizer,
                                                                              padding = True,
                                                                              return_tensors = "pt"),
                                          pin_memory = True)
            
            return train_dataloader, valid_dataloader
        else:
            return train_dataloader




