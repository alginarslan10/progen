# %%
import torch
from models.progen.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from torchinfo import summary
from datasets import load_metric

import numpy as np
from peft import LoraConfig,TaskType,get_peft_model
from transformers import TrainingArguments,Trainer,DataCollatorForLanguageModeling,PreTrainedTokenizerFast

# %%
#Hyperparamters
lr = 2e-4
batch_size = 4
num_of_epochs = 60

# %%
checkpoint = "/home/algin/Downloads/progen/progen2/checkpoints/progen2-small"
device =  "cuda:0"
tokenizer_conf = "/home/algin/Downloads/progen/progen2/autotokenizer/tokenizer_config.json"

example_token = "2GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATEAYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM1"

torch.cuda.set_device(device)

#Init Tokenizer
#with open(tokenizer_conf,"r") as f:
    #tokenizer = Tokenizer.from_str(f.read())


tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_conf)
tokenizer.pad_token_id = 0
tokenizer.pad_token = "<|pad|>"

#Training Arguments
training_arguments = TrainingArguments(
    output_dir="./",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_of_epochs,
    weight_decay=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

accuracy = load_metric("accuracy")

# %%
#Init Model
model = ProGenForCausalLM.from_pretrained(checkpoint)
#summary(model)
#print(model)

#Init Peft
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,inference_mode=False,lora_alpha=16,lora_dropout=0.1,bias="lora_only",target_modules=["qkv_proj","lm_head"])
peft_model = get_peft_model(model,peft_config)
#peft_model.print_trainable_parameters()


# %%
def tokenize(sequence_list:list):
    return tokenizer(sequence_list,padding=True,return_tensors="pt")

def cross_entropy(logits, target, reduction='mean'):
        return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)

def ce_loss(tokens):    

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
            logits = peft_model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]
        
        return cross_entropy(logits,target).item()

def compute_metrics(p):
    
    predictions,labels = p
    predictions = np.argmax(predictions,axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions,references=labels),
            "loss": ce_loss(predictions)}



# %%
#Dataset

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
df = pd.read_csv("../format_data_file/xylanase.csv",header=0)

sequences = df["Sequence"].to_numpy()

train, test = train_test_split(sequences,test_size=0.2,random_state=1)

tokenized_train_dataset = tokenize(list(train))["input_ids"]
tokenized_validation_dataset = tokenize(list(test))["input_ids"]


# %%
#Trainer
trainer = Trainer(
    model=peft_model,
    args=training_arguments,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# %%
print(torch.cuda.is_available())
trainer.train()



# %%
