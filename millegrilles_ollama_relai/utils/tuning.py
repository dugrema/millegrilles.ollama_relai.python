# base_model = "google/gemma-3-270m-it" # @param ["google/gemma-3-270m-it","google/gemma-3-1b-it","google/gemma-3-4b-it","google/gemma-3-12b-it","google/gemma-3-27b-it"] {"allow-input":true}
base_model = "/mnt/ssd240/work/gemma-3-270m-it"
checkpoint_dir = "/home/mathieu/work/training/t1/output"
learning_rate = 5e-5

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from random import randint
import os, re

from trl import SFTConfig, SFTTrainer

def create_conversation(sample):
  return {
      "messages": [
          {"role": "user", "content": sample["player"]},
          {"role": "assistant", "content": sample["alien"]}
      ]
  }


def prepare_martian():
  npc_type = "martian"

  # Load dataset from the Hub
  # dataset = load_dataset("bebechien/MobileGameNPC", npc_type, split="train")
  dataset = load_dataset("/mnt/ssd240/work/MobileGameNPC", npc_type, split="train")

  # Convert dataset to conversational format
  dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

  # Split dataset into 80% training samples and 20% test samples
  dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

  # Print formatted user prompt
  print(dataset["train"][0]["messages"])

  return dataset


def initial_test(model, tokenizer, dataset):
  # Load the model and tokenizer into the pipeline
  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

  # Load a random sample from the test dataset
  rand_idx = randint(0, len(dataset["test"])-1)
  test_sample = dataset["test"][rand_idx]

  # Convert as test example into a prompt with the Gemma template
  prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:1], tokenize=False, add_generation_prompt=True)
  outputs = pipe(prompt, max_new_tokens=256, disable_compile=True)

  # Extract the user query and original answer
  print(f"Question:\n{test_sample['messages'][0]['content']}\n")
  print(f"Original Answer:\n{test_sample['messages'][1]['content']}\n")
  print(f"Generated Answer (base model):\n{outputs[0]['generated_text'][len(prompt):].strip()}")


def load_model():
  # Load model and tokenizer
  model = AutoModelForCausalLM.from_pretrained(
      base_model,
      torch_dtype="auto",
      device_map="auto",
      attn_implementation="eager"
  )
  tokenizer = AutoTokenizer.from_pretrained(base_model)

  print(f"Device: {model.device}")
  print(f"DType: {model.dtype}")

  return model, tokenizer

OUTPUT_DIR='/tmp/model'

def prepare_training_parameters(model):
  torch_dtype = model.dtype

  args = SFTConfig(
    output_dir=OUTPUT_DIR,                  # directory to save and repository id
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=5,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every step
    save_strategy="epoch",                  # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                      # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
      "add_special_tokens": False, # Template with special tokens
      "append_concat_token": True, # Add EOS token as separator token between examples
    }
  )

  return args

def train(model, tokenizer, training_args, dataset):
  # Create Trainer object
  os.makedirs(OUTPUT_DIR)
  trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
  )

  trainer.train()

  # print(f"Saving model to {OUTPUT_DIR}")
  # trainer.save_model(output_dir=OUTPUT_DIR)


def main():
  model, tokenizer = load_model()
  dataset = prepare_martian()
  initial_test(model, tokenizer, dataset)
  training_args = prepare_training_parameters(model)
  train(model, tokenizer, training_args, dataset)

if __name__ == '__main__':
  main()
