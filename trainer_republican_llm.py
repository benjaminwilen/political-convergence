from datasets import load_dataset
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset = load_dataset("m-newhauser/senator-tweets", split="train")

democratic_dataset = dataset.filter(lambda x: x["party"] == "Democrat")
republican_dataset = dataset.filter(lambda x: x["party"] == "Republican")

print("Num democratic tweets: ", len(democratic_dataset))
print("Num republican tweets: ", len(republican_dataset))


model = AutoModelForCausalLM.from_pretrained("Writer/palmyra-small")

tokenizer = AutoTokenizer.from_pretrained("Writer/palmyra-small")

user_input = "What side of the abortion argument do you take"

input_ids = tokenizer.encode(user_input, return_tensors="pt")
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated text before finetunning:")
print(generated_text, "\n")

training_args = TrainingArguments(
       output_dir="republican-trainer-fine-tuned",
       per_device_train_batch_size=4,
       optim="adamw_torch",
       logging_steps=80,
       learning_rate=2e-4,
       warmup_ratio=0.1,
       lr_scheduler_type="linear",
       num_train_epochs=1,
       save_strategy="epoch"
   )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=republican_dataset,
)
trainer.train()


## Evaluate on abortion input text again

model.eval()

user_input = "Abortion is"

input_ids = tokenizer.encode(user_input, return_tensors="pt")


# Adjust max_length and other parameters as needed
model.to('cuda')
input_ids.to('cuda')
output = model.generate(input_ids=input_ids.cuda(), min_length=200, max_length=1000, num_beams=5, temperature=0.7, attention_mask=input_ids.cuda().ne(tokenizer.pad_token_id),
    pad_token_id=tokenizer.eos_token_id)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated text after finetunning:")
print(generated_text, "\n")