import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import transformers
import matplotlib.pyplot as plt




def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

def create_lora_config():
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return lora_config

def load_model(model_name):
    quant_config = create_bnb_config()
    lora_config = create_lora_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map={"":0}, max_memory='12000MB')
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,)
    
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def save_pretrained(model_name, checkpoint_path):

    """
    Loads a QLORa fine-tuned model from a checkpoint and saves it.

    Args:
    model_name (str): Name of the base model.
    checkpoint_path (str): Path to the checkpoint directory.
    """

    save_path = checkpoint_path.replace("checkpoint", "model")
    config = create_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=config, device_map={"":0})
    model = PeftModel.from_pretrained(model_name, checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path) 


def train_model(model, tokenizer, data, output_dir):
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            max_steps=1000,
            save_steps=100,
            num_train_epochs=4,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=1,
            output_dir=output_dir,
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    trainer.model.save_pretrained(output_dir)


def prepare_data():
    train_data = load_dataset("json", data_files='train.jsonl', split="train")
    eval_data = load_dataset("json", data_files='validation.jsonl', split="train")

    return train_data, eval_data

def formatting_func(workout):
    text = f"### The following is a workout from Crossfit Tetuan: {workout['workout']}"
    return text

def generate_and_tokenize_prompt(tokenizer, prompt):
    return tokenizer(formatting_func(prompt))

def tokenize_dataset(tokenizer, dataset):
    return dataset.map(lambda samples: generate_and_tokenize_prompt(tokenizer, samples))

def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.savefig('lengths.png')

def identify_entries_over_threshold(dataset, length_threshold=512):
    outliers = []
    filtered_dataset = []

    for index, item in enumerate(dataset):
        if len(item['input_ids']) > length_threshold:
            outliers.append((index, item))
        else:
            filtered_dataset.append(item)

    # Optionally, print information about the outliers
    for index, outlier in outliers:
        print(f"Outlier found at index {index}: {outlier}")

    return filtered_dataset

def generate_and_tokenize_prompt_with_padding(tokenizer, prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_dataset_with_padding(tokenizer, dataset):
    return dataset.map(lambda samples: generate_and_tokenize_prompt_with_padding(tokenizer, samples))



#model = prepare_model_for_kbit_training(model)

#model = get_peft_model(model, config)

#data = load_dataset("Abirate/english_quotes")
#data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)



#trainer = transformers.Trainer(
    #model=model,
    #train_dataset=data["train"],
    #args=transformers.TrainingArguments(
        #per_device_train_batch_size=1,
        #gradient_accumulation_steps=1,
        #warmup_steps=2,
        ## max_steps=20,
        ## report_to="tensorboard",
        #save_steps=100,
        #num_train_epochs=4,
       #learning_rate=2e-5,
        #fp16=True,
        #logging_steps=1,
        #output_dir="outputs",
        #optim="paged_adamw_8bit"
    #),
    #data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
#)
# trainer.train()

if '__main__' == __name__:

    # model_name = "mistralai/Mistral-7B-v0.1"

    # model, tokenizer = load_model(model_name)

    # train_data, eval_data = prepare_data()

    # train_data = tokenize_dataset(tokenizer, train_data)
    # eval_data = tokenize_dataset(tokenizer, eval_data)

    # plot_data_lengths(train_data, eval_data)

    # identify_entries_over_threshold(train_data)
    # identify_entries_over_threshold(eval_data)

    # train_data = tokenize_dataset_with_padding(tokenizer, train_data)
    # eval_data = tokenize_dataset_with_padding(tokenizer, eval_data)

    # plot_data_lengths(train_data, eval_data)

    # tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True)

    # model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda:0")
    # model.eval()
    # with torch.no_grad():
    #     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.2)[0], skip_special_tokens=True))


    # train_model(model, tokenizer, train_data, "outputs/mistralTetuan")
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # quant_config = create_bnb_config()

    # base_model = AutoModelForCausalLM.from_pretrained(
    # model_name,  # Mistral, same as before
    # quantization_config=quant_config,  # Same quantization config as before
    # device_map={"": 0},  # Use GPU 0
    # trust_remote_code=True,
    # max_memory='12000MB'
    # # use_auth_token=True
    # )

    # tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True)

    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

    base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    # use_auth_token=True
)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

    ft_model = PeftModel.from_pretrained(base_model, "outputs/mistralTetuan/checkpoint-900")

    eval_prompt = "The following is a workout from Crossfit Tetuan: "
    model_inputs = tokenizer(eval_prompt, return_tensors="pt").to("cuda:0")

    ft_model.eval()
    with torch.no_grad():
        print(tokenizer.decode(ft_model.generate(**model_inputs, max_new_tokens=250, repetition_penalty=1.2)[0], skip_special_tokens=True))

