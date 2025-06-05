import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")
# Aggiungi callback al trainer
from transformers import TrainerCallback


class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            loss_callback.on_log(logs)


# Configurazione della quantizzazione 4-bit per QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Modello Qwen (leggero e performante)
model_name = "Qwen/Qwen1.5-1.8B"

print(f"Caricamento del modello: {model_name}")

# Caricamento del modello con quantizzazione
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Caricamento del tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Aggiunta del pad token se mancante
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preparazione del modello per il training con quantizzazione
model = prepare_model_for_kbit_training(model)

# Configurazione LoRA ottimizzata per Qwen
lora_config = LoraConfig(
    r=16,  # Rank della decomposizione
    lora_alpha=32,  # Parametro di scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Tutti i layer lineari
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Applicazione di LoRA al modello
model = get_peft_model(model, lora_config)

print("Modello preparato con QLoRA:")
model.print_trainable_parameters()

# Dataset Alpaca per instruction tuning
dataset_name = "tatsu-lab/alpaca"
print(f"Caricamento del dataset: {dataset_name}")

# Caricamento e preparazione del dataset Alpaca con split train/val/test
full_dataset = load_dataset(dataset_name, split="train[:3000]")  # 3000 esempi totali


def format_alpaca(example):
    """Formatta i dati Alpaca nel formato corretto per Qwen"""
    if example["input"] and example["input"].strip():
        # Con input aggiuntivo
        text = f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    else:
        # Solo istruzione
        text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    return {"text": text}


# Applica la formattazione
full_dataset = full_dataset.map(format_alpaca)

# Split del dataset: 70% train, 15% validation, 15% test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Shuffle e split
shuffled_dataset = full_dataset.shuffle(seed=42)
train_dataset = shuffled_dataset.select(range(train_size))
val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
test_dataset = shuffled_dataset.select(range(train_size + val_size, len(full_dataset)))

print(f"Dataset split:")
print(f"  - Train: {len(train_dataset)} esempi")
print(f"  - Validation: {len(val_dataset)} esempi")
print(f"  - Test: {len(test_dataset)} esempi")


# Tokenizzazione dei dataset
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,
        return_overflowing_tokens=False,
    )
    return tokenized


# Tokenizza tutti i dataset
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)

test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=test_dataset.column_names,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Non è masked language modeling
)

# Configurazione del training ottimizzata per Qwen
training_args = TrainingArguments(
    output_dir="./qwen_qlora_alpaca",
    num_train_epochs=2,  # 2 epoch per training più completo
    per_device_train_batch_size=8,  # Qwen 1.8B può gestire batch più grandi
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=200,
    max_steps=1000,  # Più step per training migliore
    learning_rate=5e-5,  # Learning rate leggermente più alto per Qwen
    fp16=True,
    logging_steps=25,
    save_steps=200,
    evaluation_strategy="steps",  # Abilita evaluation
    eval_steps=200,  # Valuta ogni 200 step
    save_strategy="steps",
    load_best_model_at_end=True,  # Carica il modello migliore
    metric_for_best_model="eval_loss",  # Usa eval_loss per scegliere il migliore
    greater_is_better=False,  # Per la loss, più bassa è meglio
    ddp_find_unused_parameters=False,
    group_by_length=True,
    report_to=None,
    run_name="qwen_qlora_alpaca",
    remove_unused_columns=False,
    logging_dir="./logs",  # Directory per i log
    save_total_limit=1,  # Mantieni solo i 3 checkpoint migliori
)

# Inizializzazione del trainer con validation set
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,  # Aggiungi validation set
    data_collator=data_collator,
)


if torch.cuda.is_available():
    print(f"Memoria allocata: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Memoria riservata: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


# Training con logging della loss
class LossCallback:
    def __init__(self):
        self.losses = []
        self.eval_losses = []

    def on_log(self, logs):
        if 'loss' in logs:
            self.losses.append(logs['loss'])
            print(f"Step {len(self.losses) * 25}: Training Loss = {logs['loss']:.4f}")
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
            print(f"Step {len(self.eval_losses) * 200}: Validation Loss = {logs['eval_loss']:.4f}")


# Callback personalizzato per stampare la loss
loss_callback = LossCallback()



trainer.add_callback(CustomCallback())

# Training
training_result = trainer.train()

# Stampa statistiche finali del training
print("\n" + "=" * 50)
print("RISULTATI TRAINING:")
print("=" * 50)
print(f"Training completato!")
print(f"Loss finale: {training_result.training_loss:.4f}")
print(f"Passi totali: {training_result.global_step}")
print(f"Tempo totale: {training_result.training_time:.2f} secondi")

# Salvataggio del modello
trainer.model.save_pretrained("./qwen_qlora_alpaca")
tokenizer.save_pretrained("./qwen_qlora_alpaca")

print("Fine-tuning completato!")
print("Modello salvato in: ./qwen_qlora_alpaca")

# === VALUTAZIONE COMPLETA ===
print("\n" + "=" * 50)
print("VALUTAZIONE SUL TEST SET:")
print("=" * 50)

# Valutazione sul test set
test_results = trainer.evaluate(eval_dataset=test_tokenized)
print(f"Test Loss: {test_results['eval_loss']:.4f}")
print(f"Test Perplexity: {torch.exp(torch.tensor(test_results['eval_loss'])):.2f}")

# === TEST QUALITATIVO ===
print("\n" + "=" * 50)
print("TEST QUALITATIVO:")
print("=" * 50)

# Funzione per generare testo con formato Qwen


def generate_response(instruction, input_text="", max_tokens=150):
    """Genera risposta usando il formato chat di Qwen"""
    if input_text and input_text.strip():
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Estrai solo la parte della risposta
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]

    return response.strip()


# Test con domande dal test set
print("=== ESEMPI DAL TEST SET ===")
for i in range(3):
    example = test_dataset[i]
    # Estrai istruzione e input dal testo formattato
    text = example['text']
    if '<|im_start|>user\n' in text:
        user_content = text.split('<|im_start|>user\n')[1].split('<|im_end|>')[0]
        expected_response = text.split('<|im_start|>assistant\n')[1].split('<|im_end|>')[0]

        print(f"\n--- Esempio {i + 1} ---")
        print(f"Domanda: {user_content}")
        print(f"Risposta attesa: {expected_response}")
        print(f"Risposta generata: {generate_response(user_content)}")
        print("-" * 40)

# Test con domande personalizzate
print("\n=== TEST CON DOMANDE PERSONALIZZATE ===")
test_questions = [
    "Spiega cos'è l'intelligenza artificiale in modo semplice",
    "Come si prepara un caffè espresso perfetto?",
    "Quali sono i vantaggi del machine learning?",
    "Scrivi una breve storia su un robot che impara a cucinare",
    "Come posso migliorare le mie capacità di programmazione?",
]

for i, question in enumerate(test_questions):
    print(f"\n--- Domanda {i + 1} ---")
    print(f"Q: {question}")
    response = generate_response(question)
    print(f"A: {response}")
    print("-" * 40)

# === ANALISI DELLE PERFORMANCE ===
print("\n" + "=" * 50)
print("ANALISI DELLE PERFORMANCE:")
print("=" * 50)



