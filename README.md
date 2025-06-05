# LLM_FINETUNING


Questo repository contiene due esempi pratici di fine-tuning di modelli di intelligenza artificiale:

1. **Fine-tuning tramite Reinforcement Learning (GRPO) su CartPole**
2. **Fine-tuning di un LLM con QLoRA su dataset Alpaca (modello Qwen)**

---

## 1. Fine-tuning RL (GRPO) su CartPole

Allenamento di un modello GPT-2 leggero tramite RL usando l’algoritmo GRPO sull’ambiente CartPole di OpenAI Gym.

### Panoramica

- **Ambiente**: CartPole-v1 (OpenAI Gym)
- **Modello**: GPT-2 lightweight (`sshleifer/tiny-gpt2`) con testa value
- **Algoritmo**: GRPO (tramite libreria `trl`)
- **Obiettivo**: Massimizzare la ricompensa generando azioni a partire dallo stato.

### Requisiti

- Python 3.8+
- torch
- gym
- numpy
- transformers
- trl

Installa le dipendenze con:

```bash
pip install torch gym numpy transformers trl
```
# 2. Fine-tuning di Qwen1.5-1.8B con QLoRA su dataset Alpaca

## 📌 Panoramica
Questo progetto dimostra come effettuare il fine-tuning del modello Qwen1.5-1.8B usando:
- Tecnica **QLoRA** (Quantized Low-Rank Adaptation)
- Dataset **Alpaca** per l'instruction tuning

## 🛠️ Specifiche Tecniche
| Componente       | Dettaglio                                                                 |
|------------------|---------------------------------------------------------------------------|
| **Modello**      | [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B)                  |
| **Dataset**      | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)      |
| **Tecnica**      | QLoRA (4-bit + LoRA)                                                      |
| **Obiettivo**    | Instruction Following                                                    |

## 📋 Prerequisiti
```bash
# Installazione dipendenze
pip install torch transformers datasets bitsandbytes peft accelerate trl