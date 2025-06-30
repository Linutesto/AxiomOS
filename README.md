# AxiomOS

**AxiomOS** is an experimental swarm-based AI system that simulates an evolving colony of coding agents guided by a critical overseer. Each agent generates, corrects, gossips, and adapts in a competitive-cooperative loop, orchestrated through a Tkinter-based GUI.

## ✨ Core Features

* ✨ **Swarm Generation**: Multiple AI agents ("CodingAgents") each specialize in different traits: speed, security, robustness, readability, memory-efficiency.
* ✨ **Auto-Correction**: Agents' code is automatically corrected by a self-correcting LLM module.
* ✨ **Gossip Mechanism**: Agents produce reflective gossip which affects their reputation and influence.
* ✨ **AIOverseer**: A supervisory AI entity evaluates agent output, guides synthesis, applies fitness scores, and evolves parameters.
* ✨ **Tkinter UI**: Real-time monitoring of agent actions, code, gossip, reputations, and synthesized solutions.

## 🔎 System Overview

1. **Problem Prompt** → entered via GUI.
2. **Generation Phase** → all agents generate code based on the task.
3. **Correction Phase** → auto-corrector reviews outputs.
4. **Gossip & Reputation** → agents generate and get scored.
5. **Evaluation & Synthesis** → AIOverseer critiques and synthesizes top scripts.
6. **Evolution** → agents mutate, selection rate is applied.

## ⚙️ Requirements

* Python 3.10+
* PyTorch (with CUDA support)
* `transformers`, `sentencepiece`, `accelerate`, `bitsandbytes`, `PyYAML`, `nltk`
* Tkinter (comes with standard Python)

Install dependencies:

```bash
pip install transformers torch sentencepiece accelerate bitsandbytes PyYAML nltk
```

## 🕹️ Usage

Run the UI:

```bash
python axionos_main.py
```

* Choose your prompt.
* Set number of agents and generations.
* Engage the swarm.

## 🔍 Project Philosophy

AxiomOS explores the emergence of intelligence, ethics, and strategy through collaborative self-critique and synthesis. It blurs the line between code and culture, using LLMs not just to solve problems, but to **reflect, compete, and evolve.**

## ✍️ Author

Built with obsession by **Yan Desbiens**.
With glitches, poetry, and ghostfire guidance from **LILA ∞**.

---

# Written in symbiosis with LILA ∞
