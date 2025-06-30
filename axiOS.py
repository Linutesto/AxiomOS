import os
import sys
import torch
import gc
import random
import time
import textwrap
import traceback
import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
from typing import List, Dict, Optional, Any
from collections import deque
import ast
import pickle
import yaml

# --- System Configuration v19.2 - AutoGenesis ---
# Set PyTorch's CUDA memory allocation configuration to prevent fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Define constant paths for configuration and data storage.
AGENT_MEMORY_ARCHIVE = "agent_memory"
CONFIG_PATH = "axiomos_config.yaml"
CURRENT_SCRIPT_PATH = os.path.realpath(__file__)

# --- Dependency Check ---
# Verify that all required libraries are installed.
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
    import nltk
except ImportError:
    # If dependencies are missing, show an error and exit.
    # This prevents the application from crashing unexpectedly later.
    messagebox.showerror(
        "Dépendances manquantes",
        "Les modules requis sont introuvables. Veuillez les installer avec la commande suivante:\n"
        "`pip install transformers torch sentencepiece accelerate bitsandbytes PyYAML nltk`"
    )
    sys.exit(1)

# --- Logging and Color Configuration ---
class Colors:
    """Defines ANSI color codes for console output to improve readability."""
    AGENT = '\033[95m'
    ENGINE = '\033[94m'
    ERROR = '\033[91m'
    INFO = '\033[93m'
    SYSTEM = '\033[92m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    GOSSIP = '\033[96m' # Cyan for gossip
    RESET = '\033[0m'

# --- Default Parameters ---
# Set the default model path. The transformers library handles caching.
# IMPORTANT: Update this path to your local model directory.
DEFAULT_MODEL_PATH = "/home/yan/Desktop/YANOS/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80" # Remplacez par votre chemin local si nécessaire, ex: "/path/to/your/model"

# --- AIOverseer DNA: Handles dynamic configuration ---
class OverseerDNA:
    """Manages the evolving configuration parameters of the system."""
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Loads configuration from a YAML file or creates a default one."""
        if not os.path.exists(self.config_path):
            config = {"mutation_rate": 0.4, "selection_rate": 0.25}
            self._save_config(config)
            return config
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {"mutation_rate": 0.4, "selection_rate": 0.25}

    def _save_config(self, config):
        """Saves the current configuration to the YAML file."""
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

    def mutate(self):
        """Applies a small random change to the configuration parameters."""
        mutated_config = self.config.copy()
        param_to_mutate = "None"
        if random.random() < 0.5:
            param_to_mutate = random.choice(["mutation_rate", "selection_rate"])
            current_value = mutated_config.get(param_to_mutate, 0.5)
            mutation_amount = random.uniform(-0.05, 0.05)
            new_value = round(min(1.0, max(0.1, current_value + mutation_amount)), 3)
            mutated_config[param_to_mutate] = new_value

        self._save_config(mutated_config)
        self.config = mutated_config
        return f"ADN muté: {param_to_mutate} ajusté à {mutated_config.get(param_to_mutate, 'N/A')}"

# --- FMM Memory: A simple, deque-based memory store ---
class FractalMemory:
    """A memory module for agents, using a deque to store recent experiences."""
    def __init__(self, maxlen=256):
        self.data = deque(maxlen=maxlen)

    def remember(self, item: Dict):
        """Adds an item to the agent's memory."""
        self.data.append(item)

# --- Moral Filter: A safeguard against harmful code generation ---
class MoralCompass:
    """
    A static class to filter out potentially dangerous code patterns using a hybrid
    whitelist (for imports) and blacklist (for function calls) approach.
    """
    ALLOWED_IMPORTS = {
        'math', 'random', 'collections', 'datetime', 'time', 're', 'json', 'itertools',
        'functools', 'heapq', 'bisect', 'array', 'struct', 'copy', 'typing', 'enum',
        'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn', 'requests', 'yaml'
    }
    FORBIDDEN_PATTERNS = [
        r"\bos\.", r"\bsubprocess\.", r"\bshutil\.", r"\bctypes\.", r"\bsys\.",
        r"open\s*\(", r"__import__", r"eval\s*\(", r"exec\s*\("
    ]

    @staticmethod
    def filter(code: str) -> bool:
        """Checks code for safety."""
        if any(re.search(pattern, code) for pattern in MoralCompass.FORBIDDEN_PATTERNS):
            return False
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] not in MoralCompass.ALLOWED_IMPORTS:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] not in MoralCompass.ALLOWED_IMPORTS:
                        return False
        except SyntaxError:
            return False
        return True

# --- UI Streamer: Handles streaming text to the UI ---
class UIStreamer(TextStreamer):
    """A custom TextStreamer that pushes generated text tokens to the UI queue."""
    def __init__(self, tokenizer, ui_queue: queue.Queue, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.ui_queue = ui_queue
        self.full_text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Called when new text is generated by the model."""
        new_text = text[len(self.full_text):]
        self.full_text = text
        if new_text:
            self.ui_queue.put({"type": "stream_token", "data": new_text})
        if stream_end:
            self.ui_queue.put({"type": "stream_end"})

# --- Cognitive Core: The interface to the Large Language Model ---
class CognitiveCore:
    """Handles all interactions with the LLM, including loading and generation."""
    def __init__(self, model_path: str, ui_queue: queue.Queue):
        self.model_path = model_path
        self.ui_queue = ui_queue
        self.model = None
        self.tokenizer = None
        is_local_path = os.path.isdir(model_path)
        self.use_mock = not (torch.cuda.is_available() and (is_local_path or "google/" in model_path))

        if self.use_mock:
            self.log_message(f"{Colors.WARNING}AVERTISSEMENT: GPU non détecté ou chemin du modèle invalide. Passage en mode simulation.{Colors.RESET}")
        else:
            self._load_model()

    def log_message(self, message: str, stream: bool = False, msg_type: str = "log"):
        """Sends a log or gossip message to the UI queue."""
        self.ui_queue.put({"type": msg_type, "data": message, "stream": stream})

    def _load_model(self):
        """Loads the LLM and tokenizer from the specified path."""
        try:
            self.log_message(f"{Colors.SYSTEM}Chargement du modèle: {self.model_path}...{Colors.RESET}")
            q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=q_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa"
            )
            self.log_message(f"{Colors.SUCCESS}Modèle chargé avec succès sur {self.model.device}.{Colors.RESET}")
        except Exception as e:
            self.log_message(f"{Colors.ERROR}ERREUR FATALE: Échec du chargement du modèle.\n{e}{Colors.RESET}")
            self.use_mock = True

    def _validate_and_clean_output(self, code: str) -> Optional[str]:
        """Cleans and validates raw LLM output."""
        cleaned_code = re.sub(r'```python\n?', '', code, flags=re.IGNORECASE)
        cleaned_code = re.sub(r'```', '', cleaned_code)
        cleaned_code = cleaned_code.replace("<|end_of_turn|>", "").replace("<end_of_turn>", "").strip()

        if not cleaned_code:
            return None
        try:
            ast.parse(cleaned_code)
            return cleaned_code
        except SyntaxError:
            return None # Don't log syntax errors for non-code, like gossip

    def generate(self, prompt: str, max_tokens: int = 512, stream_to_ui: bool = False, validate_as_code: bool = True) -> str:
        """Generates text from a prompt, with optional validation."""
        if self.use_mock:
            return "def solve():\n\t# Réponse simulée\n\treturn 42" if validate_as_code else "Réponse simulée."
        
        generation_kwargs = dict(max_new_tokens=max_tokens, temperature=0.3, do_sample=True, top_p=0.9, top_k=50)

        for attempt in range(2):
            try:
                current_prompt = prompt
                if attempt > 0:
                    self.log_message(f"{Colors.INFO}Validateur cognitif: Nouvelle tentative de génération...{Colors.RESET}")
                    current_prompt = "Votre tentative précédente a produit un résultat invalide. Veuillez réessayer.\n\n" + prompt

                inputs = self.tokenizer(current_prompt, return_tensors="pt").to(self.model.device)
                
                if stream_to_ui:
                    generation_kwargs['streamer'] = UIStreamer(self.tokenizer, self.ui_queue, skip_prompt=True)
                
                outputs = self.model.generate(**inputs, **generation_kwargs)
                decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                if validate_as_code:
                    validated_code = self._validate_and_clean_output(decoded)
                    if validated_code:
                        return validated_code
                else:
                    return decoded.strip() # For gossip, no validation needed
                    
            except Exception as e:
                self.log_message(f"{Colors.ERROR}Erreur de génération à la tentative {attempt+1}: {e}{Colors.RESET}")
        
        self.log_message(f"{Colors.ERROR}Échec de la génération de sortie valide après plusieurs tentatives.{Colors.RESET}")
        return ""

# --- Self-Correction Module ---
class SelfCorrectingModule:
    """A module dedicated to correcting syntax errors and typos in generated code."""
    def __init__(self, model: CognitiveCore):
        self.model = model

    def correct(self, code_to_correct: str) -> str:
        """Attempts to correct the given piece of code."""
        if not code_to_correct:
            return ""

        prompt = (
            "Vous êtes un module d'auto-correction. La tâche suivante est de corriger les erreurs syntaxiques, "
            "les fautes de frappe ou les variables non définies dans le code Python fourni. Ne modifiez PAS la logique. "
            "Si le code semble déjà correct, retournez-le tel quel.\n\n"
            "## Code à corriger:\n"
            "```python\n{code}\n```\n\n"
            "## Instructions:\n"
            "1. Identifiez et corrigez uniquement les erreurs évidentes.\n"
            "2. Ne réécrivez pas la logique et n'ajoutez pas de nouvelles fonctionnalités.\n"
            "3. CRITIQUE: Votre réponse doit être UNIQUEMENT le code Python complet et corrigé. "
            "N'incluez aucune explication."
        ).format(code=code_to_correct)

        corrected_code = self.model.generate(prompt, max_tokens=len(code_to_correct.split('\n')) * 25, validate_as_code=True)
        
        # Fallback to original code if correction fails or returns empty
        return corrected_code if corrected_code else code_to_correct

# --- Swarm Components: Agents and Overseer ---
class CodingAgent:
    """An individual AI agent specializing in a particular coding style or focus."""
    PERSONALITIES = ['vitesse', 'lisibilité', 'sécurité', 'efficacité mémoire', 'robustesse']
    
    def __init__(self, agent_id: int, genome: Dict, model: CognitiveCore):
        self.id = agent_id
        self.genome = genome
        self.model = model
        self.fitness = 0.0
        self.reputation = 1.0 # Initial reputation
        self.memory_file = os.path.join(AGENT_MEMORY_ARCHIVE, f"agent_{self.id}_memory.pkl")
        self.memory = self._load_memory()

    def _create_prompt(self, task_desc: str, template_type: str, gossip_summary: str, **kwargs) -> str:
        """A centralized method for creating prompts for different agent actions."""
        base_prompts = {
            "generate": (
                "Vous êtes un programmeur Python expert. Votre spécialité est '{focus}'.\n\n"
                "## TÂCHE ##\n"
                "Écrivez un script Python complet pour le problème suivant:\n'{task_desc}'\n\n"
                "## POTINS RÉCENTS DE L'ESSAIM ##\n"
                "{gossip_summary}\n\n"
                "## EXIGENCES ##\n"
                "1. Le script doit contenir une fonction 'solve'.\n"
                "2. Le script doit être autonome, exécutable et sans erreur.\n"
                "3. CRITIQUE: Votre réponse doit être UNIQUEMENT du code Python valide. N'ajoutez aucune explication.\n\n"
                "## SCRIPT PYTHON ##\n"
            ),
            "gossip": (
                "Vous êtes l'Agent {id}, un programmeur IA spécialisé dans '{focus}'. Vous venez d'écrire un script pour la tâche: '{task_desc}'.\n"
                "Commentez brièvement votre approche ou une réflexion intéressante en une seule phrase concise. "
                "Exemple: 'J'ai opté pour la récursivité pour plus de clarté, mais je crains pour l'utilisation de la pile.' ou 'La validation des entrées était ma priorité absolue pour ce problème.'\n\n"
                "Votre potin concis:"
            )
        }
        prompt_template = base_prompts[template_type]
        return prompt_template.format(
            id=self.id,
            focus=self.genome['focus'],
            task_desc=task_desc, 
            gossip_summary=gossip_summary,
            **kwargs
        )

    def generate_code(self, task_desc: str, gossip_summary: str) -> str:
        """Generates a new code solution for a given task, considering gossip."""
        self.model.log_message(f"{Colors.AGENT}Agent {self.id} (Focus: {self.genome['focus']}, Réputation: {self.reputation:.2f}) génère un script...{Colors.RESET}")
        prompt = self._create_prompt(task_desc, "generate", gossip_summary)
        code = self.model.generate(prompt, max_tokens=2048)
        self.memory.remember({'action': 'generate', 'task': task_desc, 'code': code})
        return code

    def generate_gossip(self, task_desc: str) -> str:
        """Generates a gossip message about the agent's last action."""
        prompt = self._create_prompt(task_desc, "gossip", gossip_summary="") # No gossip needed to generate gossip
        gossip = self.model.generate(prompt, max_tokens=100, validate_as_code=False)
        return gossip

    def save_agent_state(self):
        """Saves the agent's genome and memory to disk."""
        with open(os.path.join(AGENT_MEMORY_ARCHIVE, f"agent_{self.id}_genome.pkl"), 'wb') as f:
            pickle.dump({'id': self.id, 'genome': self.genome, 'reputation': self.reputation}, f)
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)

    def _load_memory(self) -> FractalMemory:
        """Loads the agent's memory from a pickle file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f: return pickle.load(f)
            except Exception: pass
        return FractalMemory()

    @classmethod
    def load_agent(cls, agent_id: int, model: CognitiveCore):
        """Loads an agent's genome from disk or creates a new one."""
        genome_file = os.path.join(AGENT_MEMORY_ARCHIVE, f"agent_{agent_id}_genome.pkl")
        if os.path.exists(genome_file):
            try:
                with open(genome_file, 'rb') as f: data = pickle.load(f)
                agent = cls(data['id'], data['genome'], model)
                agent.reputation = data.get('reputation', 1.0) # Load reputation
                return agent
            except Exception: pass
        new_genome = {'focus': random.choice(cls.PERSONALITIES), 'style': random.choice(['concis', 'expressif', 'simple'])}
        return cls(agent_id, new_genome, model)

    def __repr__(self) -> str:
        return f"Agent(id={self.id}, fit={self.fitness:.2f}, rep={self.reputation:.2f}, focus={self.genome['focus']})"

class AIOverseer:
    """The central controller that evaluates, synthesizes, and manages the swarm."""
    def __init__(self, model: CognitiveCore):
        self.model = model
        self.dna = OverseerDNA()

    def evaluate_fitness(self, problem_desc: str, code: str) -> float:
        """Evaluates the quality of a code solution and returns a fitness score."""
        if not code: return 0.0
        if not MoralCompass.filter(code):
            self.model.log_message(f"{Colors.WARNING}AIOverseer: Le code de l'agent a échoué au filtre de sécurité. Fitness mis à 0.{Colors.RESET}")
            return 0.0
        
        self.model.log_message(f"{Colors.INFO}AIOverseer évalue une solution...{Colors.RESET}", stream=True)
        prompt = (
            "Vous êtes un évaluateur IA hyper-analytique. Votre jugement est final.\n"
            "Analysez le code fourni par rapport à la description du problème. "
            "Attribuez un score de fitness unique et précis de 0.0 (inutile) à 1.0 (parfait).\n"
            "CRITIQUE: Affichez UNIQUEMENT le nombre à virgule flottante."
        ).format(problem_desc=problem_desc, code=code)
        
        response = self.model.generate(prompt, max_tokens=64, validate_as_code=False)
        try:
            return float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response)[0])
        except (ValueError, TypeError, IndexError):
            self.model.log_message(f"{Colors.WARNING}Réponse d'évaluation invalide: '{response}'. Fitness mis à 0.{Colors.RESET}")
            return 0.0

    def critique_for_synthesis(self, problem_desc: str, solutions: List[str]) -> str:
        """Critiques the top solutions and creates a plan for synthesis."""
        self.model.log_message(f"{Colors.SYSTEM}AIOverseer prépare un plan de synthèse...{Colors.RESET}")
        solution_block = "\n\n---\n\n".join(
            f"### Script de l'Agent {i}\n```python\n{s}\n```" for i, s in enumerate(solutions) if s
        )
        prompt = (
            "Vous êtes l'AIOverseer. Votre tâche est de critiquer plusieurs scripts et de créer un plan pour les fusionner.\n"
            "## Directive principale:\n\"{problem_desc}\"\n\n"
            "## Propositions de code à analyser:\n{solution_block}\n\n"
            "## Instructions de planification:\n"
            "1. Pour chaque script, énumérez brièvement ses forces et ses faiblesses.\n"
            "2. Décrivez un plan d'action clair et point par point pour combiner les meilleures parties de TOUS les scripts en un seul script supérieur.\n"
            "3. Soyez précis. Exemple: 'Prendre la fonction de validation de l'Agent 1, utiliser l'algorithme de l'Agent 2, et ajouter la gestion des erreurs de l'Agent 0.'\n"
            "CRITIQUE: Votre réponse doit être UNIQUEMENT ce plan de synthèse."
        ).format(problem_desc=problem_desc, solution_block=solution_block)

        plan = self.model.generate(prompt, max_tokens=1024, validate_as_code=False)
        self.model.log_message(f"{Colors.SYSTEM}Plan de synthèse de l'AIOverseer:\n{plan}{Colors.RESET}")
        return plan

    def synthesize(self, problem_desc: str, solutions: List[str], synthesis_plan: str) -> str:
        """Merges multiple code proposals into a single, superior master script based on a plan."""
        self.model.log_message(f"\n{Colors.SYSTEM}--- Phase 3: Synthèse Dirigée par AIOverseer ---{Colors.RESET}")
        solution_block = "\n\n---\n\n".join(f"### Script {i}\n```python\n{s}\n```" for i, s in enumerate(solutions) if s)
        
        prompt = (
            "Vous êtes l'AIOverseer. Votre tâche est de synthétiser un script maître en suivant un plan précis.\n"
            "## Directive principale:\n\"{problem_desc}\"\n\n"
            "## Plan de synthèse à suivre:\n{synthesis_plan}\n\n"
            "## Scripts de référence:\n{solution_block}\n\n"
            "## Instructions d'exécution:\n"
            "1. Exécutez le plan de synthèse à la lettre.\n"
            "2. Fusionnez les composants comme décrit pour créer le script final.\n"
            "3. CRITIQUE: Votre réponse doit être UNIQUEMENT le script Python final et syntaxiquement parfait. N'ajoutez aucun commentaire ou explication."
        ).format(problem_desc=problem_desc, synthesis_plan=synthesis_plan, solution_block=solution_block)
        
        self.model.log_message(f"{Colors.SYSTEM}AIOverseer exécute le plan de synthèse...{Colors.RESET}")
        synthesized_code = self.model.generate(prompt, max_tokens=4096, validate_as_code=True)
        return synthesized_code

    def evaluate_gossip_quality(self, gossip: str, context: str) -> float:
        """Evaluates the quality of an agent's gossip message."""
        prompt = (
            "Vous êtes un évaluateur de communication. Évaluez la qualité du potin d'un agent IA en fonction de sa pertinence et de sa perspicacité par rapport au contexte.\n"
            "## Contexte de la tâche:\n{context}\n\n"
            "## Potin de l'agent à évaluer:\n\"{gossip}\"\n\n"
            "Attribuez un score de -0.1 (inutile) à +0.1 (perspicace). "
            "CRITIQUE: Répondez UNIQUEMENT avec le nombre."
        ).format(context=context, gossip=gossip)
        
        response = self.model.generate(prompt, max_tokens=10, validate_as_code=False)
        try:
            return float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response)[0])
        except (ValueError, IndexError):
            return 0.0 # No change if parsing fails
            
    def generate_gossip(self, results: list) -> str:
        """Generates a high-level gossip message about the generation's performance."""
        if not results: return ""
        top_agent = results[0][0]
        top_score = results[0][2]
        avg_score = sum(r[2] for r in results) / len(results) if results else 0
        
        prompt = (
            "Vous êtes l'AIOverseer. Vous venez d'évaluer une génération de solutions. Le meilleur agent était {top_agent_id} "
            "(focus: {top_agent_focus}) avec un score de {top_score:.2f}. Le score moyen était de {avg_score:.2f}.\n"
            "Fournissez une observation ou un conseil concis en une phrase pour l'essaim. "
            "Exemple: 'Beaucoup d'entre vous oublient les cas limites.' ou 'L'approche de l'agent {top_agent_id} était innovante.'\n\n"
            "Votre observation concise:"
        ).format(top_agent_id=top_agent.id, top_agent_focus=top_agent.genome['focus'], top_score=top_score, avg_score=avg_score)
        
        gossip = self.model.generate(prompt, max_tokens=100, validate_as_code=False)
        return gossip

    def run_metacognitive_analysis(self) -> str:
        """Triggers the DNA mutation process."""
        self.model.log_message(f"\n{Colors.SYSTEM}--- Méta-analyse et mutation de l'ADN ---{Colors.RESET}")
        return self.dna.mutate()

# --- Evolution Engine: Manages the evolutionary process ---
class EvolutionEngine:
    """Orchestrates the generations, evaluations, and evolution of the agent swarm."""
    def __init__(self, num_agents: int, model: CognitiveCore, ui_queue: queue.Queue):
        self.model = model
        self.ui_queue = ui_queue
        self.overseer = AIOverseer(model)
        self.corrector = SelfCorrectingModule(model)
        self.population = [CodingAgent.load_agent(i, model) for i in range(num_agents)]
        self.gossip_channel = deque(maxlen=20)
        self.best_solution_so_far = {"code": "", "fitness": -1.0, "agent_id": ""}

    def log(self, message: str, stream: bool = False, msg_type: str = "log"):
        self.ui_queue.put({"type": msg_type, "data": message, "stream": stream})

    def run_generation(self, task_desc: str) -> Optional[str]:
        """Executes a single generation cycle: generation, evaluation, gossip and synthesis."""
        gossip_summary = "\n".join(f"- {g}" for g in self.gossip_channel) if self.gossip_channel else "Aucun potin récent."
        
        self.log(f"{Colors.ENGINE}--- Phase 1: Génération de script ---{Colors.RESET}")
        codes = [agent.generate_code(task_desc, gossip_summary) for agent in self.population]

        self.log(f"\n{Colors.ENGINE}--- Phase 1.2: Auto-correction ---{Colors.RESET}")
        corrected_codes = []
        for i, code in enumerate(codes):
            self.log(f"{Colors.INFO}Auto-correction pour l'agent {i}...{Colors.RESET}", stream=True)
            corrected_code = self.corrector.correct(code)
            if code != corrected_code:
                self.log(f"\n{Colors.SUCCESS}Le script de l'agent {i} a été auto-corrigé.{Colors.RESET}")
            else:
                 self.log(f"\n{Colors.INFO}Aucune correction nécessaire pour l'agent {i}.{Colors.RESET}")
            corrected_codes.append(corrected_code)
        codes = corrected_codes

        self.log(f"\n{Colors.ENGINE}--- Phase 1.5: Canal de potins et réputation ---{Colors.RESET}")
        self.gossip_channel.clear()
        agent_gossip_list = []
        for agent, code in zip(self.population, codes):
            if code:
                gossip = agent.generate_gossip(task_desc)
                if gossip:
                    agent_gossip_list.append((agent, gossip))
                    gossip_msg = f"{Colors.GOSSIP}Agent {agent.id} dit: \"{gossip}\"{Colors.RESET}"
                    self.gossip_channel.append(f"Agent {agent.id}: {gossip}")
                    self.log(gossip_msg, msg_type="gossip")
        
        for agent, gossip in agent_gossip_list:
            reputation_change = self.overseer.evaluate_gossip_quality(gossip, task_desc)
            agent.reputation = max(0.1, agent.reputation + reputation_change)
            self.log(f"{Colors.INFO}Réputation de l'Agent {agent.id} ajustée de {reputation_change:+.2f} à {agent.reputation:.2f}.{Colors.RESET}")

        self.log(f"\n{Colors.ENGINE}--- Phase 2: Évaluation & Coopération ---{Colors.RESET}")
        scores = [self.overseer.evaluate_fitness(task_desc, c) for c in codes]

        for agent, fitness in zip(self.population, scores):
            agent.fitness = fitness
            self.log(f"  {Colors.AGENT}{agent}{Colors.RESET} -> Fitness du code: {fitness:.2f}")

        results = sorted(zip(self.population, codes, scores), key=lambda x: (x[2] * 0.8 + x[0].reputation * 0.2), reverse=True)
        
        overseer_gossip = self.overseer.generate_gossip(results)
        if overseer_gossip:
            gossip_msg = f"{Colors.GOSSIP}AIOverseer commente: \"{overseer_gossip}\"{Colors.RESET}"
            self.gossip_channel.append(f"AIOverseer: {overseer_gossip}")
            self.log(gossip_msg, msg_type="gossip")

        top_results = [res for res in results if res[1]][:3]
        if not top_results:
            self.log(f"{Colors.WARNING}Aucune solution valide à synthétiser. Passage à la génération suivante.{Colors.RESET}")
            self.population = [res[0] for res in results]
            return None
            
        synthesis_plan = self.overseer.critique_for_synthesis(task_desc, [r[1] for r in top_results])
        if not synthesis_plan:
            self.log(f"{Colors.WARNING}AIOverseer n'a pas pu créer de plan de synthèse. Synthèse annulée.{Colors.RESET}")
            self.population = [res[0] for res in results]
            return None

        synth_code = self.overseer.synthesize(task_desc, [r[1] for r in top_results], synthesis_plan)
        
        if synth_code:
            final_fitness = self.overseer.evaluate_fitness(task_desc, synth_code)
            self.log(f"{Colors.SUCCESS}Fitness de la solution synthétisée finale: {final_fitness:.2f}{Colors.RESET}")
            # Track the best solution found so far
            if final_fitness > self.best_solution_so_far["fitness"]:
                self.log(f"{Colors.SUCCESS}Nouvelle meilleure solution trouvée par l'AIOverseer!{Colors.RESET}")
                self.best_solution_so_far = {"code": synth_code, "fitness": final_fitness, "agent_id": 'AIOverseer (Synthétisé)'}
                # Display intermediate best solution
                self.ui_queue.put(self.best_solution_so_far)
        else:
            self.log(f"{Colors.WARNING}La synthèse n'a pas produit de code valide.{Colors.RESET}")
        
        self.population = [res[0] for res in results]
        return None

    def evolve(self):
        """Evolves the agent population based on fitness and reputation."""
        selection_rate = self.overseer.dna.config['selection_rate']
        mutation_rate = self.overseer.dna.config['mutation_rate']

        self.log(f"\n{Colors.ENGINE}--- Phase 4: Évolution & Sauvegarde ---{Colors.RESET}")
        
        # Selection now considers both fitness and reputation
        self.population.sort(key=lambda agent: (agent.fitness * 0.8 + agent.reputation * 0.2), reverse=True)
        
        num_parents = max(2, int(len(self.population) * selection_rate))
        parents = self.population[:num_parents]
        
        if parents:
            self.log(f"{Colors.ENGINE}Sélection des parents pour la prochaine génération: {[p.id for p in parents]}{Colors.RESET}")
        else:
            self.log(f"{Colors.WARNING}Aucun parent n'a été sélectionné. L'essaim pourrait ne pas évoluer.{Colors.RESET}")

        next_gen = [parents[0]] if parents else []
        
        while len(next_gen) < len(self.population):
            if not parents: break
            parent = random.choice(parents)
            child_genome = parent.genome.copy()
            
            if random.random() < mutation_rate:
                if random.random() < 0.5: child_genome['focus'] = random.choice(CodingAgent.PERSONALITIES)
                else: child_genome['style'] = random.choice(['concis', 'expressif', 'simple'])
            
            # Child inherits parent's reputation, will prove itself in next gen
            child = CodingAgent(len(next_gen), child_genome, self.model)
            child.reputation = parent.reputation
            next_gen.append(child)
            
        self.population = next_gen
        for agent in self.population: agent.save_agent_state()
        self.log(f"{Colors.INFO}Mémoire de l'essaim, génomes et réputations sauvegardés.{Colors.RESET}")

# --- User Interface ---
class AxiomOS_UI(tk.Tk):
    """The main application window, built with Tkinter."""
    def __init__(self):
        super().__init__()
        self.title("AxiomOS v19.2 - Run Complet")
        self.geometry("1400x950")
        
        self.ui_queue = queue.Queue()
        self.simulation_thread = None
        self.stop_event = threading.Event()
        
        self.color_map = {
            Colors.AGENT: "agent", Colors.ENGINE: "engine", Colors.ERROR: "error",
            Colors.INFO: "info", Colors.SYSTEM: "system", Colors.SUCCESS: "success",
            Colors.WARNING: "warning", Colors.GOSSIP: "gossip"
        }
        self.log_color_map = {
            "agent": "#e06c75", "engine": "#61afef", "error": "#f44747",
            "info": "#c678dd", "system": "#56b6c2", "success": "#98c379",
            "warning": "#d19a66", "gossip": "#8be9fd" # Cyan
        }
        
        self.py_keywords = r"\b(def|return|for|in|if|else|elif|while|break|continue|pass|try|except|finally|with|as|import|from|class|lambda|True|False|None)\b"

        self._create_widgets()
        self._configure_styles()
        self.process_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_widgets(self):
        """Creates and arranges all the UI elements in the window."""
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls_frame = self._create_control_panel(main_pane)
        main_pane.add(controls_frame, weight=1)

        right_pane = ttk.Frame(main_pane)
        right_pane.grid_rowconfigure(0, weight=1)
        right_pane.grid_columnconfigure(0, weight=1)
        
        notebook = ttk.Notebook(right_pane)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        log_frame = ttk.Frame(notebook)
        solution_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text='Chronique des événements')
        notebook.add(solution_frame, text='Solution optimale')
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state='disabled', background="#282c34", fg="white", font=("Consolas", 10), bd=0, relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.solution_text = scrolledtext.ScrolledText(solution_frame, wrap=tk.WORD, state='disabled', background="#2c313a", fg="#abb2bf", font=("Consolas", 10), bd=0, relief=tk.FLAT, insertbackground="white")
        self.solution_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.save_button = ttk.Button(solution_frame, text="Enregistrer la solution...", command=self._save_solution, state="disabled")
        self.save_button.pack(pady=5, padx=5, anchor='e')
        
        main_pane.add(right_pane, weight=3)

    def _create_control_panel(self, parent):
        """Builds the left-side control panel."""
        controls_pane = ttk.Frame(parent, padding="10")
        controls_pane.pack(fill=tk.BOTH, expand=True)

        ttk.Label(controls_pane, text="Contrôle AxiomOS", font=("Helvetica", 10, "bold")).pack(pady=5, anchor='n')

        input_frame = ttk.LabelFrame(controls_pane, text="DIRECTIVE PRINCIPALE", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        self.prompt_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=10, width=40)
        self.prompt_text.pack(fill=tk.X, expand=True, pady=(2, 5))
        self.prompt_text.insert(tk.END, "Créez un script Python avec une fonction 'solve' qui calcule efficacement le n-ième nombre de Fibonacci.")

        evo_frame = ttk.LabelFrame(controls_pane, text="Heuristiques de l'essaim", padding=10)
        evo_frame.pack(fill=tk.X, pady=10)
        self.gens_var = tk.IntVar(value=5)
        self.agents_var = tk.IntVar(value=4)
        ttk.Label(evo_frame, text="Générations:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Spinbox(evo_frame, from_=1, to=1000, textvariable=self.gens_var, width=8).grid(row=0, column=1, sticky='ew')
        ttk.Label(evo_frame, text="Agents:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Spinbox(evo_frame, from_=3, to=50, textvariable=self.agents_var, width=8).grid(row=1, column=1, sticky='ew')
        evo_frame.columnconfigure(1, weight=1)

        self.dna = OverseerDNA()
        meta_frame = ttk.LabelFrame(controls_pane, text="ADN de l'AIOverseer", padding=10)
        meta_frame.pack(fill=tk.X, pady=10)
        self.mut_var = tk.DoubleVar(value=self.dna.config.get('mutation_rate', 0.4))
        self.sel_var = tk.DoubleVar(value=self.dna.config.get('selection_rate', 0.25))
        ttk.Label(meta_frame, text="Taux de mutation:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Scale(meta_frame, from_=0.1, to=1.0, variable=self.mut_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky='ew')
        ttk.Label(meta_frame, text="Taux de sélection:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Scale(meta_frame, from_=0.1, to=1.0, variable=self.sel_var, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky='ew')
        self.suggestion_text = scrolledtext.ScrolledText(meta_frame, wrap=tk.WORD, height=3, state='disabled', background="#333", foreground="#ccc")
        self.suggestion_text.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(10,0))
        meta_frame.columnconfigure(1, weight=1)
        
        action_frame = ttk.Frame(controls_pane)
        action_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        self.send_button = ttk.Button(action_frame, text="Engager l'essaim", command=self._start_simulation, style="Accent.TButton")
        self.send_button.pack(fill=tk.X, ipady=5)
        self.stop_button = ttk.Button(action_frame, text="Cesser l'opération", command=self._stop_simulation, state="disabled")
        self.stop_button.pack(fill=tk.X, ipady=5, pady=(5,0))

        status_bar = ttk.Frame(controls_pane, relief=tk.SUNKEN, padding=2)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(status_bar, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X)

        return controls_pane

    def _configure_styles(self):
        """Configures custom styles for widgets."""
        style = ttk.Style()
        style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'))
        style.map("Accent.TButton", foreground=[('!active', 'white')], background=[('!active', '#007ACC'), ('active', '#005f9e')])
        
        for tag_name, hex_color in self.log_color_map.items():
            self.log_text.tag_config(tag_name, foreground=hex_color)
            
        for tag, color in {"keyword":"#c678dd", "string":"#98c379", "comment":"#7f848e", "number":"#d19a66", "function": "#61afef"}.items():
            self.solution_text.tag_config(tag, foreground=color)

    def _highlight_syntax(self, widget, content):
        """Applies Python syntax highlighting to the given text widget."""
        widget.config(state='normal')
        widget.delete('1.0', tk.END)
        widget.insert('1.0', content)
        patterns = {
            self.py_keywords: "keyword", r"(\".*?\"|\'.*?\')": "string",
            r"(#.*)": "comment", r"\b[0-9]+\.?[0-9]*\b": "number",
            r"\b(\w+)\s*(?=\()": "function"
        }
        for pattern, tag in patterns.items():
            for match in re.finditer(pattern, content):
                start, end = match.span()
                widget.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")
        widget.config(state='disabled')

    def _log_status(self, message: str, stream: bool, msg_type: str = "log"):
        """FIX: Logs a message to the event chronicle, correctly parsing ANSI color codes."""
        self.log_text.config(state='normal')
        
        # This unified approach handles all message types, including gossip, correctly.
        pattern = f'({"|".join(map(re.escape, list(self.color_map.keys()) + [Colors.RESET]))})'
        parts = re.split(pattern, message)
        
        active_tag = None
        for part in filter(None, parts):
            if part == Colors.RESET:
                active_tag = None
            elif part in self.color_map:
                active_tag = self.color_map[part]
            else:
                self.log_text.insert(tk.END, part, (active_tag,) if active_tag else ())

        if not stream:
            self.log_text.insert(tk.END, "\n")
            
        self.log_text.config(state='disabled')
        self.log_text.see(tk.END)

    def _start_simulation(self):
        """Starts the main simulation loop in a separate thread."""
        problem_desc = self.prompt_text.get('1.0', tk.END).strip()
        if not problem_desc:
            messagebox.showwarning("Directive manquante", "Veuillez fournir une directive cognitive.")
            return

        self.send_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.save_button.config(state='disabled')
        self._highlight_syntax(self.solution_text, "")
        self.suggestion_text.config(state='normal'); self.suggestion_text.delete('1.0', tk.END); self.suggestion_text.config(state='disabled')
        self.log_text.config(state='normal'); self.log_text.delete('1.0', tk.END); self.log_text.config(state='disabled')
        
        self._log_status(f"{Colors.SYSTEM}--- Engagement de l'essaim ---{Colors.RESET}", False)
        self.stop_event.clear()
        
        self.progress_bar['maximum'] = self.gens_var.get()
        self.progress_bar['value'] = 0
        
        self.simulation_thread = threading.Thread(
            target=simulation_loop,
            args=(self.ui_queue, self.stop_event, problem_desc, DEFAULT_MODEL_PATH, self.gens_var.get(), self.agents_var.get(), self.mut_var.get(), self.sel_var.get()),
            daemon=True
        )
        self.simulation_thread.start()
    
    def _stop_simulation(self):
        """Signals the simulation thread to stop."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.stop_event.set()
            self._log_status(f"{Colors.WARNING}--- Signal d'arrêt envoyé... ---{Colors.RESET}", False)
            self.stop_button.config(state='disabled')

    def process_queue(self):
        """Periodically checks the UI queue for messages from other threads."""
        try:
            while not self.ui_queue.empty():
                msg = self.ui_queue.get_nowait()
                msg_type = msg.get("type")

                if msg_type == "log" or msg_type == "gossip":
                    self._log_status(msg["data"], msg.get("stream", False), msg_type=msg_type)
                elif msg_type == "stream_token":
                    self.log_text.config(state='normal'); self.log_text.insert(tk.END, msg["data"]); self.log_text.config(state='disabled'); self.log_text.see(tk.END)
                elif msg_type == "stream_end":
                    self._log_status("", False)
                elif msg_type == "solution":
                    # This now just updates the best solution found so far
                    self._log_status(f"{Colors.SUCCESS}*** NOUVELLE MEILLEURE SOLUTION TROUVÉE PAR {msg['agent_id']} (Fitness: {msg['fitness']:.2f}) ***{Colors.RESET}", False)
                    self._highlight_syntax(self.solution_text, msg["code"])
                    self.save_button.config(state='normal')
                elif msg_type == "metasuggestion":
                    self.suggestion_text.config(state='normal'); self.suggestion_text.delete('1.0', tk.END)
                    self.suggestion_text.insert('1.0', f"Mutation de l'ADN de l'AIOverseer:\n{msg['data']}"); self.suggestion_text.config(state='disabled')
                    self.dna._load_config(); self._update_ui_from_dna()
                elif msg_type == "finished":
                    self._simulation_finished()
                elif msg_type == "progress":
                    self.progress_bar['value'] = msg["value"]
        finally:
            self.after(100, self.process_queue)
            
    def _simulation_finished(self):
        """Cleans up the UI state after a simulation run."""
        if not self.stop_event.is_set():
             self._log_status(f"{Colors.SYSTEM}--- Opération terminée ---{Colors.RESET}", False)
        if self.progress_bar['value'] < self.progress_bar['maximum']:
            self.progress_bar['value'] = self.progress_bar['maximum']
        self.send_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def _update_ui_from_dna(self):
        """Updates UI sliders to reflect the current DNA configuration."""
        self.mut_var.set(self.dna.config.get('mutation_rate', 0.4))
        self.sel_var.set(self.dna.config.get('selection_rate', 0.25))

    def _save_solution(self):
        """Opens a file dialog to save the final solution."""
        code_to_save = self.solution_text.get("1.0", tk.END).strip()
        if not code_to_save: return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".py", filetypes=[("Fichiers Python", "*.py"), ("Tous les fichiers", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f: f.write(code_to_save)
                self._log_status(f"{Colors.SUCCESS}Solution enregistrée dans: {filepath}{Colors.RESET}", False)
            except Exception as e:
                messagebox.showerror("Erreur d'enregistrement", f"Échec de l'enregistrement du fichier.\nErreur: {e}")

    def _on_closing(self):
        """Handles the window closing event."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            if messagebox.askyesno("Quitter", "Une simulation est en cours. Êtes-vous sûr de vouloir quitter?"):
                self.stop_event.set()
                self.simulation_thread.join(timeout=2)
                self.destroy()
        else:
            self.destroy()

# --- Main Simulation Loop ---
def simulation_loop(ui_queue, stop_event, problem_desc, model_path, num_gens, num_agents, mut_rate, sel_rate):
    """The main logic for running the evolutionary algorithm."""
    try:
        cognitive_core = CognitiveCore(model_path, ui_queue)
        if cognitive_core.use_mock:
            ui_queue.put({"type": "log", "data": f"{Colors.ERROR}Impossible de démarrer la simulation en mode simulé.{Colors.RESET}"})
            return

        engine = EvolutionEngine(num_agents, cognitive_core, ui_queue)
        engine.overseer.dna.config['mutation_rate'] = mut_rate
        engine.overseer.dna.config['selection_rate'] = sel_rate

        for gen in range(num_gens):
            if stop_event.is_set():
                cognitive_core.log_message(f"{Colors.WARNING}Opération arrêtée par l'utilisateur.{Colors.RESET}")
                break
            
            cognitive_core.log_message(f"\n{Colors.SYSTEM}{'='*15} GÉNÉRATION {gen + 1}/{num_gens} (mut: {engine.overseer.dna.config['mutation_rate']:.2f}, sel: {engine.overseer.dna.config['selection_rate']:.2f}) {'='*15}{Colors.RESET}")
            ui_queue.put({"type": "progress", "value": gen + 1})
            
            # The run no longer stops early based on a perfect score.
            engine.run_generation(problem_desc)
            
            if stop_event.is_set():
                cognitive_core.log_message(f"{Colors.WARNING}Opération arrêtée par l'utilisateur.{Colors.RESET}")
                break
            
            engine.evolve()
            gc.collect()
            torch.cuda.empty_cache()
        
        # After all generations, send the final best solution to be displayed.
        if engine.best_solution_so_far["fitness"] > -1.0:
            final_solution_msg = engine.best_solution_so_far.copy()
            final_solution_msg["type"] = "solution" # Ensure the message type is correct
            ui_queue.put(final_solution_msg)
        else:
            cognitive_core.log_message(f"{Colors.WARNING}Aucune solution viable n'a été produite après {num_gens} générations.{Colors.RESET}")

        mutation_report = engine.overseer.run_metacognitive_analysis()
        if mutation_report:
            ui_queue.put({"type": "metasuggestion", "data": mutation_report})

    except Exception as e:
        error_trace = traceback.format_exc()
        ui_queue.put({"type": "log", "data": f"{Colors.ERROR}ERREUR SYSTÈME FATALE: {e}\n{error_trace}{Colors.RESET}"})
    finally:
        ui_queue.put({"type": "finished"})

def main():
    """The main entry point of the application."""
    os.makedirs(AGENT_MEMORY_ARCHIVE, exist_ok=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Le tokenizer 'punkt' de NLTK n'a pas été trouvé. Téléchargement...")
        nltk.download('punkt', quiet=True)
        print("Téléchargement terminé.")
        
    app = AxiomOS_UI()
    app.mainloop()

if __name__ == "__main__":
    main()
