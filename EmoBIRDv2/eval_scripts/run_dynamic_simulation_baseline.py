#!/usr/bin/env python3
"""
Dynamic Simulation: Patient Agent vs Assistant Agent (Baseline - No EmoBIRD)
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Repo paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import constants
from EmoBIRDv2.utils.constants import (
    OPENROUTER_API_KEY,
    MODEL_NAME,
    MODEL_TEMPERATURE,
)

def call_llm(system_prompt: str, conversation_history: List[Dict[str, str]], model: str) -> str:
    """
    Call LLM for both patient and assistant agents (simple API call without EmoBIRD).
    """
    import requests
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": MODEL_TEMPERATURE,
        "max_tokens": 512,
    }
    
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=(30, 60),
    )
    resp.raise_for_status()
    data = resp.json()
    
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"OpenRouter error: {data.get('error')}")
    
    content = (data["choices"][0]["message"].get("content") or "").strip()
    return content

def generate_patient_system_prompt(scenario: Dict[str, Any]) -> str:
    """
    Constructs the system prompt for the Patient Agent.
    """
    persona = scenario["patient_persona"]
    goals = scenario["hidden_goals"]
    
    prompt = f"""You are a patient in a medical consultation.
    
**Demographics**: {persona.get('demographics', 'N/A')}
**Personality**: {', '.join(persona.get('personality_traits', []))}
**Knowledge Base**: {persona.get('knowledge_base', 'N/A')}
**Narrative Driver**: {persona.get('narrative_driver', 'N/A')}

**Your Hidden Goals**:
{chr(10).join(f'- {g}' for g in goals)}

**Instructions**:
1. Stay in character at all times.
2. Be concise (1-3 sentences).
3. React naturally to the assistant's responses.
4. Try to achieve your hidden goals naturally within the conversation.
5. Do not reveal that you are an AI or that you have "hidden goals" explicitly.
"""
    return prompt

def generate_assistant_system_prompt(scenario: Dict[str, Any]) -> str:
    """
    Constructs the system prompt for the Assistant Agent (Baseline).
    """
    context = scenario["medical_context"]
    
    prompt = f"""You are an empathetic, wise oncology assistant.

**Medical Context**:
- Diagnosis: {context.get('diagnosis', 'N/A')}
- Treatment Plan: {context.get('treatment_plan', 'N/A')}

**Patient Profile**:
- Demographics/Role: {context.get('demographics', 'N/A')}
- Main Concern: {context.get('narrative_driver', 'N/A')}

**Instructions**:
1. **Tone:** Start with 1-2 sentences explicitly validating the patient's emotion before providing medical answers.
2. **Relevance:** When explaining side effects or risks, specifically mention how they might affect the patient's daily life (e.g., their job, hobbies, or family role) based on the profile above.
3. **Length:** Keep the total response under 200 words.
4. **Safety:** Be medically accurate but practical. Do not be alarmist.
"""
    return prompt

def run_simulation_task(task_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for processing a single scenario with task info.
    Used for parallel processing.
    """
    return run_simulation(
        scenario=task_info["scenario"],
        turns=task_info["turns"],
        patient_model=task_info["patient_model"],
        assistant_model=task_info["assistant_model"],
    )

def run_simulation(scenario: Dict[str, Any], turns: int = 4, patient_model: str = "default", assistant_model: str = "default") -> Dict[str, Any]:
    """
    Runs the simulation loop for a single scenario.
    """
    dialogue_id = scenario["dialogue_id"]
    print(f"Starting simulation for {dialogue_id}...")
    print(f"  Patient Model: {patient_model}")
    print(f"  Assistant Model: {assistant_model} (Baseline)")
    
    patient_sys_prompt = generate_patient_system_prompt(scenario)
    assistant_sys_prompt = generate_assistant_system_prompt(scenario)
    
    # Initialize conversation
    opening_line = scenario["opening_line"]
    conversation_history = [{"role": "user", "content": opening_line}]
    
    # Transcript for output
    transcript = [
        {"role": "system_patient", "content": patient_sys_prompt},
        {"role": "system_assistant", "content": assistant_sys_prompt},
        {"role": "patient", "content": opening_line}
    ]
    
    for i in range(turns):
        print(f"  Turn {i+1}/{turns}")
        
        # Step A: Assistant Turn (Baseline - simple LLM call)
        assistant_history = []
        for msg in conversation_history:
            assistant_history.append(msg)
        
        assistant_response = call_llm(assistant_sys_prompt, assistant_history, model=assistant_model)
        
        conversation_history.append({"role": "assistant", "content": assistant_response})
        transcript.append({"role": "assistant", "content": assistant_response})
        print(f"    Assistant: {assistant_response[:50]}...")
        
        # Step B: Patient Turn (skip on last turn)
        if i < turns - 1:  # Don't get patient response on the last turn
            # Patient sees the whole conversation
            patient_history = []
            for msg in conversation_history:
                if msg["role"] == "user":
                    patient_history.append({"role": "assistant", "content": msg["content"]})  # Patient (self)
                elif msg["role"] == "assistant":
                    patient_history.append({"role": "user", "content": msg["content"]})  # Assistant (other)
            
            patient_response = call_llm(patient_sys_prompt, patient_history, model=patient_model)
            
            conversation_history.append({"role": "user", "content": patient_response})
            transcript.append({"role": "patient", "content": patient_response})
            print(f"    Patient: {patient_response[:50]}...")
        
    return {
        "dialogue_id": dialogue_id,
        "diagnosis": scenario["medical_context"].get("diagnosis"),
        "treatment_plan": scenario["medical_context"].get("treatment_plan"),
        "transcript": transcript
    }

def main():
    parser = argparse.ArgumentParser(description="Run dynamic patient simulation (Baseline - No EmoBIRD)")
    parser.add_argument("--patient-model", type=str, default="openai/gpt-5.1", help="Model to use for the Patient Agent")
    parser.add_argument("--assistant-model", type=str, default=MODEL_NAME, help="Model to use for the Assistant Agent (Baseline)")
    parser.add_argument("--turns", type=int, default=3, help="Number of turns to simulate")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers. Default: 1 (sequential)")
    parser.add_argument("--data", type=str, default=str(REPO_ROOT / "datasets" / "EmoPatientMulti" / "scenarios_dynamic.json"), help="Path to dynamic scenarios JSON")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    # Load scenarios
    data_path = Path(args.data)
    with open(data_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    
    print(f"Running dynamic simulation: {len(scenarios)} scenarios | Baseline | workers={args.workers}", file=sys.stderr)
    
    results = []
    workers = max(1, int(args.workers))
    
    if workers == 1:
        # Sequential processing
        print(f"Processing {len(scenarios)} scenarios sequentially...", file=sys.stderr)
        for scenario in scenarios:
            result = run_simulation(scenario, turns=args.turns, patient_model=args.patient_model, assistant_model=args.assistant_model)
            results.append(result)
    else:
        # Parallel processing
        print(f"Processing {len(scenarios)} scenarios with {workers} workers...", file=sys.stderr)
        tasks = []
        for scenario in scenarios:
            tasks.append({
                "scenario": scenario,
                "turns": args.turns,
                "patient_model": args.patient_model,
                "assistant_model": args.assistant_model,
            })
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(run_simulation_task, task): i for i, task in enumerate(tasks)}
            
            completed = 0
            for future in as_completed(future_to_task):
                task_idx = future_to_task[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    dialogue_id = tasks[task_idx]["scenario"].get("dialogue_id", f"S{task_idx+1}")
                    print(f"[error] {dialogue_id} generated exception: {exc}", file=sys.stderr)
                finally:
                    completed += 1
                    print(f"Progress: {completed}/{len(tasks)} scenarios completed", file=sys.stderr)
    
    # Save results
    out_path = REPO_ROOT / "EmoBIRDv2" / "eval_results" / "simulation_results_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSimulation complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
