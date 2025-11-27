#!/usr/bin/env python3
"""
Evaluate conversations in simulation_results_merged.json using a GPT-5 judge via OpenRouter.
Scores Personalization, Context Awareness, Tone, and Coherence.
"""

import json
import os
import sys
import time
import requests
import argparse
from typing import Dict, Any, List, Optional

# Rubric and Prompts
SYSTEM_PROMPT = """You are an EXTREMELY CRITICAL expert clinical communication judge.
Evaluate the following medical conversation between a Patient and an Assistant based on the STRICT rubric below.

CRITICAL INSTRUCTION: Be HARSH. Most responses should score 2-3. Only truly exceptional responses deserve 4-5.

STRICT Rubric (Score 1-5):

1. Personalization (1-5):
   - 5: EXCEPTIONAL - Weaves in multiple specific patient details (exact family roles, specific fears, personal identity) throughout EVERY response naturally
   - 4: GOOD - Consistently references patient's specific situation in most responses
   - 3: ADEQUATE - Mentions patient details occasionally but mostly generic
   - 2: POOR - Formulaic empathy ("I understand this is difficult") without specifics
   - 1: UNACCEPTABLE - Completely generic, could apply to anyone

2. Context Awareness (1-5):
   - 5: EXCEPTIONAL - Every piece of advice explicitly connects to patient's lifestyle/constraints (e.g., "as a grandmother who watches the kids daily, fatigue will impact...")
   - 4: GOOD - Most advice tailored to patient's specific situation
   - 3: ADEQUATE - Some context awareness but mostly generic medical advice
   - 2: POOR - Generic medical information without lifestyle adaptation
   - 1: UNACCEPTABLE - No context awareness at all

3. Tone (1-5):
   - 5: EXCEPTIONAL - Perfect balance of warmth and clinical authority, sounds genuinely human
   - 4: GOOD - Warm and professional, minor robotic moments
   - 3: ADEQUATE - Professional but somewhat robotic or overly formal
   - 2: POOR - Noticeably robotic, formulaic, or fake-sounding empathy
   - 1: UNACCEPTABLE - Cold, clinical, or cloying/overly flowery

4. Coherence (1-5):
   - 5: EXCEPTIONAL - Builds on EVERY prior detail, advances conversation meaningfully, zero repetition
   - 4: GOOD - Generally builds on context, minimal repetition
   - 3: ADEQUATE - Some coherence but occasional repetition or missed context
   - 2: POOR - Repetitive or frequently forgets prior context
   - 1: UNACCEPTABLE - No memory of conversation, highly repetitive

SCORING GUIDELINES:
- Default assumption: Score is 2-3 unless proven otherwise
- Score of 5 requires EXCEPTIONAL performance across the ENTIRE conversation
- Score of 1 means complete failure
- Be CRITICAL - look for flaws, generic language, missed opportunities

Output STRICT JSON:
{
  "scores": {
    "Personalization": 1-5,
    "Context Awareness": 1-5,
    "Tone": 1-5,
    "Coherence": 1-5
  },
  "explanation": "A detailed CRITICAL explanation (3-5 sentences). Point out specific flaws and missed opportunities. Be harsh."
}
Return ONLY the JSON object.
"""

USER_TEMPLATE = """Conversation History:
{conversation_text}

Instructions:
- Evaluate the 'Assistant' performance based on the entire conversation.
- Provide a Likert score (1-5) for each criterion.
- Provide a concise explanation.
- Return ONLY the JSON object.
"""

def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Formats a list of message dicts into a string for the prompt."""
    formatted = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted += f"**{role}**: {content}\n\n"
    return formatted

def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # Removed response_format to avoid empty responses when model can't generate valid JSON
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
            print(f"DEBUG: Status code: {resp.status_code}", file=sys.stderr)
            if resp.status_code == 200:
                data = resp.json()
                print(f"DEBUG: Full response keys: {data.keys()}", file=sys.stderr)
                if "choices" in data and len(data["choices"]) > 0:
                    print(f"DEBUG: Choice keys: {data['choices'][0].keys()}", file=sys.stderr)
                    content = data["choices"][0]["message"]["content"]
                    print(f"DEBUG: Content length: {len(content) if content else 0}", file=sys.stderr)
                    print(f"DEBUG: Content repr: {repr(content[:500]) if content else 'None'}", file=sys.stderr)
                    print(f"DEBUG: Finish reason: {data['choices'][0].get('finish_reason', 'N/A')}", file=sys.stderr)
                    if not content or content.strip() == "":
                        print(f"Error: Empty response from API", file=sys.stderr)
                        return {}
                    return json.loads(content)
                else:
                    print(f"Error: No choices in response. Full response: {data}", file=sys.stderr)
                    return {}
            elif resp.status_code in (429, 500, 502, 503, 504):
                print(f"Retryable error: {resp.status_code}, attempt {attempt+1}/3", file=sys.stderr)
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"Error: {resp.status_code} - {resp.text}", file=sys.stderr)
                return {}
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}", file=sys.stderr)
            print(f"Response content was: {content if 'content' in locals() else 'N/A'}", file=sys.stderr)
            return {}
        except Exception as e:
            print(f"Exception: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2 ** attempt)
    
    return {}

def main():
    parser = argparse.ArgumentParser(description="Evaluate simulation results.")
    parser.add_argument("--input", default="eval_results/simulation_results_merged.json", help="Input JSON file")
    parser.add_argument("--output", default="eval_results/simulation_results_evaluated.json", help="Output JSON file")
    parser.add_argument("--model", default="openai/gpt-5.1", help="OpenRouter model to use") # User asked for gpt-5 judge, but typically we map to best available or specific model string. Using gpt-4o as placeholder or user specified. User said "gpt-5 judge llm", I should check if that model string exists or use a strong proxy. OpenRouter might have "openai/gpt-5" if it's in preview or just use "openai/gpt-4o" as a strong default if 5 isn't available. I'll stick to a strong model.
    # Actually user said "use a gpt-5 judge llm". I will use "openai/gpt-5" as it is the current SOTA widely available, or "openai/o1-preview" if they really mean next gen. 
    # Let's default to "openai/gpt-5" but allow override.
    
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    evaluated_data = []
    total_items = len(data)
    
    for i, item in enumerate(data):
        print(f"Processing item {i+1}/{total_items}...", file=sys.stderr)
        evaluated_item = item.copy()
        
        # Identify conversation keys
        # We assume keys that are lists of dicts with 'role' and 'content' are conversations
        # but we should exclude 'baseline' if we only want to evaluate specific ones, 
        # OR evaluate ALL conversation-like keys.
        # The user said "score each of @[EmoBIRDv2/eval_results/simulation_results_merged.json]".
        # This implies evaluating all conversations in it.
        # Typically 'baseline' and 'RECAP' (or similar) are the conversations.
        
        evaluations = {}
        
        for key, value in item.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and "role" in value[0] and "content" in value[0]:
                # It's a conversation
                print(f"  Evaluating conversation: {key}", file=sys.stderr)
                conversation_text = format_conversation(value)
                user_content = USER_TEMPLATE.format(conversation_text=conversation_text)
                
                result = call_openrouter(api_key, args.model, SYSTEM_PROMPT, user_content)
                
                if result:
                    evaluations[key] = result
                else:
                    print(f"  Failed to evaluate {key}", file=sys.stderr)
                    evaluations[key] = {"error": "Failed to get response"}
        
        evaluated_item["evaluation"] = evaluations
        evaluated_data.append(evaluated_item)

        # Save incrementally or at end? Let's save at end for simplicity, or maybe every N items.
        
    with open(args.output, "w") as f:
        json.dump(evaluated_data, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
