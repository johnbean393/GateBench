import argparse
import json
import os
import sys
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Ensure we can import modules from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from llm import LLM
    from expression_evaluator import ExpressionEvaluator
except ImportError:
    # Fallback for when running from root directory
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from llm import LLM
    from expression_evaluator import ExpressionEvaluator

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SVGBench: A challenging LLM benchmark that tests knowledge, coding, physical\nreasoning capabilities of LLMs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--model", required=True, help='The model to test. Test multiple models by separating\nthem with a semicolon. (e.g.\n"google/gemini-2.5-flash;qwen/qwen3-30b-a3b")')
    parser.add_argument("--endpoint", default="https://openrouter.ai/api/v1", help="The OpenAI compatible endpoint to test with (default:\nhttps://openrouter.ai/api/v1)")
    parser.add_argument("--open-router-endpoint", default="https://openrouter.ai/api/v1", help="The OpenRouter endpoint to use (default:\nhttps://openrouter.ai/api/v1)")
    parser.add_argument("--api-key", help="Your API key for the endpoint")
    parser.add_argument("--open-router-api-key", help="Your OpenRouter API key")
    parser.add_argument("--reasoning-effort", choices=["high", "medium", "low"], help="Reasoning effort level for models that support it\n(high, medium, low)")
    parser.add_argument("--reasoning-max-tokens", type=int, help="Maximum number of tokens to use for reasoning\n(Anthropic-style models)")
    parser.add_argument("--max-output-tokens", type=int, help="Maximum number of output tokens for the response")
    parser.add_argument("--no-auto-update-leaderboard", action="store_true", help="Disable automatic leaderboard updates in README.md")

    return parser.parse_args()

def sanitize_filename(name):
    return re.sub(r'[^\w\-_]', '_', name)

def extract_expression(response_text):
    # Look for code blocks
    # Matches ``` followed by optional language identifier, then content, then ```
    code_block_pattern = r"```(?:\w+)?\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, response_text, re.DOTALL)
    if matches:
        # Return the last match as it's often the final answer, or the first? 
        # The example shows just one. Let's take the first one that looks non-empty.
        for match in matches:
            cleaned = match.strip()
            if cleaned:
                return cleaned
    return None

def run_question(llm, evaluator, question, question_id, image_base_path, cache_dir=None, model_name=None):
    image_path = os.path.join(image_base_path, str(question['gate_count']), f"question_{question_id}.png")
    expected_expression = question['expression']
    
    # Check cache
    if cache_dir and model_name:
        sanitized_model = sanitize_filename(model_name)
        model_cache_dir = os.path.join(cache_dir, sanitized_model)
        os.makedirs(model_cache_dir, exist_ok=True)
        cache_file = os.path.join(model_cache_dir, f"{question_id}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    response = cached_data.get("response")
                    # If we have a cached response, proceed to extraction
                    if response:
                        extracted_expr = extract_expression(response)
                        if extracted_expr:
                            is_equivalent = evaluator.check_equivalence(extracted_expr, expected_expression)
                            return 1.0 if is_equivalent else 0.0
                        else:
                            return 0.0
            except Exception as e:
                print(f"Error reading cache for question {question_id}: {e}")

    prompt = """Extract the boolean algebra expression from the image.
    
    Respond with the single line boolean algebra expression in a code block. Use operators in word form, not symbols. (e.g. "and" instead of "∧")
    
    Example:
    ```
    not ((A and B) xor C)
    ```
    """
    
    correct_count = 0
    tries = 1 # Retry logic is now in LLM class, but we might want to retry question logic? 
              # The original code had retry loop for the question, but tries=1.
              # I will keep tries=1 here as LLM handles API retries.
    
    for _ in range(tries):
        try:
            response = llm.generate_text(prompt=prompt, image_path=image_path, retries=3, timeout=90)
            
            # Save to cache
            if cache_dir and model_name:
                try:
                    sanitized_model = sanitize_filename(model_name)
                    model_cache_dir = os.path.join(cache_dir, sanitized_model)
                    # Ensure dir exists (redundant but safe)
                    os.makedirs(model_cache_dir, exist_ok=True)
                    cache_file = os.path.join(model_cache_dir, f"{question_id}.json")
                    with open(cache_file, 'w') as f:
                        json.dump({
                            "question_id": question_id,
                            "response": response,
                            "timestamp": datetime.now().isoformat()
                        }, f)
                except Exception as e:
                    print(f"Error saving cache for question {question_id}: {e}")

            extracted_expr = extract_expression(response)
            
            if extracted_expr:
                is_equivalent = evaluator.check_equivalence(extracted_expr, expected_expression)
                if is_equivalent:
                    correct_count += 1
            else:
                # Failed to extract
                pass
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            
    return correct_count / tries

def run_benchmark_for_model(model_name, args, questions, project_root):
    # Determine which API key/endpoint to use
    
    effective_api_key = args.api_key or args.open_router_api_key
    effective_endpoint = args.endpoint
    
    # If the endpoint is the default OpenRouter endpoint and the open-router-endpoint is different, use the open-router-endpoint.
    if args.endpoint == "https://openrouter.ai/api/v1" and args.open_router_endpoint != "https://openrouter.ai/api/v1":
        effective_endpoint = args.open_router_endpoint
        
    if not effective_api_key:
        # Try env var
        effective_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        
    if not effective_api_key:
        print(f"Warning: No API key provided for model {model_name}. Skipping.")
        return None

    print(f"Initializing model: {model_name} at {effective_endpoint}")
    
    llm = LLM(
        model=model_name,
        endpoint=effective_endpoint,
        api_key=effective_api_key,
        reasoning_effort=args.reasoning_effort,
        reasoning_max_tokens=args.reasoning_max_tokens,
        max_output_tokens=args.max_output_tokens
    )
    
    evaluator = ExpressionEvaluator()
    
    image_base_path = os.path.join(project_root, "questions", "images")
    cache_dir = os.path.join(project_root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    results_by_gate_count = defaultdict(list)
    total_score = 0
    total_questions = len(questions)
    
    print(f"Starting benchmark for {model_name} with {total_questions} questions...")
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        # Submit all tasks
        future_to_qid = {
            executor.submit(run_question, llm, evaluator, q, i+1, image_base_path, cache_dir, model_name): (i+1, q)
            for i, q in enumerate(questions)
        }
        
        completed = 0
        for future in as_completed(future_to_qid):
            qid, question = future_to_qid[future]
            try:
                score = future.result()
                results_by_gate_count[question['gate_count']].append(score)
                total_score += score
                completed += 1
                if completed % 10 == 0:
                    print(f"Completed {completed}/{total_questions} questions.")
            except Exception as e:
                print(f"Exception for question {qid}: {e}")
                
    # Calculate stats
    final_score = total_score / total_questions if total_questions > 0 else 0
    
    gate_count_stats = {}
    for gc, scores in results_by_gate_count.items():
        if scores:
            gate_count_stats[gc] = sum(scores) / len(scores)
        else:
            gate_count_stats[gc] = 0

    result_entry = {
        "model": model_name,
        "score": final_score,
        "timestamp": datetime.now().isoformat(),
        "breakdown": gate_count_stats # "average score per group of gate_count"
    }
    
    return result_entry

def update_leaderboard(readme_path, model_name, score):
    """Update the leaderboard section in README.md with the new model result."""
    try:
        with open(readme_path, 'r') as f:
            lines = f.readlines()
        
        # Find the leaderboard section
        leaderboard_start = -1
        table_start = -1
        table_end = -1
        
        for i, line in enumerate(lines):
            if line.strip() == "## Leaderboard":
                leaderboard_start = i
            elif leaderboard_start >= 0 and line.strip().startswith("| Model | Score |"):
                table_start = i
            elif table_start >= 0 and line.strip().startswith("|") and not line.strip().startswith("| :"):
                continue
            elif table_start >= 0 and not line.strip().startswith("|"):
                table_end = i
                break
        
        if leaderboard_start < 0 or table_start < 0:
            print(f"Warning: Could not find leaderboard section in README.md")
            return
        
        # Parse existing entries
        entries = {}
        for i in range(table_start + 2, table_end if table_end > 0 else len(lines)):
            line = lines[i].strip()
            if line.startswith("|"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3 and parts[1] and parts[2]:
                    try:
                        # Extract model name and score
                        entry_model = parts[1]
                        entry_score_str = parts[2].replace("%", "").strip()
                        entry_score = float(entry_score_str)
                        entries[entry_model] = entry_score
                    except (ValueError, IndexError):
                        continue
        
        # Update or add the new model
        entries[model_name] = score * 100  # Convert to percentage
        
        # Sort by score (descending)
        sorted_entries = sorted(entries.items(), key=lambda x: x[1], reverse=True)
        
        # Rebuild the table
        new_table_lines = [
            "| Model | Score |\n",
            "| :--- | ---: |\n"
        ]
        
        for model, score_pct in sorted_entries:
            new_table_lines.append(f"| {model} | {score_pct:.1f}% |\n")
        
        # Replace the old table with the new one
        if table_end < 0:
            table_end = len(lines)

        new_lines = lines[:table_start] + new_table_lines + lines[table_end:]
        
        # Write back to README
        with open(readme_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"Updated leaderboard in README.md")
        
    except Exception as e:
        print(f"Warning: Failed to update leaderboard in README.md: {e}")

def main():
    args = parse_arguments()
    
    # Locate project root
    # Assumes src/run.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load questions
    expressions_path = os.path.join(project_root, "questions", "expressions.json")
    if not os.path.exists(expressions_path):
        print(f"Error: Could not find expressions file at {expressions_path}")
        sys.exit(1)
        
    with open(expressions_path, 'r') as f:
        questions = json.load(f)
        
    models = [m.strip() for m in args.model.split(';') if m.strip()]
    
    results_file = os.path.join(project_root, "results", "results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Load existing results
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            all_results = []
    else:
        all_results = []
        
    for model in models:
        result = run_benchmark_for_model(model, args, questions, project_root)
        if result:
            all_results.append(result)
            
            # Write after each model to save progress
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
                
            print(f"Results for {model} saved. Score: {result['score']:.2%}")
            
            # Update leaderboard in README.md unless disabled
            if not args.no_auto_update_leaderboard:
                readme_path = os.path.join(project_root, "README.md")
                update_leaderboard(readme_path, model, result['score'])

if __name__ == "__main__":
    main()

