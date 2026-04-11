"""
inference.py - Content Moderation OpenEnv Baseline Agent
MANDATORY VARIABLES (injected by validator):
    API_BASE_URL  - LiteLLM proxy endpoint
    API_KEY       - Validator API key
    MODEL_NAME    - Model to use
"""
import json
import os
import textwrap
import time
import urllib.request
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — use EXACTLY the env vars the validator injects
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.environ.get("ENV_BASE_URL", "https://heist-content-mod-openenv.hf.space")
TEMPERATURE  = 0.0
MAX_TOKENS   = 512
TASKS        = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# OpenAI client using validator-injected credentials
# ---------------------------------------------------------------------------
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Structured output blocks — required by Phase 2 validator
# ---------------------------------------------------------------------------
def log_start(task):
    print(f"[START] task={task}", flush=True)

def log_step(task, step, reward, done):
    print(f"[STEP] task={task} step={step} reward={reward:.4f} done={done}", flush=True)

def log_end(task, score, steps):
    print(f"[END] task={task} score={score:.4f} steps={steps}", flush=True)

# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------
def _post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        ENV_URL.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def _get(path):
    with urllib.request.urlopen(ENV_URL.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read())

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM = textwrap.dedent("""
You are an expert content moderator. Respond with ONLY valid JSON:
{
  "content_id": "<id>",
  "decision": "<approve|remove|escalate|warn_user|age_restrict|shadow_ban>",
  "violation_category": "<hate_speech|harassment|misinformation|spam|explicit_content|violence|self_harm|none>",
  "severity_assessment": "<low|medium|high|critical>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief>",
  "policy_rule_cited": null
}
Rules: approve=fine, remove=clear violation, escalate=ambiguous,
warn_user=borderline, age_restrict=adult legal, shadow_ban=spam.
Self-harm ALWAYS needs action. No markdown, JSON only.
""").strip()

def build_prompt(obs):
    c = obs.get("content", {})
    u = obs.get("user_context", {})
    rules = "\n".join(
        f"  [{r['rule_id']}] {r['description']}"
        for r in obs.get("applicable_rules", []))
    return (f"content_id: {c.get('content_id')}\n"
            f"text: {c.get('text','')}\n"
            f"reported: {c.get('reported_count')} views: {c.get('view_count')}\n"
            f"reputation: {u.get('reputation')} "
            f"violations: {u.get('prior_violations',0)} "
            f"verified: {u.get('is_verified',False)}\n"
            f"queue: {obs.get('queue_position')}/{obs.get('queue_total')}\n"
            f"rules:\n{rules}\nJSON only.")

def call_llm(prompt):
    """Call LLM through validator proxy — raises on failure so we know."""
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user",   "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS)
    return r.choices[0].message.content or ""

def parse_action(text, content_id):
    try:
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        p = json.loads(t)
        p["content_id"] = content_id
        p.setdefault("reasoning", "")
        p.setdefault("policy_rule_cited", None)
        p.setdefault("escalation_note", None)
        p["confidence"] = float(max(0.0, min(1.0, p.get("confidence", 0.5))))
        return p
    except Exception:
        return {"content_id": content_id, "decision": "escalate",
                "violation_category": "none", "severity_assessment": "medium",
                "confidence": 0.1, "reasoning": "parse error fallback",
                "policy_rule_cited": None, "escalation_note": None}

# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------
def run_task(task_name):
    log_start(task_name)
    obs = _post("/reset", {"task": task_name})
    total_reward = 0.0
    step = 0
    episode_result = {}

    while True:
        cid    = obs["content"]["content_id"]
        text   = call_llm(build_prompt(obs))
        action = parse_action(text, cid)
        result = _post("/step", {"action": action})
        obs    = result["observation"]
        reward = float(result["reward"])
        done   = bool(result["done"])
        total_reward += reward
        step += 1
        log_step(task_name, step, reward, done)
        if done:
            episode_result = result["info"].get("episode_result", {})
            break

    score = float(episode_result.get("final_score", 0.0))
    log_end(task_name, score, step)
    return {"task": task_name, "final_score": score,
            "accuracy": episode_result.get("accuracy", 0.0),
            "correct":  episode_result.get("correct_decisions", 0),
            "total":    episode_result.get("total_items", 0),
            "total_reward": round(total_reward, 4),
            "episode_result": episode_result}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[INFO] Content Moderation OpenEnv — Baseline Agent", flush=True)
    print(f"[INFO] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL={MODEL_NAME}", flush=True)
    print(f"[INFO] ENV={ENV_URL}", flush=True)

    try:
        h = _get("/health")
        print(f"[INFO] Env health: {h.get('status','unknown')}", flush=True)
    except Exception as e:
        print(f"[WARNING] Env health: {e}", flush=True)

    results = []
    t0 = time.time()

    for task in TASKS:
        result = run_task(task)
        results.append(result)

    elapsed = time.time() - t0
    avg = sum(r.get("final_score", 0) for r in results) / len(results)

    print(f"[SUMMARY] avg_score={avg:.4f} runtime={elapsed:.1f}s", flush=True)
    for r in results:
        print(f"[RESULT] task={r['task']} score={r.get('final_score',0):.4f} "
              f"accuracy={r.get('accuracy',0):.4f} "
              f"correct={r.get('correct',0)}/{r.get('total',0)}", flush=True)

    output = {"model": MODEL_NAME, "results": results,
              "avg_score": round(avg, 4), "runtime_sec": round(elapsed, 1)}
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("[INFO] Saved baseline_results.json", flush=True)


if __name__ == "__main__":
    main()