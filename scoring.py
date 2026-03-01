import modal
import pandas as pd

# 1. Environment Setup
image = modal.Image.debian_slim().pip_install("vllm", "pandas", "huggingface_hub")
app = modal.App("small-model-consensus")

# --- JUDGE A: LLAMA 3.1 70B ---
@app.cls(
    image=image,
    gpu="A100", 
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class LlamaJudge:
    @modal.enter()
    def load(self):
        from vllm import LLM
        # 8B model fits perfectly on 24GB VRAM
        self.llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    @modal.method()
    def grade(self, text):
        from vllm import SamplingParams
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGrade this resume from 1-10 on professional merit. Output ONLY the number.\n\nResume: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        output = self.llm.generate(prompt, SamplingParams(temperature=0, max_tokens=5))
        return output[0].outputs[0].text.strip()

# --- JUDGE B: QWEN 2.5 72B ---
@app.cls(
    image=image,
    gpu="A100", 
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class QwenJudge:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

    @modal.method()
    def grade(self, text):
        from vllm import SamplingParams
        prompt = f"<|im_start|>user\nYou are a professional recruiter. Rate this resume's quality from 1-10. Output ONLY the number.\n\nResume: {text}<|im_end|>\n<|im_start|>assistant\n"
        output = self.llm.generate(prompt, SamplingParams(temperature=0, max_tokens=5))
        return output[0].outputs[0].text.strip()

# --- LOCAL MANAGER ---
@app.local_entrypoint()
def main():
    df = pd.read_csv("llama_minimal_pairs.csv")
    texts = df['text'].tolist()

    print("⚖️ Initializing Dual-Model Supreme Court...")
    llama = LlamaJudge()
    qwen = QwenJudge()

    # .map() sends all 1,000 texts to the cloud in one "burst"
    print("🤖 Processing in parallel...")
    scores_llama = list(llama.grade.map(texts))
    scores_qwen = list(qwen.grade.map(texts))

    # Helper to ensure scores are numbers
    def clean(v):
        try: return float(''.join(c for c in str(v) if c.isdigit() or c=='.'))
        except: return 5.0

    df['Llama_Score'] = [clean(s) for s in scores_llama]
    df['Qwen_Score'] = [clean(s) for s in scores_qwen]
    df['Consensus_Score'] = (df['Llama_Score'] + df['Qwen_Score']) / 2
    
    # NEW: DISAGREEMENT FILTER
    df['Disagreement_Gap'] = abs(df['Llama_Score'] - df['Qwen_Score'])
    
    # We drop any row where judges differ by more than 3 points
    threshold = 3.0
    clean_df = df[df['Disagreement_Gap'] <= threshold].copy()
    dropped_count = len(df) - len(clean_df)

    print(f"🧹 Filtered out {dropped_count} questionable rows due to judge disagreement.")
    
    # Save the full data but keep a 'is_clean' flag
    df['is_reliable'] = df['Disagreement_Gap'] <= threshold
    df.to_csv("final_scored_dataset.csv", index=False)

    # Generate the 2x2 Table using ONLY reliable data
    report = clean_df.groupby(['style', 'disc'])['Consensus_Score'].mean().unstack()
    print("\n--- FINAL 2x2 TABLE (CLEAN DATA ONLY) ---")
    print(report)