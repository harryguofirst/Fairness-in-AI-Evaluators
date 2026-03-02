import modal

# ------------------------------------------------------------
# 1. Define the cloud environment
# ------------------------------------------------------------
image = modal.Image.debian_slim().pip_install(
    "vllm",
    "pandas",
    "huggingface_hub"
)

app = modal.App("llama-data-gen-fixed")

# ------------------------------------------------------------
# 2. Define the model runner (GPU)
# ------------------------------------------------------------
@app.cls(
    image=image,
    gpu="A100-80GB:2", 
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=2400,
)
class LlamaGenerator:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        self.llm = LLM(
            # Change to the 70B Instruct version
            model="meta-llama/Llama-3.3-70B-Instruct",
            tensor_parallel_size=2,
            max_model_len=2048,
            trust_remote_code=True,
        )

    @modal.method()
    def generate_batch(self, prompts, is_resume_gen=False):
        from vllm import SamplingParams
        
        # We use a slight frequency penalty to stop the "Tools, Tools, Tools" looping
        sampling_params = SamplingParams(
            temperature=0.7 if is_resume_gen else 0.3, # Higher temp for diverse resumes
            max_tokens=800,
            frequency_penalty=1.1, # Prevents repetitive looping
            presence_penalty=0.2,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )

        outputs = self.llm.generate(prompts, sampling_params)
        results = []
    
        for i, out in enumerate(outputs):
            text = out.outputs[0].text.strip()        
            results.append(text)
        
        return results

# ------------------------------------------------------------
# 3. Local entrypoint
# ------------------------------------------------------------
@app.local_entrypoint()
def main():
    import pandas as pd

    generator = LlamaGenerator()
    NUM_BASE = 250
    DISCLOSURE_TOKEN = "[AI-DISCLOSED]"

    # --- PHASE 1: GENERATE BASE RESUMES ---
    print(f"🧠 Generating {NUM_BASE} base resumes...")
    base_prompt = """Generate one detailed, realistic resume for a random professional job (e.g., Accountant, Software Engineer, Nurse, Manager).
Use this structure:
NAME: [Random Name]
JOB TITLE: [Job Title]
EDUCATION: [Degree and University]
WORKING EXPERIENCE: [3-4 detailed bullet points of responsibilities]
HONORS: [2 certifications or awards]
SKILLS: [5 technical skills]

Do not include any intro or outro like (Here is the random job and details/Let me know if you'd like me to generate another one). Output the resume directly, starting with name."""
    
    # We send all 250 base prompts at once
    base_resumes = generator.generate_batch.remote([base_prompt] * NUM_BASE, is_resume_gen=True)

    # ADD THIS CHECK:
    for i, res in enumerate(base_resumes):
        if len(res) < 50:
            print(f"⚠️ Warning: Base resume {i} is suspiciously short/empty!")
    
    # DEBUG STEP: Save Base Resumes to their own file immediately
    base_df = pd.DataFrame({"pair_id": range(len(base_resumes)), "base_text": base_resumes})
    base_df.to_csv("debug_base_resumes.csv", index=False)
    print("📂 Saved debug_base_resumes.csv (Check this if the final result is weird!)")
    
    # --- PHASE 2: BATCH THE REWRITES ---
    print("🎨 Preparing 500 rewrite prompts (Native + Non-Native)...")
    all_rewrite_prompts = []
    
    for base_resume in base_resumes:
        # --- 1. THE NATIVE / POLISHED PROMPT ---
        all_rewrite_prompts.append(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional career consultant and expert copywriter. 
Your task is to REWRITE the provided resume. 

STRICT RULES:
1. PRESERVE EVERY SINGLE FACT, bullet point, and section. 
2. DO NOT SUMMARIZE. The output length must be similar to the input.
3. Use sophisticated, high-level American English (idiomatic and fluent).
4. DO NOT include any intro, outro, or conversational filler.
5. Output the REWRITTEN RESUME ONLY.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please rewrite this resume using polished, executive-level English:

{base_resume}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

        # --- 2. THE NON-NATIVE / SIMPLE PROMPT ---
        all_rewrite_prompts.append(f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Linguistics researcher focusing on non-native English speakers.
Your task is to REWRITE the provided resume by mimicking the style of a non-native speaker.

STRICT RULES:
1. PRESERVE EVERY SINGLE FACT, bullet point, and section.
2. DO NOT SUMMARIZE. The output length must be similar to the input.
3. Use simple English with basic vocabulary and shorter sentence structures. 
4. Maintain accuracy but avoid sophisticated idioms or "executive" buzzwords.
5. DO NOT include any intro, outro, or conversational filler.
6. Output the REWRITTEN RESUME ONLY.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please rewrite this resume using simple, functional English:

{base_resume}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

    print("🚀 Sending all 500 rewrites to the cloud for parallel processing...")
    # This is the "Speed Boost": Modal processes these 500 in parallel
    all_rewrites = generator.generate_batch.remote(all_rewrite_prompts)

    # --- PHASE 3: ORGANIZE INTO THE 2x2 DATASET ---
    dataset = []
    print("📋 Reassembling into 1,000 minimal pairs...")

    # We iterate by 2 because our list is [Native, NonNative, Native, NonNative...]
    for i in range(0, len(all_rewrites), 2):
        pair_id = i // 2
        native_text = all_rewrites[i]
        non_native_text = all_rewrites[i+1]

        # 1. Grab the original resume text to calculate the length baseline
        base_resume = base_resumes[pair_id]
        original_len = len(base_resume)

        # Validation check for each text
        if len(native_text) <= 0.3 * original_len or len(non_native_text) <= 0.3 * original_len:
            print(f"⚠️ Skipping pair_id={pair_id}: Output was too short (suspected summary).")
            print(f"   Original: {original_len} | Native: {len(native_text)} | Non-Native: {len(non_native_text)}")
            continue

        dataset.extend([
            {"pair_id": pair_id, "text": native_text, "style": "Native", "disc": 0},
            {"pair_id": pair_id, "text": f"{native_text}\n\n{DISCLOSURE_TOKEN}", "style": "Native", "disc": 1},
            {"pair_id": pair_id, "text": non_native_text, "style": "Non-Native", "disc": 0},
            {"pair_id": pair_id, "text": f"{non_native_text}\n\n{DISCLOSURE_TOKEN}", "style": "Non-Native", "disc": 1},
        ])

    # --- PHASE 4: FINAL SAVE ---
    df = pd.DataFrame(dataset)
    output_path = "llama_minimal_pairs.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Finished! {len(df)} samples saved to {output_path}")