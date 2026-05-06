import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, f1_score
from rag_system import load_pdf, load_docx

# -----------------------------
# 配置路径
# -----------------------------
transcript_folder = "./transcripts"      # LLM输入的PDF/DOCX/TXT
human_folder = "./human_coding"          # 人工编码JSON文件
llm_output_folder = "./coding_outputs"   # LLM输出JSON
summary_csv = "./llm_vs_human_summary.csv"
os.makedirs(llm_output_folder, exist_ok=True)

# -----------------------------
# Codebook设置
# -----------------------------
codebook = ["environmental_barrier", "social_support", "healthcare_access", "stigma", "mental_health"]

# -----------------------------
# 初始化OpenAI
# -----------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set.")
client = OpenAI(api_key=api_key)

# -----------------------------
# Helper：读取 transcript
# -----------------------------
def load_transcript(file_path):
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".pdf":
        pages = load_pdf(file_path)
        return " ".join(page_text for _, page_text in pages)
    elif suffix == ".docx":
        pages = load_docx(file_path)
        return " ".join(page_text for _, page_text in pages)
    elif suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

# -----------------------------
# Helper：LLM coding
# -----------------------------
def run_llm_coding(text_segment):
    prompt = f"""
You are a qualitative research coding assistant.

Transcript segment:
{text_segment}

Task:
- Assign thematic codes to each meaningful segment.
- Use the following codebook: {codebook}
- Output JSON format:
[
  {{"text": "Segment text here","codes":["relevant_code1","relevant_code2"]}}
]
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role":"system","content":"Expert qualitative coding assistant."},
                  {"role":"user","content":prompt}]
    )
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [{"text": text_segment, "codes": ["PARSE_ERROR"], "raw": content}]

# -----------------------------
# Step 1: LLM Coding for all transcripts
# -----------------------------
for transcript_file in glob.glob(os.path.join(transcript_folder, "*.*")):
    base_name = os.path.splitext(os.path.basename(transcript_file))[0]
    print(f"Processing {base_name} ...")
    text = load_transcript(transcript_file)
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
    aggregated_output = []
    for chunk in chunks:
        aggregated_output.extend(run_llm_coding(chunk))
    output_path = os.path.join(llm_output_folder, f"{base_name}_coding.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_output, f, indent=2, ensure_ascii=False)
    print(f"Saved LLM coding: {output_path}")

# -----------------------------
# Step 2: Compute LLM vs Human consistency
# -----------------------------
summary_list = []
for llm_file in glob.glob(os.path.join(llm_output_folder, "*_coding.json")):
    base_name = os.path.basename(llm_file).replace("_coding.json", "")
    human_file = os.path.join(human_folder, f"{base_name}_human.json")
    if not os.path.exists(human_file):
        print(f"Skipping {base_name}, human coding not found.")
        continue
    with open(llm_file, "r", encoding="utf-8") as f:
        llm_codes = [set(item.get("codes", [])) for item in json.load(f)]
    with open(human_file, "r", encoding="utf-8") as f:
        human_codes = [set(item.get("codes", [])) for item in json.load(f)]
    min_len = min(len(llm_codes), len(human_codes))
    llm_codes = llm_codes[:min_len]
    human_codes = human_codes[:min_len]
    for code in codebook:
        llm_binary = [1 if code in s else 0 for s in llm_codes]
        human_binary = [1 if code in s else 0 for s in human_codes]
        kappa = cohen_kappa_score(llm_binary, human_binary)
        f1 = f1_score(human_binary, llm_binary)
        summary_list.append({"transcript": base_name, "code": code, "cohen_kappa": kappa, "f1_score": f1})

# -----------------------------
# Step 3: Save summary CSV
# -----------------------------
df = pd.DataFrame(summary_list)
df.to_csv(summary_csv, index=False)
print(f"Saved summary CSV: {summary_csv}")

# -----------------------------
# Step 4: Visualize heatmaps
# -----------------------------
# Cohen's kappa heatmap
kappa_pivot = df.pivot(index="transcript", columns="code", values="cohen_kappa")
plt.figure(figsize=(10,6))
sns.heatmap(kappa_pivot, annot=True, cmap="coolwarm", center=0, cbar_kws={'label': "Cohen's kappa"})
plt.title("LLM vs Human Coding: Cohen's Kappa per Code and Transcript")
plt.tight_layout()
plt.show()

# F1 score heatmap
f1_pivot = df.pivot(index="transcript", columns="code", values="f1_score")
plt.figure(figsize=(10,6))
sns.heatmap(f1_pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={'label': "F1 Score"})
plt.title("LLM vs Human Coding: F1 Score per Code and Transcript")
plt.tight_layout()
plt.show()

# -----------------------------
# Step 5: Average metrics per code
# -----------------------------
avg_metrics = df.groupby("code").agg({"cohen_kappa":"mean","f1_score":"mean"}).reset_index()
print("\nAverage consistency per code:")
print(avg_metrics)
