# Generated from: foldsmith-copilot.ipynb
# Converted at: 2026-02-02T11:39:09.163Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install -q gradio transformers torch accelerate py3Dmol biopython


import torch
import gradio as gr
import py3Dmol
import numpy as np

from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


# Load model from HuggingFace
model_name = "facebook/esmfold_v1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForProteinFolding.from_pretrained(model_name)
model = model.cuda()
model.eval()


def fold_protein(sequence):
    sequence = sequence.replace(" ", "").upper()
    
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model(**inputs)
        
    # Convert structure to PDB
    pdb = to_pdb(outputs)
    
    # Extract pLDDT scores
    plddt = outputs["plddt"][0].cpu().numpy()
    avg_plddt = float(np.mean(plddt))
    
    return pdb, avg_plddt, plddt


def render_structure(pdb_str):
    view = py3Dmol.view(width=500, height=500)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view


def foldsmit_copilot(sequence):
    pdb, avg_plddt, plddt = fold_protein(sequence)
    
    structure_view = render_structure(pdb)
    
    confidence_msg = f"""
    🔬 **FoldSmith Confidence Report**
    
    • Average pLDDT: **{avg_plddt:.2f}**
    
    **Interpretation**
    - >90 → Very high confidence
    - 70–90 → Reliable fold
    - 50–70 → Moderate
    - <50 → Low confidence / flexible regions
    """
    
    return structure_view, pdb, avg_plddt, confidence_msg


import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import torch
torch.cuda.empty_cache()


with gr.Blocks(title="FoldSmith Copilot 🧬") as demo:
    
    gr.Markdown(
        """
        # 🧬 FoldSmith Copilot
        **Protein Structure Prediction powered by HuggingFace ESMFold**
        
        Paste an amino acid sequence and FoldSmith will:
        - Predict 3D structure
        - Show confidence (pLDDT)
        - Allow PDB download
        """
    )
    
    with gr.Row():
        seq_input = gr.Textbox(
            label="Protein Sequence",
            placeholder="MKTAYIAKQRQISFVKSHFSRQDILDLWQ...",
            lines=5
        )
    
    fold_btn = gr.Button("🚀 Fold Protein")
    
    with gr.Row():
        structure_output = gr.HTML(label="3D Structure")
        pdb_output = gr.File(label="Download PDB")
    
    plddt_output = gr.Number(label="Average pLDDT Score")
    copilot_output = gr.Markdown()
    
    def run_all(sequence):
        structure, pdb, avg_plddt, msg = foldsmit_copilot(sequence)
        
        # Save PDB for download
        pdb_path = "prediction.pdb"
        with open(pdb_path, "w") as f:
            f.write(pdb)
        
        return structure._make_html(), pdb_path, avg_plddt, msg
    
    fold_btn.click(
        run_all,
        inputs=seq_input,
        outputs=[structure_output, pdb_output, plddt_output, copilot_output]
    )

demo.launch(debug=True)