✨ Features

🔬 Protein Structure Prediction

Uses facebook/esmfold_v1 for fast and accurate structure prediction

Accepts raw amino acid sequences as input

🎨 pLDDT Confidence Coloring

Per-residue pLDDT scores embedded into the PDB B-factor column

3D structure colored by confidence (AlphaFold-style visualization)

🧊 Interactive 3D Visualization

Real-time 3D protein rendering using py3Dmol

Structure displayed directly from the generated PDB output

📥 Downloadable PDB

Export predicted structures with confidence scores included

Ready for downstream analysis in PyMOL, Chimera, or VMD

🤖 LLM-Powered Copilot

Integrated Hugging Face language model (e.g. FLAN-T5)

Ask questions about structure, confidence, domains, or function

🧩 Single-Window Gradio UI

Sequence input, structure visualization, and copilot chat in one interface

🖥️ Demo Workflow

Paste an amino acid sequence

Click Fold Protein

View the pLDDT-colored 3D structure

Download the predicted PDB file

Ask FoldSmith Copilot questions about the protein

🧠 Tech Stack

Protein Folding: ESMFold (facebook/esmfold_v1)

LLM Copilot: Hugging Face Transformers

Visualization: py3Dmol

UI: Gradio

Frameworks: PyTorch, Transformers

🚀 Use Cases

Protein structure exploration

Confidence-aware model inspection

Education and teaching structural biology

Rapid hypothesis generation in research

AI-assisted protein analysis

⚠️ Notes

Best performance with sequences between 50–300 amino acids

Requires GPU for optimal speed (CPU supported but slower)

Predictions are computational models, not experimental structures

📌 Future Enhancements

Per-residue pLDDT plots

Mutation comparison (WT vs mutant)

Domain segmentation

Binding site highlighting

Persistent chat memory

📜 License

This project is intended for research and educational use.
Please follow the licensing terms of ESMFold and Hugging Face models used.

If you want, I can also:

Write a short GitHub tagline

Create a project logo name

Draft a paper-style abstract

Optimize this for AI/ML portfolio visibility
