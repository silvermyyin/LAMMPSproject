# LAMMPS-Gen Project – Current Overview (June 2025)

This document consolidates the **current state** of the project and the **action plan** for the next development cycle. It supersedes previous scattered notes.

---
## 1. Milestones Already Completed ✅

| Area | Status | Details |
|------|--------|---------|
| **Directory Re-organisation** | **Done** | Adopted InputFileSynthesis-style hierarchy for `data/`, `src/`, `scripts/`, `results/`, `configs/`, `tests/`, `experiments/`, `docs/`. |
| **README Coverage** | **Done** | Added/updated READMEs for every top-level folder; marked existing (✅) vs. planned (🔧) components. |
| **Data Migration** | **Done** | • Reference `.in` scripts moved → `data/real_world/research_scripts/`.<br>• Obsolete duplicates removed.<br>• Legacy content kept in `data_backup_*/` for safety. |
| **Baseline Driver Patch** | **Done** | `src/generators/run_baseline_experiment.py` now:<br>• reads from the new reference directory.<br>• saves logs in `results/experiments/dated_runs/`.<br>• parameterises `num_samples`, `temperature`, `reference_dir`. |
| **OpenAI Key Handling** | **Done** | `src/calculations/llm_interface.py` now retrieves `OPENAI_API_KEY` from the environment (no hard-coded secrets). |
| **Logging & Plots** | **Done** | Baseline run auto-saves CSV + metric histograms & validity plots under `results/plots/`. |
| **Testing Skeleton** | **Done** | `tests/README.md` + `tests/unit/` scaffold added for pytest; CI instructions documented. |
| **Experiments Framework** | **Done** | `experiments/README.md` + `ongoing/`, `completed/`, `archived/` placeholders created. |

---
## 2. Baseline Experiment – Next Steps 📏

### Immediate Tasks (Days 1-2)
| Step | Action | Expected Output |
|------|--------|-----------------|
| **1. Environment Setup** | `source .venv/bin/activate`<br>`export OPENAI_API_KEY="your-key"`<br>`export LAMMPS_PATH="/path/to/lmp_serial"` | Ready environment |
| **2. Smoke Test** | `python -c "from src.generators.run_baseline_experiment import run_baseline_experiment; run_baseline_experiment(num_samples=1)"` | 1 CSV row + log file |
| **3. Fix Import Errors** | Resolve any missing dependencies in `src/evaluation/evaluator.py` | Clean 1-sample run |
| **4. Verify Metrics** | Check CSV contains: `bleu_score`, `f1_score`, `naive_bleu_score`, `semantic_similarity`, `kw_f1`, `is_valid`, `syntax_valid` | Complete metrics |

### Scale-Up Phase (Days 3-4)
| Step | Action | Expected Output |
|------|--------|-----------------|
| **5. Small Batch** | `run_baseline_experiment(num_samples=10)` | 10 samples with metrics |
| **6. Full Baseline** | `run_baseline_experiment(num_samples=50)` | Complete baseline dataset |
| **7. Analysis** | Review plots in `results/plots/`, aggregate metrics in logs | Baseline performance report |

---
## 3. Enhanced Performance – Main Steps 🚀

### A. RAG Pipeline (Week 2: June 16-22)
| Day | Task | Deliverable |
|-----|------|-------------|
| **Mon** | Implement `src/rag/document_processing/pdf_processor.py` | PDF → text extraction |
| **Tue** | Create embedding pipeline using OpenAI ada-002 | Text → vectors |
| **Wed** | Build FAISS index in `data/documents/embedded/` | Searchable knowledge base |
| **Thu** | Implement retrieval in `src/rag/retrieval/similarity_search.py` | Context retrieval API |
| **Fri** | Integrate RAG into baseline script | RAG-enhanced generation |
| **Sat-Sun** | Test RAG vs baseline on 20 samples | Performance comparison |

### B. Prompt Engineering (Week 3: June 23-25)
| Day | Task | Deliverable |
|-----|------|-------------|
| **Mon** | Create templates: `basic.txt`, `cot.txt`, `cove.txt` in `data/prompts/system_prompts/` | Prompt template library |
| **Tue** | Implement template selector in `src/prompt_creation/system_prompts/` | Dynamic prompt system |
| **Wed** | Run A/B tests: baseline vs CoT vs CoVe (10 samples each) | Prompt effectiveness data |

### C. Fine-Tuning (Week 3-4: June 26-30)
| Day | Task | Deliverable |
|-----|------|-------------|
| **Thu (26th)** | Format training data as JSONL in `data/training/fine_tuning/` | Training dataset |
| **Fri (27th)** | Set up LoRA fine-tuning script using HuggingFace PEFT | Training pipeline |
| **Sat (28th)** | Train model on LAMMPS data (8-12 hours) | Fine-tuned model checkpoint |
| **Sun (29th)** | Evaluate fine-tuned model, debug issues, retrain if needed | Validated fine-tuned model |
| **Mon (30th)** | Final comparison: baseline vs RAG vs prompt vs fine-tuned | Project completion report |

---
## 4. June 30th Completion Checklist ✅

### Week 1 (June 12-15): Foundation
- [ ] Baseline experiment running (50+ samples) - **Days 1-4**
- [ ] All metrics validated (F1, BLEU, Executability, Syntax)
- [ ] Results visualization working
- [ ] Unit tests for core components

### Week 2 (June 16-22): RAG Implementation  
- [ ] PDF processing pipeline
- [ ] FAISS vector database
- [ ] RAG-enhanced generation
- [ ] RAG vs baseline comparison

### Week 3 (June 23-25): Prompt Engineering
- [ ] Prompt template system - **Days 1-3**
- [ ] A/B testing framework
- [ ] Best prompt identification

### Week 3-4 (June 26-30): Fine-Tuning Focus
- [ ] Training data preparation - **Day 1**
- [ ] LoRA fine-tuning pipeline - **Day 2**
- [ ] Model training completed - **Day 3**
- [ ] Model validation and debugging - **Day 4**
- [ ] Comprehensive final evaluation - **Day 5**

### Success Metrics by June 30th
- **Baseline**: 50+ samples with complete metrics
- **RAG**: 15%+ improvement in F1 or BLEU score
- **Prompt Engineering**: Best template identified and validated
- **Fine-Tuning**: Custom model outperforming baseline (5+ days allocated)
- **Documentation**: Complete results analysis and recommendations

---
## 5. Daily Execution Commands

### Baseline (Week 1: Days 1-4)
```bash
# Day 1: Setup and smoke test
export OPENAI_API_KEY="your-key"
export LAMMPS_PATH="/path/to/lammps"
python -c "from src.generators.run_baseline_experiment import run_baseline_experiment; run_baseline_experiment(num_samples=1)"

# Day 3: Full baseline
python -c "from src.generators.run_baseline_experiment import run_baseline_experiment; run_baseline_experiment(num_samples=50)"
```

### RAG (Week 2)
```bash
# Day 1: Process documents
python src/rag/document_processing/pdf_processor.py --input data/documents/raw/ --output data/documents/embedded/

# Day 5: RAG-enhanced generation
python -c "from src.generators.run_baseline_experiment import run_baseline_experiment; run_baseline_experiment(num_samples=20, use_rag=True)"
```

### Prompt Engineering (Week 3: Days 1-3)
```bash
# Day 3: A/B test prompts
python scripts/evaluation/comparison/benchmark_prompts.py --templates basic,cot,cove --samples 10
```

### Fine-Tuning (Week 3-4: Days 4-8)
```bash
# Day 1 (26th): Prepare training data
python scripts/training/data_preparation/format_training_data.py --input data/real_world/research_scripts/ --output data/training/fine_tuning/

# Day 2 (27th): Setup training pipeline
python scripts/training/fine_tuning/train_lora.py --config configs/model_configs/lora_config.yaml --dry-run

# Day 3 (28th): Start training (long-running)
python scripts/training/fine_tuning/train_lora.py --config configs/model_configs/lora_config.yaml

# Day 4 (29th): Evaluate and debug
python scripts/evaluation/model_validation/validate_finetuned.py --model results/models/lora_checkpoint/

# Day 5 (30th): Final comparison
python scripts/evaluation/performance_metrics/final_comparison.py --models baseline,rag,prompt,finetune
```

---
## 6. Post-June 30th Vision 🔮

After completing the core project by June 30th, potential extensions include:

1. **Advanced RAG** with hierarchical retrieval and multi-modal documents
2. **Physics-aware validation** (energy conservation, pressure consistency)  
3. **Web dashboard** for experiment tracking (Gradio/Streamlit)
4. **Cloud deployment** with GPU acceleration
5. **Community integration** with LAMMPS user forums

---
*Document updated: 2025-06-12 for June 30th deadline with extended fine-tuning phase.* 