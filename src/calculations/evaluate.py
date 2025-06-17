import os
import json
import subprocess
from typing import Dict, List, Tuple, Union, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import re
from nltk.translate.bleu_score import sentence_bleu
import datetime

class LAMMPSEvaluator:
    def __init__(self, config_path: str = "configs/model_configs.json"):
        """Initialize the LAMMPS evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize semantic similarity model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        self.required_sections = [
            'units', 'atom_style', 'boundary', 'pair_style',
            'fix', 'thermo', 'timestep', 'run'
        ]
        
    def calculate_f1_score(self, reference: str, generated: str) -> float:
        """计算F1分数"""
        ref_tokens = set(self._tokenize_script(reference))
        gen_tokens = set(self._tokenize_script(generated))
        
        if not ref_tokens or not gen_tokens:
            return 0.0
            
        precision = len(ref_tokens & gen_tokens) / len(gen_tokens)
        recall = len(ref_tokens & gen_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def validate_lammps_script(self, script_content: str, experiment_type: str = "baseline") -> Tuple[bool, List[str]]:
        """
        Validate a LAMMPS script by attempting to run it.
        
        Args:
            script_content (str): The content of the LAMMPS script to validate
            experiment_type (str): Type of experiment (baseline, prompt_engineering, rag, fine_tuning)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        # 检查必要的命令
        required_commands = {
            'units': False,
            'atom_style': False,
            'boundary': False,
            'mass': False,
            'fix': False
        }
        
        # 分析脚本内容
        lines = script_content.split('\n')
        for line in lines:
            line = line.strip().lower()
            for cmd in required_commands:
                if line.startswith(cmd):
                    required_commands[cmd] = True
        
        # 检查缺失的命令
        missing_commands = [cmd for cmd, present in required_commands.items() if not present]
        if missing_commands:
            return False, [f"Missing required commands: {', '.join(missing_commands)}"]
        
        # 生成唯一的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 从脚本内容中提取简短描述
        description = "unknown"
        for line in lines:
            if line.startswith("#"):
                description = line.strip("# ").lower().replace(" ", "_")[:30]
                break
        
        # 构建文件名
        script_filename = f"{experiment_type}_{timestamp}_{description}.in"
        script_path = os.path.join("LAMMPSrun", experiment_type, script_filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        # 写入脚本文件
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 运行LAMMPS
        try:
            result = subprocess.run(
                ['lmp', '-in', script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # 检查返回码
            if result.returncode != 0:
                return False, [f"LAMMPS execution failed with return code {result.returncode}"]
            
            # 检查输出中的错误和警告
            error_messages = []
            for line in result.stderr.split('\n'):
                if 'ERROR' in line or 'WARNING' in line:
                    error_messages.append(line.strip())
            
            if error_messages:
                return False, error_messages
            
            return True, []
            
        except subprocess.TimeoutExpired:
            return False, ["LAMMPS execution timed out"]
        except Exception as e:
            return False, [f"Error running LAMMPS: {str(e)}"]
    
    def check_syntax_validity(self, script: str) -> Tuple[bool, List[str]]:
        """检查LAMMPS脚本的语法有效性"""
        errors = []
        tokens = self._tokenize_script(script)
        
        # 检查必需部分
        for section in self.required_sections:
            if not any(section in token for token in tokens):
                errors.append(f"Missing required section: {section}")
        
        # 检查基本语法
        if not re.search(r'run\s+\d+', script):
            errors.append("Missing or invalid run command")
            
        if not re.search(r'timestep\s+\d+\.?\d*', script):
            errors.append("Missing or invalid timestep")
            
        return len(errors) == 0, errors
    
    def calculate_semantic_similarity(self, reference: str, generated: str) -> float:
        """计算语义相似度"""
        ref_params = self._extract_parameters(reference)
        gen_params = self._extract_parameters(generated)
        
        if not ref_params or not gen_params:
            return 0.0
            
        common_params = set(ref_params.keys()) & set(gen_params.keys())
        if not common_params:
            return 0.0
            
        similarities = []
        for param in common_params:
            if ref_params[param] == gen_params[param]:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
                
        return sum(similarities) / len(similarities)
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """计算 BLEU Score"""
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        return sentence_bleu([reference_tokens], candidate_tokens)
    
    def evaluate_script(self, reference_script: str, generated_script: str) -> Dict[str, Any]:
        """评估生成的脚本"""
        f1_score = self.calculate_f1_score(reference_script, generated_script)
        semantic_similarity = self.calculate_semantic_similarity(reference_script, generated_script)
        executable, exec_errors = self.validate_lammps_script(generated_script)
        syntax_validity, syntax_errors = self.check_syntax_validity(generated_script)
        bleu_score = self.calculate_bleu_score(reference_script, generated_script)
        
        # Calculate keyword F1 score
        ref_tokens = self._tokenize_script(reference_script)
        gen_tokens = self._tokenize_script(generated_script)
        keyword_f1, _, _ = self.calculate_f1(gen_tokens, ref_tokens)
        
        return {
            'f1_score': f1_score,
            'semantic_similarity': semantic_similarity,
            'executable': executable,
            'syntax_valid': syntax_validity,
            'syntax_errors': syntax_errors,
            'bleu_score': bleu_score,
            'keyword_f1_score': keyword_f1
        }
    
    def _tokenize_script(self, script: str) -> List[str]:
        """将LAMMPS脚本分词"""
        # 移除注释和空行
        lines = [line.strip() for line in script.split('\n') 
                if line.strip() and not line.strip().startswith('#')]
        
        # 分词
        tokens = []
        for line in lines:
            tokens.extend(line.split())
            
        return tokens

    def _extract_parameters(self, script: str) -> Dict[str, str]:
        """提取LAMMPS脚本中的参数"""
        params = {}
        
        # 提取units
        if match := re.search(r'units\s+(\w+)', script):
            params['units'] = match.group(1)
            
        # 提取atom_style
        if match := re.search(r'atom_style\s+(\w+)', script):
            params['atom_style'] = match.group(1)
            
        # 提取boundary
        if match := re.search(r'boundary\s+([pfs]\s+[pfs]\s+[pfs])', script):
            params['boundary'] = match.group(1)
            
        # 提取pair_style
        if match := re.search(r'pair_style\s+(\w+)', script):
            params['pair_style'] = match.group(1)
            
        # 提取fix参数
        if match := re.search(r'fix\s+\d+\s+all\s+(\w+)\s+temp\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', script):
            params['ensemble'] = match.group(1)
            params['temp_start'] = match.group(2)
            params['temp_end'] = match.group(3)
            params['temp_damp'] = match.group(4)
            
        return params

    def calculate_f1(self, predicted: List[str], reference: List[str]) -> Tuple[float, float, float]:
        """
        Calculate F1 score, precision, and recall for two lists of strings.
        
        Args:
            predicted (List[str]): List of predicted strings
            reference (List[str]): List of reference strings
            
        Returns:
            Tuple[float, float, float]: (F1 score, precision, recall)
        """
        if len(predicted) == 0 and len(reference) == 0:
            return 1.0, 1.0, 1.0
            
        # Calculate true positives, false positives, and false negatives
        true_positives = len([item for item in predicted if item in reference])
        false_positives = len([item for item in predicted if item not in reference])
        false_negatives = len([item for item in reference if item not in predicted])
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall
    
    def evaluate_batch(self, references: List[str], generated: List[str]) -> Dict[str, float]:
        """Evaluate a batch of generated scripts."""
        batch_results = []
        for ref, gen in zip(references, generated):
            batch_results.append(self.evaluate_script(ref, gen))
            
        # Calculate aggregate metrics
        aggregate_results = {
            'avg_f1_score': np.mean([r['f1_score'] for r in batch_results]),
            'avg_semantic_similarity': np.mean([r['semantic_similarity'] for r in batch_results]),
            'executability_rate': np.mean([r['executability'] for r in batch_results]),
            'syntax_validity_rate': np.mean([r['syntax_validity'] for r in batch_results])
        }
        
        return aggregate_results

# testing function
if __name__ == "__main__":
    # 示例：真实 LAMMPS 代码 vs. LLM 生成代码
    reference_code = """
    units metal
    atom_style atomic
    boundary p p p
    pair_style eam
    fix 1 all nvt temp 300.0 300.0 0.1
    thermo 100
    timestep 1.0
    run 10000
    """

    generated_code = """
    units metal
    atom_style atomic
    boundary p p p
    pair_style eam
    fix 1 all nvt temp 300.0 300.0 0.1
    thermo 50
    timestep 1.0
    run 10000
    """

    # 计算 F1-score
    f1_score = calculate_f1_score(reference_code, generated_code)
    print(f"F1-score: {f1_score:.4f}")

    # 检查可执行性
    executable = validate_lammps_script(generated_code)
    print(f"LAMMPS 代码可执行: {executable}")
