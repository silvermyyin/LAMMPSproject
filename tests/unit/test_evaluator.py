import pytest
from scripts.evaluate import LAMMPSEvaluator

@pytest.fixture
def evaluator():
    return LAMMPSEvaluator()

@pytest.fixture
def sample_scripts():
    reference = """
    units metal
    atom_style atomic
    boundary p p p
    pair_style eam
    fix 1 all nvt temp 300.0 300.0 0.1
    thermo 100
    timestep 1.0
    run 10000
    """
    
    valid_generated = """
    units metal
    atom_style atomic
    boundary p p p
    pair_style eam
    fix 1 all nvt temp 300.0 300.0 0.1
    thermo 50
    timestep 1.0
    run 10000
    """
    
    invalid_generated = """
    units metal
    atom_style atomic
    boundary p p p
    pair_style eam
    fix 1 all nvt temp 300.0 300.0 0.1
    thermo 50
    timestep 1.0
    run 10000
    """
    
    return reference, valid_generated, invalid_generated

def test_calculate_f1_score(evaluator, sample_scripts):
    reference, valid_generated, _ = sample_scripts
    f1_score = evaluator.calculate_f1_score(reference, valid_generated)
    assert 0 <= f1_score <= 1
    assert f1_score > 0.5  # Should be reasonably high for similar scripts

def test_check_syntax_validity(evaluator, sample_scripts):
    _, valid_generated, invalid_generated = sample_scripts
    is_valid, errors = evaluator.check_syntax_validity(valid_generated)
    assert is_valid
    assert len(errors) == 0
    
    # Test with missing required section
    invalid_script = "units metal\nrun 1000"
    is_valid, errors = evaluator.check_syntax_validity(invalid_script)
    assert not is_valid
    assert len(errors) > 0

def test_calculate_semantic_similarity(evaluator, sample_scripts):
    reference, valid_generated, _ = sample_scripts
    similarity = evaluator.calculate_semantic_similarity(reference, valid_generated)
    assert 0 <= similarity <= 1
    assert similarity > 0.5  # Should be reasonably high for similar scripts

def test_evaluate_script(evaluator, sample_scripts):
    reference, valid_generated, _ = sample_scripts
    results = evaluator.evaluate_script(reference, valid_generated)
    
    assert 'f1_score' in results
    assert 'semantic_similarity' in results
    assert 'executability' in results
    assert 'syntax_validity' in results
    assert 'syntax_errors' in results
    
    assert 0 <= results['f1_score'] <= 1
    assert 0 <= results['semantic_similarity'] <= 1
    assert isinstance(results['executability'], bool)
    assert isinstance(results['syntax_validity'], bool)
    assert isinstance(results['syntax_errors'], list)

def test_evaluate_batch(evaluator, sample_scripts):
    reference, valid_generated, _ = sample_scripts
    references = [reference] * 3
    generated = [valid_generated] * 3
    
    results = evaluator.evaluate_batch(references, generated)
    
    assert 'avg_f1_score' in results
    assert 'avg_semantic_similarity' in results
    assert 'executability_rate' in results
    assert 'syntax_validity_rate' in results
    
    assert all(0 <= v <= 1 for v in results.values()) 