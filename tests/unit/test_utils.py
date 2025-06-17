import pytest
import os
import json
import tempfile
from scripts.utils import LLMInterface, ExperimentLogger

@pytest.fixture
def temp_config():
    config = {
        "models": {
            "gpt-4": {
                "name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000
            }
        },
        "experiment": {
            "num_samples": 10,
            "batch_size": 1
        },
        "paths": {
            "log_file": "test.log",
            "results_file": "test_results.csv"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def llm_interface(temp_config):
    return LLMInterface(config_path=temp_config)

@pytest.fixture
def experiment_logger(temp_config):
    return ExperimentLogger(config_path=temp_config)

def test_llm_interface_initialization(llm_interface):
    assert llm_interface is not None
    assert hasattr(llm_interface, 'config')
    assert 'models' in llm_interface.config
    assert 'experiment' in llm_interface.config

def test_experiment_logger_initialization(experiment_logger):
    assert experiment_logger is not None
    assert hasattr(experiment_logger, 'config')
    assert 'paths' in experiment_logger.config

def test_log_experiment_start(experiment_logger):
    experiment_name = "test_experiment"
    params = {"param1": "value1", "param2": 2}
    
    experiment_logger.log_experiment_start(experiment_name, params)
    
    # Verify log file exists and contains the expected content
    assert os.path.exists(experiment_logger.config['paths']['log_file'])
    with open(experiment_logger.config['paths']['log_file'], 'r') as f:
        log_content = f.read()
        assert experiment_name in log_content
        assert "param1" in log_content
        assert "value1" in log_content

def test_log_experiment_end(experiment_logger):
    experiment_name = "test_experiment"
    results = {"metric1": 0.95, "metric2": 0.85}
    
    experiment_logger.log_experiment_end(experiment_name, results)
    
    # Verify log file contains the results
    with open(experiment_logger.config['paths']['log_file'], 'r') as f:
        log_content = f.read()
        assert experiment_name in log_content
        assert "metric1" in log_content
        assert "0.95" in log_content

def test_log_error(experiment_logger):
    error_msg = "Test error message"
    context = {"error_type": "ValueError", "details": "Invalid input"}
    
    experiment_logger.log_error(error_msg, context)
    
    # Verify error is logged with context
    with open(experiment_logger.config['paths']['log_file'], 'r') as f:
        log_content = f.read()
        assert error_msg in log_content
        assert "error_type" in log_content
        assert "ValueError" in log_content

def test_log_metric(experiment_logger):
    metric_name = "accuracy"
    value = 0.95
    context = {"model": "gpt-4", "dataset": "test"}
    
    experiment_logger.log_metric(metric_name, value, context)
    
    # Verify metric is logged with context
    with open(experiment_logger.config['paths']['log_file'], 'r') as f:
        log_content = f.read()
        assert metric_name in log_content
        assert "0.95" in log_content
        assert "model" in log_content
        assert "gpt-4" in log_content

@pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not set")
def test_call_llm(llm_interface):
    prompt = "Generate a simple LAMMPS script for a molecular dynamics simulation"
    response = llm_interface.call_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not set")
def test_batch_call_llm(llm_interface):
    prompts = [
        "Generate a simple LAMMPS script",
        "Generate a LAMMPS script with NVT ensemble"
    ]
    responses = llm_interface.batch_call_llm(prompts, batch_size=1)
    assert len(responses) == len(prompts)
    assert all(isinstance(r, str) for r in responses)
    assert all(len(r) > 0 for r in responses) 