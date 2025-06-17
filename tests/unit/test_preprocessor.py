import pytest
import os
import tempfile
from scripts.preprocess import LAMMPSDataProcessor

@pytest.fixture
def processor():
    return LAMMPSDataProcessor()

@pytest.fixture
def sample_lammps_code():
    return """
    units metal
    atom_style atomic
    boundary p p p
    pair_style eam
    fix 1 all nvt temp 300.0 300.0 0.1
    thermo 100
    timestep 1.0
    run 10000
    """

@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some sample LAMMPS files
        for i in range(3):
            with open(os.path.join(temp_dir, f"test_{i}.in"), "w") as f:
                f.write(sample_lammps_code())
        yield temp_dir

def test_generate_prompt_from_code(processor, sample_lammps_code):
    prompt = processor.generate_prompt_from_code(sample_lammps_code)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "LAMMPS input script" in prompt
    assert "metal" in prompt
    assert "atomic" in prompt
    assert "NVT" in prompt

def test_extract_simulation_params(processor, sample_lammps_code):
    params = processor._extract_simulation_params(sample_lammps_code)
    assert isinstance(params, dict)
    assert params['units'] == 'metal'
    assert params['atom_style'] == 'atomic'
    assert params['ensemble'] == 'NVT'
    assert params['boundary_conditions'] == 'p p p'

def test_load_lammps_scripts(processor, temp_data_dir):
    dataset = processor.load_lammps_scripts(temp_data_dir)
    assert isinstance(dataset, list)
    assert len(dataset) == 3
    for prompt, code in dataset:
        assert isinstance(prompt, str)
        assert isinstance(code, str)
        assert len(prompt) > 0
        assert len(code) > 0

def test_load_lammps_scripts_with_num_samples(processor, temp_data_dir):
    dataset = processor.load_lammps_scripts(temp_data_dir, num_samples=2)
    assert len(dataset) == 2

def test_save_and_load_dataset(processor, temp_data_dir):
    # Create a dataset
    dataset = processor.load_lammps_scripts(temp_data_dir)
    
    # Save it
    output_path = os.path.join(temp_data_dir, "dataset.csv")
    processor.save_dataset(dataset, output_path)
    assert os.path.exists(output_path)
    
    # Load it back
    loaded_dataset = processor.load_dataset(output_path)
    assert len(loaded_dataset) == len(dataset)
    for (p1, c1), (p2, c2) in zip(dataset, loaded_dataset):
        assert p1 == p2
        assert c1 == c2

def test_load_lammps_scripts_nonexistent_dir(processor):
    with pytest.raises(FileNotFoundError):
        processor.load_lammps_scripts("nonexistent_dir")

def test_load_dataset_nonexistent_file(processor):
    with pytest.raises(FileNotFoundError):
        processor.load_dataset("nonexistent_file.csv") 