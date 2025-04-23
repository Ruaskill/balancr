import pytest
import json
from unittest.mock import patch

from balancr.cli import config


@pytest.fixture
def temp_config_path(tmp_path):
    """Create a temporary path for test configuration files."""
    return tmp_path / "test_config.json"


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        "preprocessing": {
            "handle_missing": "mean",
            "scale": "standard",
            "encode": "auto",
        },
        "evaluation": {
            "test_size": 0.2,
            "cross_validation": 5,
            "random_state": 42,
        },
        "output": {
            "metrics": ["precision", "recall", "f1"],
            "visualisations": ["all"],
        },
    }


def test_initialise_config_new_file(temp_config_path):
    """Test initialising a new configuration file."""
    # Ensure file doesn't exist initially
    assert not temp_config_path.exists()

    # Initialise configuration
    config.initialise_config(temp_config_path)

    # Verify file was created
    assert temp_config_path.exists()

    # Check content
    with open(temp_config_path, "r") as f:
        saved_config = json.load(f)

    # Verify required settings are present
    assert "preprocessing" in saved_config
    assert "evaluation" in saved_config
    assert "output" in saved_config


def test_initialise_config_existing_file(temp_config_path, sample_config):
    """Test initialise_config when file already exists."""
    # Create existing config file
    with open(temp_config_path, "w") as f:
        json.dump(sample_config, f)

    # Try to initialise without force flag
    config.initialise_config(temp_config_path, force=False)

    # Verify content wasn't changed
    with open(temp_config_path, "r") as f:
        saved_config = json.load(f)

    assert saved_config == sample_config

    # Try with force flag
    config.initialise_config(temp_config_path, force=True)

    # Verify content was overwritten
    with open(temp_config_path, "r") as f:
        saved_config = json.load(f)

    assert saved_config != sample_config
    assert "preprocessing" in saved_config  # Should have default config now


@patch("logging.info")
def test_initialise_config_logs_message(mock_info, temp_config_path):
    """Test that initialise_config logs appropriate messages."""
    config.initialise_config(temp_config_path)

    # Check log message
    mock_info.assert_called_once()
    assert "Initialised configuration file" in mock_info.call_args[0][0]


def test_load_config(temp_config_path, sample_config):
    """Test loading configuration from a file."""
    # Create config file with known content
    with open(temp_config_path, "w") as f:
        json.dump(sample_config, f)

    # Load configuration
    loaded_config = config.load_config(temp_config_path)

    # Verify content matches
    assert loaded_config == sample_config


def test_load_config_file_not_found():
    """Test load_config raises FileNotFoundError when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        config.load_config("non_existent_file.json")


def test_load_config_invalid_json(temp_config_path):
    """Test load_config handling of invalid JSON files."""
    # Create file with invalid JSON
    with open(temp_config_path, "w") as f:
        f.write("This is not valid JSON")

    # Attempt to load should raise JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        config.load_config(temp_config_path)


def test_update_config_existing_file(temp_config_path, sample_config):
    """Test updating an existing configuration file."""
    # Create existing config
    with open(temp_config_path, "w") as f:
        json.dump(sample_config, f)

    # Update a setting
    new_settings = {"preprocessing": {"scale": "minmax"}}
    config.update_config(temp_config_path, new_settings)

    # Verify it was updated
    with open(temp_config_path, "r") as f:
        updated_config = json.load(f)

    assert updated_config["preprocessing"]["scale"] == "minmax"
    # Other settings should remain unchanged
    assert updated_config["preprocessing"]["handle_missing"] == "mean"


def test_update_config_new_file(temp_config_path):
    """Test update_config when file doesn't exist yet."""
    # Ensure file doesn't exist
    if temp_config_path.exists():
        temp_config_path.unlink()

    # Update a setting (should create the file with default + updates)
    new_settings = {"preprocessing": {"scale": "minmax"}}
    config.update_config(temp_config_path, new_settings)

    # Verify file was created with settings
    with open(temp_config_path, "r") as f:
        updated_config = json.load(f)

    assert updated_config["preprocessing"]["scale"] == "minmax"
    # Should also have other default settings
    assert "evaluation" in updated_config


def test_update_config_nested_settings(temp_config_path, sample_config):
    """Test updating deeply nested settings."""
    # Create existing config
    with open(temp_config_path, "w") as f:
        json.dump(sample_config, f)

    # Update a nested setting and add a new one
    new_settings = {
        "preprocessing": {"new_option": "value"},
        "new_section": {"key": "value"},
    }
    config.update_config(temp_config_path, new_settings)

    # Verify updates
    with open(temp_config_path, "r") as f:
        updated_config = json.load(f)

    assert updated_config["preprocessing"]["new_option"] == "value"
    assert updated_config["new_section"]["key"] == "value"
    # Original settings should be preserved
    assert updated_config["preprocessing"]["handle_missing"] == "mean"


def test_deep_update():
    """Test deep_update function for merging dictionaries."""
    original = {
        "a": 1,
        "b": {"c": 2, "d": 3},
        "e": {"f": {"g": 4}},
    }
    update = {
        "b": {"c": 5, "new": 6},
        "e": {"f": {"h": 7}},
        "i": 8,
    }

    result = config.deep_update(original, update)

    # Should be the same object (modified in place)
    assert result is original

    # Check values
    assert original["a"] == 1  # Unchanged
    assert original["b"]["c"] == 5  # Updated
    assert original["b"]["d"] == 3  # Unchanged
    assert original["b"]["new"] == 6  # Added
    assert original["e"]["f"]["g"] == 4  # Unchanged
    assert original["e"]["f"]["h"] == 7  # Added
    assert original["i"] == 8  # Added


@patch("builtins.print")
@patch("balancr.cli.config.load_config")
def test_print_config_success(mock_load_config, mock_print, sample_config):
    """Test successful printing of configuration."""
    mock_load_config.return_value = sample_config

    # Call the function
    exit_code = config.print_config("some_path.json")

    # Verify
    assert exit_code == 0
    mock_load_config.assert_called_once_with("some_path.json")
    assert mock_print.call_count >= 2  # At least 2 print calls


@patch("logging.error")
@patch("balancr.cli.config.load_config")
def test_print_config_error(mock_load_config, mock_error):
    """Test error handling in print_config."""
    # Make load_config raise an exception
    mock_load_config.side_effect = Exception("Test error")

    # Call the function
    exit_code = config.print_config("some_path.json")

    # Verify
    assert exit_code == 1
    mock_error.assert_called_once()
    assert "Test error" in mock_error.call_args[0][0]
