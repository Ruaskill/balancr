import pytest
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock

from balancr.cli import main


@pytest.fixture
def mock_args():
    """Create mock parsed arguments."""
    args = MagicMock()
    args.verbose = False
    args.quiet = False
    args.config_path = Path.home() / ".balancr" / "config.json"
    args.command = "test-command"
    args.func = MagicMock(return_value=0)
    return args


class TestCreateParser:
    """Tests for create_parser function."""

    def test_create_parser_returns_parser(self):
        """Test create_parser returns an ArgumentParser instance."""
        parser = main.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_has_required_global_arguments(self):
        """Test parser has all required global arguments."""
        parser = main.create_parser()

        # Get all option strings from parser arguments
        options = []
        for action in parser._actions:
            options.extend(action.option_strings)

        # Check required global options
        assert "--version" in options
        assert "--verbose" in options or "-v" in options
        assert "--quiet" in options or "-q" in options
        assert "--config-path" in options

    def test_parser_has_essential_commands(self):
        """Test parser has essential command subparsers."""
        parser = main.create_parser()

        # Check for subset of important commands
        essential_commands = ["load-data", "select-techniques", "run", "reset"]

        for cmd in essential_commands:
            # Find the subparser for this command
            subparsers = None
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    subparsers = action
                    break

            assert subparsers is not None, "No subparsers found in the parser"
            assert cmd in subparsers.choices, f"Command '{cmd}' not found in parser"


class TestCommandRegistrationSample:
    """Sample tests for command registration - testing a representative subset."""

    def test_register_load_data_command(self):
        """Test register_load_data_command sets up correct parser."""
        # Create mock for the subparsers object
        mock_subparsers = MagicMock()
        mock_subparser = MagicMock()
        mock_subparsers.add_parser.return_value = mock_subparser

        # Call registration function
        main.register_load_data_command(mock_subparsers)

        # Verify it created the right parser
        mock_subparsers.add_parser.assert_called_once()
        args, _ = mock_subparsers.add_parser.call_args
        assert args[0] == "load-data"

        # Check essential arguments were added
        target_column_added = False
        for call in mock_subparser.add_argument.call_args_list:
            args, kwargs = call
            if "--target-column" in args or "-t" in args:
                target_column_added = True
                assert kwargs.get("required", False) is True

        assert target_column_added, "Required --target-column argument not added"

    def test_register_select_techniques_command(self):
        """Test register_select_techniques_command sets up correct parser."""
        # Create mock for the subparsers object
        mock_subparsers = MagicMock()
        mock_subparser = MagicMock()
        mock_subparsers.add_parser.return_value = mock_subparser

        # Mock the mutually exclusive group
        mock_group = MagicMock()
        mock_subparser.add_mutually_exclusive_group.return_value = mock_group

        # Call registration function
        main.register_select_techniques_command(mock_subparsers)

        # Verify a mutually exclusive group was created
        mock_subparser.add_mutually_exclusive_group.assert_called_once()

        # Verify it was created as required
        _, kwargs = mock_subparser.add_mutually_exclusive_group.call_args
        assert kwargs.get("required", False) is True


@patch("balancr.cli.main.create_parser")
@patch("balancr.cli.utils.setup_logging")
@patch("balancr.cli.config.initialise_config")
class TestMainFunction:
    """Essential tests for the main function."""

    def test_main_no_command(
        self, mock_init_config, mock_setup_logging, mock_create_parser
    ):
        """Test main when no command is provided."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.command = None
        parser = MagicMock()
        parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = parser

        # Main call
        result = main.main()

        # Verify logging was set up
        mock_setup_logging.assert_called_once()

        # Verify help was displayed
        parser.print_help.assert_called_once()

        # Verify result
        assert result == 0

    def test_main_with_command(
        self, mock_init_config, mock_setup_logging, mock_create_parser
    ):
        """Test main when command is provided."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.verbose = False
        mock_args.quiet = False
        mock_args.command = "test-command"
        mock_args.func = MagicMock(return_value=0)
        mock_args.config_path = Path.home() / ".balancr" / "config.json"

        parser = MagicMock()
        parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = parser

        # Make Path.exists() return True to skip initialisation
        with patch.object(Path, "exists", return_value=True):
            # Call main
            result = main.main()

            # Verify command function was called
            mock_args.func.assert_called_once_with(mock_args)

            # Verify result
            assert result == 0

    def test_main_with_error(
        self, mock_init_config, mock_setup_logging, mock_create_parser
    ):
        """Test main handles command errors."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.verbose = False
        mock_args.quiet = False
        mock_args.command = "test-command"
        mock_args.func = MagicMock(side_effect=Exception("Test error"))
        mock_args.config_path = Path.home() / ".balancr" / "config.json"

        parser = MagicMock()
        parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = parser

        # Mock logging to avoid test output pollution
        with patch("logging.error") as mock_error:
            # Call main
            result = main.main()

            # Verify error was logged
            mock_error.assert_called_once()

            # Verify result indicates error
            assert result == 1


class TestLoggingConfig:
    """Tests for logging configuration in main."""

    @patch("balancr.cli.main.create_parser")
    @patch("balancr.cli.utils.setup_logging")
    @patch("balancr.cli.config.initialise_config")
    def test_main_sets_up_verbose_logging(
        self, mock_init_config, mock_setup_logging, mock_create_parser
    ):
        """Test main sets up verbose logging when requested."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.verbose = True
        mock_args.quiet = False
        mock_args.command = "test-command"
        mock_args.func = MagicMock(return_value=0)
        mock_args.config_path = Path.home() / ".balancr" / "config.json"

        parser = MagicMock()
        parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = parser

        # Make Path.exists() return True to skip initialisation
        with patch.object(Path, "exists", return_value=True):
            # Call main
            main.main()

            # Verify logging was set up with verbose level
            mock_setup_logging.assert_called_once_with("verbose")

    @patch("balancr.cli.main.create_parser")
    @patch("balancr.cli.utils.setup_logging")
    @patch("balancr.cli.config.initialise_config")
    def test_main_sets_up_quiet_logging(
        self, mock_init_config, mock_setup_logging, mock_create_parser
    ):
        """Test main sets up quiet logging when requested."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.verbose = False
        mock_args.quiet = True
        mock_args.command = "test-command"
        mock_args.func = MagicMock(return_value=0)
        mock_args.config_path = Path.home() / ".balancr" / "config.json"

        parser = MagicMock()
        parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = parser

        # Make Path.exists() return True to skip initialisation
        with patch.object(Path, "exists", return_value=True):
            # Call main
            main.main()

            # Verify logging was set up with quiet level
            mock_setup_logging.assert_called_once_with("quiet")


class TestConfigInitialisation:
    """Tests for config initialisation in main."""

    @patch("balancr.cli.main.create_parser")
    @patch("balancr.cli.utils.setup_logging")
    @patch("balancr.cli.config.initialise_config")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_main_creates_config_dir(
        self,
        mock_mkdir,
        mock_exists,
        mock_init_config,
        mock_setup_logging,
        mock_create_parser,
    ):
        """Test main creates config directory if needed."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.verbose = False
        mock_args.quiet = False
        mock_args.command = "test-command"
        mock_args.func = MagicMock(return_value=0)
        mock_args.config_path = Path("/path/to/config.json")

        parser = MagicMock()
        parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = parser

        # Config file doesn't exist
        mock_exists.return_value = False

        # Call main
        main.main()

        # Verify directory was created
        mock_mkdir.assert_called_once()
        args, kwargs = mock_mkdir.call_args
        assert kwargs.get("parents", False) is True
        assert kwargs.get("exist_ok", False) is True

        # Verify config was initialised
        mock_init_config.assert_called_once_with(mock_args.config_path)

    @patch("balancr.cli.main.create_parser")
    @patch("balancr.cli.utils.setup_logging")
    @patch("balancr.cli.config.initialise_config")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_main_skips_config_init_if_exists(
        self,
        mock_mkdir,
        mock_exists,
        mock_init_config,
        mock_setup_logging,
        mock_create_parser,
    ):
        """Test main doesn't initialise config if it already exists."""
        # Set up mocks
        mock_args = MagicMock()
        mock_args.verbose = False
        mock_args.quiet = False
        mock_args.command = "test-command"
        mock_args.func = MagicMock(return_value=0)
        mock_args.config_path = Path("/path/to/config.json")

        parser = MagicMock()
        parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = parser

        # Config file exists
        mock_exists.return_value = True

        # Call main
        main.main()

        # Verify directory was still created
        mock_mkdir.assert_called_once()

        # Verify config was not initialised
        mock_init_config.assert_not_called()


class TestCommandLineArgParsing:
    """Test the CLI can parse real command-line arguments for key commands."""

    def test_load_data_arg_parsing(self):
        """Test load-data command argument parsing."""
        parser = main.create_parser()

        # Parse with arguments
        args = parser.parse_args(["load-data", "file.csv", "--target-column", "target"])

        # Check parsed values
        assert args.command == "load-data"
        assert args.file_path == "file.csv"
        assert args.target_column == "target"
        assert callable(args.func)

    def test_run_command_arg_parsing(self):
        """Test run command argument parsing."""
        parser = main.create_parser()

        # Parse with arguments
        args = parser.parse_args(["run", "--output-dir", "results/experiment1"])

        # Check parsed values
        assert args.command == "run"
        assert args.output_dir == "results/experiment1"
        assert callable(args.func)
