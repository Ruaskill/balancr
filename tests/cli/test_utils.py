import pytest
import logging
from unittest.mock import patch, MagicMock

from balancr.cli import utils


@pytest.fixture
def reset_logging():
    """Reset the logging configuration before and after each test."""
    # Store original handlers
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    yield

    # Reset after test
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Restore original handlers and level
    for handler in original_handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(original_level)


class TestFontMessageFilter:
    """Tests for the FontMessageFilter class."""

    def test_filter_excludes_findfont_messages(self):
        """Test that messages containing 'findfont' are filtered out."""
        filter_instance = utils.FontMessageFilter()

        # Create mock record with 'findfont' in message
        record = MagicMock()
        record.getMessage.return_value = "Some message with findfont included"

        # Should be filtered out (return False)
        assert filter_instance.filter(record) is False

    def test_filter_allows_other_messages(self):
        """Test that messages not containing 'findfont' are allowed."""
        filter_instance = utils.FontMessageFilter()

        # Create mock record without 'findfont'
        record = MagicMock()
        record.getMessage.return_value = "Some normal message"

        # Should be allowed (return True)
        assert filter_instance.filter(record) is True


class TestSetupLogging:
    """Tests for the setup_logging function."""

    @patch("colorama.init")
    @patch("colorama.Fore")
    @patch("colorama.Style")
    def test_setup_logging_verbose(
        self, mock_style, mock_fore, mock_init, reset_logging
    ):
        """Test setup_logging with verbose level."""
        utils.setup_logging("verbose")

        # Check root logger level
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

        # Should have at least one handler
        assert len(root_logger.handlers) > 0

        # Colorama should be initialised
        mock_init.assert_called_once()

    @patch("colorama.init")
    @patch("colorama.Fore")
    @patch("colorama.Style")
    def test_setup_logging_default(
        self, mock_style, mock_fore, mock_init, reset_logging
    ):
        """Test setup_logging with default level."""
        utils.setup_logging("default")

        # Check root logger level
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

        # Should have at least one handler
        assert len(root_logger.handlers) > 0

    @patch("colorama.init")
    @patch("colorama.Fore")
    @patch("colorama.Style")
    def test_setup_logging_quiet(self, mock_style, mock_fore, mock_init, reset_logging):
        """Test setup_logging with quiet level."""
        utils.setup_logging("quiet")

        # Check root logger level
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

        # Should have at least one handler
        assert len(root_logger.handlers) > 0

    @patch("colorama.init")
    @patch("colorama.Fore")
    @patch("colorama.Style")
    def test_setup_logging_invalid_level(
        self, mock_style, mock_fore, mock_init, reset_logging
    ):
        """Test setup_logging with invalid level (should default to INFO)."""
        utils.setup_logging("invalid_level")

        # Should default to INFO
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    @patch("colorama.init")
    @patch("colorama.Fore")
    @patch("colorama.Style")
    def test_setup_logging_clears_existing_handlers(
        self, mock_style, mock_fore, mock_init, reset_logging
    ):
        """Test that setup_logging clears existing handlers."""
        # Add a dummy handler first
        root_logger = logging.getLogger()
        dummy_handler = logging.StreamHandler()
        root_logger.addHandler(dummy_handler)

        original_handler_count = len(root_logger.handlers)
        assert original_handler_count > 0

        # Now set up logging
        utils.setup_logging()

        # The dummy handler should be removed
        assert dummy_handler not in root_logger.handlers

    @patch("colorama.init", side_effect=ImportError("No colorama"))
    @patch("logging.basicConfig")
    def test_setup_logging_without_colorama(
        self, mock_basicConfig, mock_colorama_error, reset_logging
    ):
        """Test setup_logging falls back to basic config if colorama is not available."""
        utils.setup_logging()

        # Should use basicConfig as fallback
        mock_basicConfig.assert_called_once()

        # Check that formatter was used with correct pattern
        call_kwargs = mock_basicConfig.call_args[1]
        assert "format" in call_kwargs
        assert "%(levelname)s: %(message)s" in call_kwargs["format"]

    @patch("colorama.init")
    @patch("colorama.Fore")
    @patch("colorama.Style")
    def test_setup_logging_sets_third_party_levels(
        self, mock_style, mock_fore, mock_init, reset_logging
    ):
        """Test that setup_logging sets higher levels for third-party libraries."""
        utils.setup_logging()

        # Check that matplotlib logger level is set to WARNING
        matplotlib_logger = logging.getLogger("matplotlib")
        assert matplotlib_logger.level == logging.WARNING

        # Check PIL logger
        pil_logger = logging.getLogger("PIL")
        assert pil_logger.level == logging.WARNING

    @patch("colorama.Fore")
    @patch("colorama.Style")
    @patch("colorama.init")
    def test_colored_formatter(self, mock_init, mock_style, mock_fore, reset_logging):
        """Test that ColoredFormatter applies colours to levelname."""
        # Set up mock colours
        mock_fore.RED = "RED-"
        mock_fore.GREEN = "GREEN-"
        mock_fore.BLUE = "BLUE-"
        mock_fore.YELLOW = "YELLOW-"
        mock_style.RESET_ALL = "-RESET"
        mock_style.BRIGHT = "BRIGHT-"

        # Setup logging with a StringIO handler to capture output
        import io

        string_io = io.StringIO()

        utils.setup_logging("verbose")

        # Get the handler that was added to the root logger
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        # Replace the handler's stream with our StringIO
        original_stream = handler.stream
        handler.stream = string_io

        try:
            # Log a test message
            root_logger.error("Test error message")

            # Get the output
            output = string_io.getvalue()

            # Verify colour codes are in the output
            assert "RED-" in output and "-RESET" in output
        finally:
            # Restore original stream
            handler.stream = original_stream
