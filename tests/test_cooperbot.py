import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch, mock_open
from datetime import datetime

# Add the parent directory to the path so we can import the bot
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the bot module (cooperbot-twitch.py)
import importlib.util
spec = importlib.util.spec_from_file_location("cooperbot_twitch", "../cooperbot-twitch.py")
cooperbot_twitch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cooperbot_twitch)
SimpleBot = cooperbot_twitch.SimpleBot


class TestSimpleBot:

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_env(self):
        """Mock environment variables"""
        env_vars = {
            'TWITCH_TOKEN': 'test_token',
            'TWITCH_CHANNEL': 'test_channel',
            'TWITCH_BOT_NAME': 'testbot',
            'OLLAMA_URL': 'http://localhost:11434',
            'OLLAMA_MODEL': 'test_model'
        }
        with patch.dict(os.environ, env_vars):
            yield env_vars

    @pytest.fixture
    def bot(self, mock_env, temp_dir):
        """Create a bot instance for testing"""
        # Create test files
        with open('system_prompt.txt', 'w') as f:
            f.write('Test system prompt')
        with open('ignored_users.txt', 'w') as f:
            f.write('ignoreduser1\nignoreduser2\n# comment\n')

        with patch('twitchio.ext.commands.Bot.__init__'):
            bot = SimpleBot()
            return bot

    def test_load_system_prompt_success(self, temp_dir):
        """Test successful loading of system prompt"""
        test_prompt = "This is a test prompt"
        with open('system_prompt.txt', 'w') as f:
            f.write(test_prompt)

        with patch('twitchio.ext.commands.Bot.__init__'):
            bot = SimpleBot()
            assert bot.system_prompt == test_prompt

    def test_load_system_prompt_file_not_found(self, temp_dir):
        """Test system prompt loading when file doesn't exist"""
        with patch('twitchio.ext.commands.Bot.__init__'):
            bot = SimpleBot()
            assert "helpful Twitch chatbot" in bot.system_prompt

    def test_load_ignored_users_success(self, temp_dir):
        """Test successful loading of ignored users"""
        with open('ignored_users.txt', 'w') as f:
            f.write('user1\nUser2\n# This is a comment\n\nuser3\n')

        with patch('twitchio.ext.commands.Bot.__init__'):
            bot = SimpleBot()
            expected_users = {'user1', 'user2', 'user3'}
            assert bot.ignored_users == expected_users

    def test_load_ignored_users_file_not_found(self, temp_dir):
        """Test ignored users loading when file doesn't exist"""
        with patch('twitchio.ext.commands.Bot.__init__'):
            bot = SimpleBot()
            assert bot.ignored_users == set()

    def test_init_log_file(self, bot, temp_dir):
        """Test log file initialization"""
        assert os.path.exists(bot.log_filename)
        with open(bot.log_filename, 'r') as f:
            content = f.read()
            assert "Twitch Bot Log Started" in content
            assert "test_channel" in content
            assert "testbot" in content

    def test_log_interaction(self, bot):
        """Test interaction logging"""
        bot.log_interaction("TEST", "testuser", "test message", "test response")

        with open(bot.log_filename, 'r') as f:
            content = f.read()
            assert "TEST" in content
            assert "testuser" in content
            assert "test message" in content
            assert "test response" in content

    @pytest.mark.asyncio
    async def test_event_message_ignores_echo(self, bot):
        """Test that bot ignores its own messages"""
        mock_message = Mock()
        mock_message.echo = True

        with patch.object(bot, 'handle_mention') as mock_handle:
            await bot.event_message(mock_message)
            mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_message_ignores_ignored_users(self, bot):
        """Test that bot ignores messages from ignored users"""
        mock_message = Mock()
        mock_message.echo = False
        mock_message.author.name = 'ignoreduser1'
        mock_message.content = '@testbot hello'

        with patch.object(bot, 'handle_mention') as mock_handle:
            with patch.object(bot, 'log_interaction') as mock_log:
                await bot.event_message(mock_message)
                mock_handle.assert_not_called()
                mock_log.assert_called_once_with("IGNORED USER ATTEMPT", "ignoreduser1", "@testbot hello")

    @pytest.mark.asyncio
    async def test_event_message_handles_mention(self, bot):
        """Test that bot handles mentions from non-ignored users"""
        mock_message = Mock()
        mock_message.echo = False
        mock_message.author.name = 'normaluser'
        mock_message.content = '@testbot hello'

        with patch.object(bot, 'handle_mention') as mock_handle:
            await bot.event_message(mock_message)
            mock_handle.assert_called_once_with(mock_message)

    @pytest.mark.asyncio
    async def test_call_ollama_success(self, bot):
        """Test successful Ollama API call"""
        mock_response_data = {"response": "Test response from AI"}

        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await bot.call_ollama("test message", "testuser")
            assert result == "Test response from AI"

    @pytest.mark.asyncio
    async def test_call_ollama_api_error(self, bot):
        """Test Ollama API error handling"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await bot.call_ollama("test message", "testuser")
            assert result is None

    @pytest.mark.asyncio
    async def test_handle_mention_success(self, bot):
        """Test successful mention handling"""
        mock_message = Mock()
        mock_message.author.name = 'testuser'
        mock_message.content = '@testbot hello'
        mock_message.channel.send = AsyncMock()

        with patch.object(bot, 'call_ollama', return_value="Hello back!"):
            with patch.object(bot, 'log_interaction') as mock_log:
                await bot.handle_mention(mock_message)

                mock_message.channel.send.assert_called_once_with("@testuser Hello back!")
                mock_log.assert_called_once_with(
                    "SUCCESSFUL RESPONSE", 
                    "testuser", 
                    "@testbot hello", 
                    "Hello back!"
                )

    @pytest.mark.asyncio
    async def test_handle_mention_long_response(self, bot):
        """Test mention handling with long response truncation"""
        mock_message = Mock()
        mock_message.author.name = 'testuser'
        mock_message.content = '@testbot hello'
        mock_message.channel.send = AsyncMock()

        long_response = "x" * 600  # Response longer than 500 characters
        expected_response = "x" * 497 + "..."

        with patch.object(bot, 'call_ollama', return_value=long_response):
            await bot.handle_mention(mock_message)
            mock_message.channel.send.assert_called_once_with(f"@testuser {expected_response}")

    @pytest.mark.asyncio
    async def test_handle_mention_no_response(self, bot):
        """Test mention handling when AI doesn't generate response"""
        mock_message = Mock()
        mock_message.author.name = 'testuser'
        mock_message.content = '@testbot hello'
        mock_message.channel.send = AsyncMock()

        with patch.object(bot, 'call_ollama', return_value=None):
            with patch.object(bot, 'log_interaction') as mock_log:
                await bot.handle_mention(mock_message)

                mock_message.channel.send.assert_called_once_with(
                    "@testuser Sorry, I couldn't generate a response!"
                )
                mock_log.assert_called_once_with(
                    "FAILED RESPONSE",
                    "testuser",
                    "@testbot hello",
                    "Sorry, I couldn't generate a response!"
                )

    @pytest.mark.asyncio
    async def test_handle_mention_exception(self, bot):
        """Test mention handling when exception occurs"""
        mock_message = Mock()
        mock_message.author.name = 'testuser'
        mock_message.content = '@testbot hello'
        mock_message.channel.send = AsyncMock()

        with patch.object(bot, 'call_ollama', side_effect=Exception("Test error")):
            with patch.object(bot, 'log_interaction') as mock_log:
                await bot.handle_mention(mock_message)

                mock_message.channel.send.assert_called_once_with(
                    "@testuser Oops, something went wrong!"
                )
                mock_log.assert_called_once_with(
                    "ERROR",
                    "testuser", 
                    "@testbot hello",
                    "Error: Test error"
                )


if __name__ == '__main__':
    pytest.main([__file__])
