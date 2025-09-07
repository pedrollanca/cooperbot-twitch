import asyncio
import os
import aiohttp
import json
from datetime import datetime
from dotenv import load_dotenv
from twitchio.ext import commands

# Load environment variables
load_dotenv()

class SimpleBot(commands.Bot):
    def __init__(self):
        # Get config from environment
        self.channel_name = os.getenv('TWITCH_CHANNEL')
        self.bot_name = os.getenv('TWITCH_BOT_NAME', 'cooperbot').lower()
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')

        # Load system prompt from file
        self.system_prompt = self.load_system_prompt()

        # Load ignored users list
        self.ignored_users = self.load_ignored_users()

        # Initialize logging
        self.log_filename = f"bot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.init_log_file()

        # Initialize bot
        super().__init__(
            token=os.getenv('TWITCH_TOKEN'),
            prefix='!',
            initial_channels=[self.channel_name]
        )

    def load_system_prompt(self):
        """Load system prompt from system_prompt.txt file"""
        try:
            with open('system_prompt.txt', 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print("Warning: system_prompt.txt not found, using default prompt")
            return """You are a helpful Twitch chatbot. Keep responses short and friendly (under 500 characters)."""
        except Exception as e:
            print(f"Error reading system_prompt.txt: {e}")
            return """You are a helpful Twitch chatbot. Keep responses short and friendly (under 500 characters)."""

    def load_ignored_users(self):
        """Load ignored users from ignored_users.txt file"""
        try:
            with open('ignored_users.txt', 'r', encoding='utf-8') as f:
                users = []
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        users.append(line.lower())
                return set(users)
        except FileNotFoundError:
            print("Info: ignored_users.txt not found, no users will be ignored")
            return set()
        except Exception as e:
            print(f"Error reading ignored_users.txt: {e}")
            return set()

    def init_log_file(self):
        """Initialize the log file with header information"""
        try:
            with open(self.log_filename, 'w', encoding='utf-8') as f:
                f.write(f"=== Twitch Bot Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"Channel: {self.channel_name}\n")
                f.write(f"Bot Name: {self.bot_name}\n")
                f.write(f"Ollama Model: {self.ollama_model}\n")
                f.write("=" * 50 + "\n\n")
            print(f"Log file created: {self.log_filename}")
        except Exception as e:
            print(f"Error creating log file: {e}")

    def log_interaction(self, interaction_type, username, user_message, bot_response=None):
        """Log an interaction to the log file"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_filename, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {interaction_type}\n")
                f.write(f"User: {username}\n")
                f.write(f"Message: {user_message}\n")
                if bot_response:
                    f.write(f"Bot Response: {bot_response}\n")
                f.write("-" * 30 + "\n\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")

    async def event_ready(self):
        print(f'Bot connected as {self.nick}')
        print(f'Watching channel: {self.channel_name}')
        print(f'Using Ollama model: {self.ollama_model}')

    async def event_message(self, message):
        # Ignore messages from the bot itself
        if message.echo:
            return

        # Check if bot is mentioned first to determine if we should log
        content = message.content.lower()
        bot_mentioned = self.bot_name in content or f'@{self.bot_name}' in content

        # Ignore messages from users in the ignored list
        if message.author.name.lower() in self.ignored_users:
            if bot_mentioned:
                print(f"Ignored user {message.author.name} tried to trigger bot with message: {message.content}")
                self.log_interaction("IGNORED USER ATTEMPT", message.author.name, message.content)
            return

        # Handle bot mentions
        if bot_mentioned:
            await self.handle_mention(message)

    async def handle_mention(self, message):
        try:
            print(f"Generating response for: {message.author.name}: {message.content}")

            # Call Ollama API
            response_text = await self.call_ollama(message.content, message.author.name)

            if response_text:
                # Keep response under 500 characters for Twitch
                if len(response_text) > 500:
                    response_text = response_text[:497] + "..."

                await message.channel.send(f"@{message.author.name} {response_text}")
                print(f"Response sent: {response_text}")
                self.log_interaction("SUCCESSFUL RESPONSE", message.author.name, message.content, response_text)
            else:
                error_msg = "Sorry, I couldn't generate a response!"
                await message.channel.send(f"@{message.author.name} {error_msg}")
                self.log_interaction("FAILED RESPONSE", message.author.name, message.content, error_msg)

        except Exception as e:
            print(f"Error handling mention: {e}")
            error_msg = "Oops, something went wrong!"
            await message.channel.send(f"@{message.author.name} {error_msg}")
            self.log_interaction("ERROR", message.author.name, message.content, f"Error: {str(e)}")

    async def call_ollama(self, user_message, username):
        try:
            # Use system prompt from file with user context
            full_prompt = f"""{self.system_prompt}

The user {username} said: {user_message}
Respond naturally as if you're chatting in Twitch."""

            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', '').strip()
                    else:
                        print(f"Ollama API error: {response.status}")
                        return None

        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None

if __name__ == "__main__":
    bot = SimpleBot()
    bot.run()
