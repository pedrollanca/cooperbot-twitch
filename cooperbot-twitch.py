import asyncio
import os
import aiohttp
import json
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

    async def event_ready(self):
        print(f'Bot connected as {self.nick}')
        print(f'Watching channel: {self.channel_name}')
        print(f'Using Ollama model: {self.ollama_model}')

    async def event_message(self, message):
        # Ignore messages from the bot itself
        if message.echo:
            return

        # Check if bot is mentioned
        content = message.content.lower()
        if self.bot_name in content or f'@{self.bot_name}' in content:
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
            else:
                await message.channel.send(f"@{message.author.name} Sorry, I couldn't generate a response!")

        except Exception as e:
            print(f"Error handling mention: {e}")
            await message.channel.send(f"@{message.author.name} Oops, something went wrong!")

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
