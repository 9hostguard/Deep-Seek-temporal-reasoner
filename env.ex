-- Environment configuration for Deep-Seek-temporal-reasoner
-- This file contains environment variable definitions for the application

-- DeepSeek API Configuration
constant DEEPSEEK_API_KEY = "your-key-here"
constant DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
constant DEEPSEEK_MODEL = "deepseek-chat"

-- Application Configuration
constant APP_NAME = "Deep-Seek-temporal-reasoner"
constant APP_VERSION = "1.0.0"
constant DEBUG_MODE = 1

-- Server Configuration
constant HOST = "localhost"
constant PORT = 8000

-- Temporal Processing Configuration
constant MAX_PROMPT_LENGTH = 4096
constant ENABLE_PAST_ANALYSIS = 1
constant ENABLE_PRESENT_ANALYSIS = 1
constant ENABLE_FUTURE_ANALYSIS = 1
