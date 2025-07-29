# Deep-Seek-temporal-reasoner

This project demonstrates 4D augmentation for LLM using Deep Seek models with enhanced security, reliability, and comprehensive error handling.

## 🌟 Features

### Core Capabilities
- **Avatar Consciousness Management**: Advanced AI consciousness simulation with temporal reasoning
- **Real-time Personality Evolution**: Dynamic personality development and adaptation
- **Quantum Visual Rendering**: Advanced visual representation with holographic displays
- **Temporal Memory Operations**: Sophisticated memory management across time dimensions
- **Expression Engine**: Emotional expression generation and control
- **WebSocket Streaming**: Real-time communication and interaction

### 🔐 Security & Reliability (Enhanced)
- **AES-256-GCM Encryption**: Enterprise-grade encryption for all sensitive data
- **Automatic Key Rotation**: Scheduled key rotation with configurable intervals
- **Secure Storage**: Encrypted file storage with metadata protection
- **Comprehensive Error Handling**: Circuit breakers, retry mechanisms, and structured logging
- **Performance Monitoring**: Real-time metrics and performance tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- FastAPI framework
- Cryptography library

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/9hostguard/Deep-Seek-temporal-reasoner.git
cd Deep-Seek-temporal-reasoner
```

2. **Install dependencies**:
```bash
pip install -r requirements
```

3. **Start the GUPPIE Avatar API server**:
```bash
python guppie_server.py
```

The API will be available at `http://localhost:8000`

### 📋 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 🧠 Core Components

### Avatar Management
- **Create Avatar**: Initialize new avatar consciousness with customizable parameters
- **Consciousness Evolution**: Real-time consciousness development and sentience tracking
- **Personality Matrix**: Multi-dimensional personality trait management
- **Memory Systems**: Temporal memory storage with importance weighting

### Visual & Expression
- **Quantum Renderer**: Advanced visual frame generation with holographic capabilities
- **Expression Engine**: Emotional state management and expression generation
- **Style Transformer**: Dynamic visual style transformation with preset management

### 🔐 Security Features

#### Encryption System
```python
from guppie.security.encryption import get_encryption_system

# Get global encryption instance
encryption = get_encryption_system()

# Encrypt sensitive data
encrypted_data = encryption.encrypt_string("Sensitive avatar data")

# Decrypt when needed
decrypted_data = encryption.decrypt_string(encrypted_data)
```

#### Key Rotation
```python
from guppie.security.encryption import get_key_manager

# Get key manager
key_manager = get_key_manager()

# Manual key rotation
new_key_id = key_manager.rotate_key()

# Start automatic rotation (every 24 hours by default)
await key_manager.start_auto_rotation()
```

#### Secure Storage
```python
from guppie.security.encryption import SecureStorage, get_encryption_system

# Initialize secure storage
storage = SecureStorage(get_encryption_system())

# Store encrypted file
storage_id = await storage.store_file("avatar_data.json", file_data)

# Retrieve decrypted file
filename, data, metadata = await storage.retrieve_file(storage_id)
```

### 📊 Logging & Error Handling

#### Structured Logging
```python
from guppie.utils.logging import get_logger

# Get logger instance
logger = get_logger("my_component")

# Log operations
logger.log_operation("avatar_creation", avatar_id="test123")

# Log errors with context
try:
    risky_operation()
except Exception as e:
    logger.log_error(e, "Failed during avatar creation")

# Log performance metrics
logger.log_performance("consciousness_evolution", duration_seconds)
```

#### Error Handling Decorators
```python
from guppie.utils.logging import error_handler, RetryHandler, CircuitBreaker

# Basic error handling
@error_handler("avatar_processing", log_performance=True)
async def process_avatar(avatar_id):
    # Your code here
    pass

# Retry with exponential backoff
@RetryHandler(max_retries=3, base_delay=1.0)
def unreliable_operation():
    # Your code here
    pass

# Circuit breaker pattern
@CircuitBreaker(failure_threshold=5, recovery_timeout=60)
def external_service_call():
    # Your code here
    pass
```

## 🧪 Testing

### Run All Tests
```bash
# Run complete test suite with coverage
python -m pytest tests/ -v --cov=guppie

# Run specific test categories
python -m pytest tests/test_guppie_complete_system.py -v
python -m pytest tests/test_security_and_logging.py -v
```

### Test Coverage
The project aims for 80%+ test coverage across all critical pathways:
- Avatar consciousness creation and evolution
- Encryption and decryption operations
- Key rotation functionality
- Error handling and retry mechanisms
- WebSocket streaming operations

## 📈 Performance & Monitoring

### Metrics Collection
```python
from guppie.utils.logging import get_all_metrics

# Get comprehensive metrics from all components
all_metrics = get_all_metrics()
print(f"Total operations: {sum(m['operations'] for m in all_metrics.values())}")
print(f"Total errors: {sum(m['errors'] for m in all_metrics.values())}")
```

### Real-time Monitoring
- **Operation Tracking**: All major operations are logged with timing
- **Error Rate Monitoring**: Error counts and types tracked per component
- **Performance Metrics**: Response times and throughput measurements
- **Circuit Breaker Status**: Real-time failure detection and recovery

## 🔧 Configuration

### Environment Variables
```bash
# Logging configuration
GUPPIE_LOG_LEVEL=INFO
GUPPIE_LOG_FILE=/var/log/guppie/app.log

# Encryption configuration
GUPPIE_KEY_STORAGE_PATH=/secure/keys
GUPPIE_KEY_ROTATION_HOURS=24

# Server configuration
GUPPIE_HOST=0.0.0.0
GUPPIE_PORT=8000
```

### Security Configuration
- **Key Storage**: Secure key storage with 0600 permissions
- **Rotation Schedule**: Configurable automatic key rotation
- **Encryption Standards**: AES-256-GCM with authenticated encryption
- **Error Recovery**: Automatic retry and circuit breaker patterns

## 🏗️ Architecture

### Component Structure
```
guppie/
├── api/                    # FastAPI endpoints and WebSocket handlers
├── consciousness/          # Avatar consciousness and personality systems
├── visual/                # Rendering and expression engines
├── security/              # Encryption and key management
├── utils/                 # Logging and error handling utilities
└── tests/                 # Comprehensive test suite
```

### Security Architecture
- **Defense in Depth**: Multiple layers of security controls
- **Zero Trust**: All data encrypted at rest and in transit
- **Fail-Safe Design**: Graceful degradation on component failure
- **Audit Trail**: Comprehensive logging of all operations

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes with tests**: Ensure 80%+ test coverage
4. **Run the test suite**: `python -m pytest tests/ -v --cov=guppie`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Quality Standards
- **Type Hints**: All functions must include type annotations
- **Error Handling**: Comprehensive error handling required
- **Testing**: Unit and integration tests for all new features
- **Documentation**: Inline comments and docstrings required
- **Security**: Security review for any crypto/auth changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **Deep Seek Models**: Advanced language model integration
- **FastAPI Framework**: High-performance API development
- **Cryptography Library**: Enterprise-grade encryption implementation
- **pytest Framework**: Comprehensive testing capabilities

---

**Note**: This enhanced version includes enterprise-grade security features, comprehensive error handling, structured logging, and extensive test coverage to ensure production readiness and reliability.
