"""
Test cases for GUPPIE Security and Error Handling Systems
Tests encryption, key rotation, logging, and error handling
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import json
from unittest.mock import patch, MagicMock

from guppie.security.encryption import (
    KeyRotationManager, SecureEncryption, SecureStorage,
    EncryptionKeyError, EncryptionError,
    get_encryption_system, get_key_manager
)
from guppie.utils.logging import (
    StructuredLogger, ErrorHandlingMixin, error_handler,
    CircuitBreaker, RetryHandler, get_logger
)


class TestEncryptionSystem:
    """Test encryption functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.key_manager = KeyRotationManager(self.temp_dir + "/keys")
        self.encryption = SecureEncryption(self.key_manager)
    
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_key_generation(self):
        """Test encryption key generation"""
        key = self.key_manager.generate_key()
        assert len(key) == 32  # 256 bits
        assert isinstance(key, bytes)
    
    def test_key_storage_and_retrieval(self):
        """Test key storage and retrieval"""
        key = self.key_manager.generate_key()
        key_id = self.key_manager.create_key_id()
        
        # Store key
        self.key_manager.store_key(key, key_id)
        
        # Retrieve key
        retrieved_key = self.key_manager.get_key_by_id(key_id)
        assert retrieved_key == key
    
    def test_key_rotation(self):
        """Test key rotation functionality"""
        # Create initial key
        initial_key = self.key_manager.generate_key()
        initial_key_id = self.key_manager.create_key_id()
        self.key_manager.store_key(initial_key, initial_key_id)
        self.key_manager.current_key_id = initial_key_id
        
        # Rotate key
        new_key_id = self.key_manager.rotate_key()
        
        assert new_key_id != initial_key_id
        assert self.key_manager.current_key_id == new_key_id
        
        # Check old key is deprecated
        assert self.key_manager.keys[initial_key_id]["status"] == "deprecated"
        assert self.key_manager.keys[new_key_id]["status"] == "active"
    
    def test_string_encryption_decryption(self):
        """Test string encryption and decryption"""
        test_string = "This is a test message with special characters: 🌟🧠💾"
        
        # Encrypt
        encrypted_package = self.encryption.encrypt_string(test_string)
        
        # Verify package structure
        assert "key_id" in encrypted_package
        assert "nonce" in encrypted_package
        assert "ciphertext" in encrypted_package
        assert "algorithm" in encrypted_package
        assert encrypted_package["algorithm"] == "AES-256-GCM"
        
        # Decrypt
        decrypted_string = self.encryption.decrypt_string(encrypted_package)
        assert decrypted_string == test_string
    
    def test_json_encryption_decryption(self):
        """Test JSON object encryption and decryption"""
        test_object = {
            "avatar_id": "test_avatar",
            "consciousness_level": 0.85,
            "memories": ["memory1", "memory2"],
            "metadata": {
                "created_at": "2023-01-01T00:00:00Z",
                "version": 1.0
            }
        }
        
        # Encrypt
        encrypted_package = self.encryption.encrypt_json(test_object)
        
        # Decrypt
        decrypted_object = self.encryption.decrypt_json(encrypted_package)
        assert decrypted_object == test_object
    
    def test_encryption_with_associated_data(self):
        """Test encryption with associated data for authentication"""
        test_data = "Sensitive avatar data"
        associated_data = "avatar_id:test123"
        
        # Encrypt with associated data
        encrypted_package = self.encryption.encrypt_string(test_data, associated_data)
        
        # Decrypt should work with same associated data
        decrypted_data = self.encryption.decrypt_string(encrypted_package)
        assert decrypted_data == test_data
        
        # Tampering with associated data should fail
        tampered_package = encrypted_package.copy()
        tampered_package["associated_data"] = "tampered_data"
        
        with pytest.raises(EncryptionError):
            self.encryption.decrypt_string(tampered_package)
    
    def test_encryption_error_handling(self):
        """Test encryption error scenarios"""
        # Test decryption with invalid key ID
        fake_package = {
            "key_id": "nonexistent_key",
            "nonce": "fake_nonce",
            "ciphertext": "fake_ciphertext",
            "algorithm": "AES-256-GCM"
        }
        
        with pytest.raises(EncryptionError):
            self.encryption.decrypt(fake_package)
    
    @pytest.mark.asyncio
    async def test_secure_storage(self):
        """Test secure file storage"""
        storage = SecureStorage(self.encryption, self.temp_dir + "/storage")
        
        # Test file storage
        test_filename = "test_file.txt"
        test_data = b"This is test file content with binary data: \x00\x01\x02"
        test_metadata = {"file_type": "test", "version": 1}
        
        # Store file
        storage_id = await storage.store_file(test_filename, test_data, test_metadata)
        assert storage_id.startswith("file_")
        
        # Retrieve file
        filename, data, metadata = await storage.retrieve_file(storage_id)
        assert filename == test_filename
        assert data == test_data
        assert metadata == test_metadata
    
    @pytest.mark.asyncio
    async def test_auto_key_rotation(self):
        """Test automatic key rotation"""
        # Use short rotation interval for testing
        key_manager = KeyRotationManager(self.temp_dir + "/auto_keys", rotation_interval_hours=0.001)  # ~3.6 seconds
        
        # Create initial key
        initial_key = key_manager.generate_key()
        initial_key_id = key_manager.create_key_id()
        key_manager.store_key(initial_key, initial_key_id)
        key_manager.current_key_id = initial_key_id
        
        # Start auto rotation
        await key_manager.start_auto_rotation()
        
        # Wait for rotation to occur
        await asyncio.sleep(4)
        
        # Stop auto rotation
        await key_manager.stop_auto_rotation()
        
        # Check that key was rotated
        assert key_manager.current_key_id != initial_key_id


class TestLoggingSystem:
    """Test logging and error handling functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")
        self.logger = StructuredLogger("test_logger", "INFO", self.log_file)
    
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_structured_logging(self):
        """Test structured logging functionality"""
        # Test operation logging
        self.logger.log_operation("test_operation", test_param="value")
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.logger.log_error(e, "test_context")
        
        # Test warning logging
        self.logger.log_warning("Test warning", extra_info="warning_data")
        
        # Test performance logging
        self.logger.log_performance("test_performance", 1.5, operation_type="test")
        
        # Check metrics
        metrics = self.logger.get_metrics()
        assert metrics["operations"] >= 1  # operation logs
        assert metrics["errors"] == 1
        assert metrics["warnings"] == 1
        assert "uptime_seconds" in metrics
        
        # Check log file was created
        assert os.path.exists(self.log_file)
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            assert "test_operation" in log_content
            assert "Test error" in log_content
            assert "Test warning" in log_content
    
    def test_error_handling_mixin(self):
        """Test error handling mixin functionality"""
        
        class TestClass(ErrorHandlingMixin):
            def __init__(self):
                super().__init__()
            
            def test_method(self):
                try:
                    raise ValueError("Test error")
                except Exception as e:
                    return self.handle_error(e, "test_method", reraise=False)
        
        test_instance = TestClass()
        error = test_instance.test_method()
        
        assert isinstance(error, ValueError)
        summary = test_instance.get_error_summary()
        assert summary["total_errors"] == 1
        assert summary["error_counts"]["ValueError"] == 1
        assert len(summary["recent_errors"]) == 1
    
    def test_error_handler_decorator(self):
        """Test error handler decorator"""
        
        @error_handler("test_context", reraise=False, log_performance=True)
        def test_function(should_fail=False):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Test successful execution
        result = test_function(should_fail=False)
        assert result == "success"
        
        # Test error handling
        result = test_function(should_fail=True)
        assert result is None  # Returns None when error handled
    
    @pytest.mark.asyncio
    async def test_async_error_handler(self):
        """Test async error handler decorator"""
        
        @error_handler("async_test_context", reraise=False)
        async def async_test_function(should_fail=False):
            if should_fail:
                raise ValueError("Async test error")
            return "async_success"
        
        # Test successful execution
        result = await async_test_function(should_fail=False)
        assert result == "async_success"
        
        # Test error handling
        result = await async_test_function(should_fail=True)
        assert result is None
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern"""
        
        @CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        def unreliable_function(should_fail=True):
            if should_fail:
                raise ValueError("Simulated failure")
            return "success"
        
        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                unreliable_function(should_fail=True)
        
        # Circuit should now be open - next call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            unreliable_function(should_fail=False)
    
    def test_retry_handler(self):
        """Test retry handler with exponential backoff"""
        
        call_count = 0
        
        @RetryHandler(max_retries=3, base_delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 times
                raise ValueError(f"Failure {call_count}")
            return "success"
        
        # Should succeed on third attempt
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_handler(self):
        """Test async retry handler"""
        
        call_count = 0
        
        @RetryHandler(max_retries=2, base_delay=0.1)
        async def async_flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:  # Fail first time
                raise ValueError(f"Async failure {call_count}")
            return "async_success"
        
        result = await async_flaky_function()
        assert result == "async_success"
        assert call_count == 2


class TestIntegration:
    """Integration tests for security and logging systems"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after integration tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_global_encryption_system(self):
        """Test global encryption system initialization"""
        # Test global instance creation
        encryption = get_encryption_system()
        assert encryption is not None
        
        key_manager = get_key_manager()
        assert key_manager is not None
        
        # Test encryption/decryption works
        test_data = "Global encryption test"
        encrypted = encryption.encrypt_string(test_data)
        decrypted = encryption.decrypt_string(encrypted)
        assert decrypted == test_data
    
    def test_error_handling_with_encryption(self):
        """Test error handling integration with encryption"""
        
        class SecureAvatarService(ErrorHandlingMixin):
            def __init__(self):
                super().__init__()
                self.encryption = get_encryption_system()
            
            @error_handler("secure_store", log_performance=True)
            def store_avatar_data(self, avatar_id: str, data: dict):
                try:
                    # Simulate storing encrypted avatar data
                    encrypted_data = self.encryption.encrypt_json(data, avatar_id)
                    return {"success": True, "encrypted_package": encrypted_data}
                except Exception as e:
                    self.handle_error(e, f"Failed to store data for {avatar_id}")
                    raise
        
        service = SecureAvatarService()
        
        # Test successful operation
        test_data = {"consciousness_level": 0.8, "memories": ["test"]}
        result = service.store_avatar_data("test_avatar", test_data)
        
        assert result["success"] is True
        assert "encrypted_package" in result
        
        # Verify we can decrypt the data
        decrypted = service.encryption.decrypt_json(result["encrypted_package"])
        assert decrypted == test_data
    
    @pytest.mark.asyncio
    async def test_comprehensive_error_scenarios(self):
        """Test comprehensive error scenarios"""
        logger = get_logger("integration_test")
        
        # Test various error scenarios
        scenarios = [
            ("encryption_error", lambda: get_encryption_system().decrypt({"invalid": "package"})),
            ("value_error", lambda: int("not_a_number")),
            ("file_error", lambda: open("/nonexistent/file.txt")),
        ]
        
        for scenario_name, scenario_func in scenarios:
            try:
                scenario_func()
            except Exception as e:
                logger.log_error(e, scenario_name)
        
        # Check that all errors were logged
        metrics = logger.get_metrics()
        assert metrics["errors"] == len(scenarios)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])