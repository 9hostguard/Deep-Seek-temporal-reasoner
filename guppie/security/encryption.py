"""
GUPPIE Encryption Module - Enterprise-grade Security System
Implements AES-256-GCM encryption with automatic key rotation
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import json
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidTag

logger = logging.getLogger(__name__)


class EncryptionKeyError(Exception):
    """Raised when encryption key operations fail"""
    pass


class EncryptionError(Exception):
    """Raised when encryption/decryption operations fail"""
    pass


class KeyRotationManager:
    """
    Manages encryption key rotation with automatic scheduling
    """
    
    def __init__(self, key_storage_path: str = "/tmp/encryption_keys", rotation_interval_hours: int = 24):
        """
        Initialize key rotation manager
        
        Args:
            key_storage_path: Directory to store encryption keys
            rotation_interval_hours: Hours between automatic key rotations
        """
        self.key_storage_path = Path(key_storage_path)
        self.rotation_interval = timedelta(hours=rotation_interval_hours)
        self.current_key_id: Optional[str] = None
        self.keys: Dict[str, Dict[str, Any]] = {}
        
        # Ensure key storage directory exists
        self.key_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize key rotation scheduler
        self._rotation_task: Optional[asyncio.Task] = None
        
        logger.info(f"KeyRotationManager initialized with storage path: {self.key_storage_path}")
    
    def generate_key(self) -> bytes:
        """Generate a new 256-bit encryption key"""
        return AESGCM.generate_key(bit_length=256)
    
    def create_key_id(self) -> str:
        """Create a unique key identifier"""
        timestamp = int(time.time())
        return f"key_{timestamp}_{os.urandom(4).hex()}"
    
    def store_key(self, key: bytes, key_id: str) -> None:
        """
        Securely store an encryption key
        
        Args:
            key: The encryption key to store
            key_id: Unique identifier for the key
        """
        try:
            key_data = {
                "key_id": key_id,
                "key": base64.b64encode(key).decode('utf-8'),
                "created_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            key_file = self.key_storage_path / f"{key_id}.json"
            with open(key_file, 'w') as f:
                json.dump(key_data, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            self.keys[key_id] = key_data
            logger.info(f"Key {key_id} stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store key {key_id}: {e}")
            raise EncryptionKeyError(f"Key storage failed: {e}")
    
    def load_keys(self) -> None:
        """Load all keys from storage"""
        try:
            for key_file in self.key_storage_path.glob("key_*.json"):
                with open(key_file, 'r') as f:
                    key_data = json.load(f)
                    self.keys[key_data["key_id"]] = key_data
            
            # Find most recent active key
            active_keys = {k: v for k, v in self.keys.items() if v["status"] == "active"}
            if active_keys:
                self.current_key_id = max(active_keys.keys(), 
                                        key=lambda k: active_keys[k]["created_at"])
                logger.info(f"Loaded {len(self.keys)} keys, current: {self.current_key_id}")
            
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            raise EncryptionKeyError(f"Key loading failed: {e}")
    
    def get_current_key(self) -> Tuple[str, bytes]:
        """
        Get the current active encryption key
        
        Returns:
            Tuple of (key_id, key_bytes)
        """
        if not self.current_key_id or self.current_key_id not in self.keys:
            raise EncryptionKeyError("No current encryption key available")
        
        key_data = self.keys[self.current_key_id]
        key_bytes = base64.b64decode(key_data["key"])
        return self.current_key_id, key_bytes
    
    def get_key_by_id(self, key_id: str) -> bytes:
        """
        Get a specific key by ID
        
        Args:
            key_id: The key identifier
            
        Returns:
            The encryption key
        """
        if key_id not in self.keys:
            raise EncryptionKeyError(f"Key {key_id} not found")
        
        key_data = self.keys[key_id]
        return base64.b64decode(key_data["key"])
    
    def rotate_key(self) -> str:
        """
        Perform key rotation - generate new key and mark old as deprecated
        
        Returns:
            New key ID
        """
        try:
            # Generate new key
            new_key = self.generate_key()
            new_key_id = self.create_key_id()
            
            # Store new key
            self.store_key(new_key, new_key_id)
            
            # Deprecate old key
            if self.current_key_id:
                old_key_data = self.keys[self.current_key_id]
                old_key_data["status"] = "deprecated"
                old_key_data["deprecated_at"] = datetime.utcnow().isoformat()
                
                # Update file
                key_file = self.key_storage_path / f"{self.current_key_id}.json"
                with open(key_file, 'w') as f:
                    json.dump(old_key_data, f, indent=2)
            
            # Set new key as current
            self.current_key_id = new_key_id
            
            logger.info(f"Key rotation completed: new key {new_key_id}")
            return new_key_id
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise EncryptionKeyError(f"Key rotation failed: {e}")
    
    async def start_auto_rotation(self) -> None:
        """Start automatic key rotation"""
        if self._rotation_task and not self._rotation_task.done():
            logger.warning("Auto rotation already running")
            return
        
        self._rotation_task = asyncio.create_task(self._rotation_loop())
        logger.info("Automatic key rotation started")
    
    async def stop_auto_rotation(self) -> None:
        """Stop automatic key rotation"""
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass
            logger.info("Automatic key rotation stopped")
    
    async def _rotation_loop(self) -> None:
        """Internal rotation loop"""
        while True:
            try:
                await asyncio.sleep(self.rotation_interval.total_seconds())
                self.rotate_key()
                logger.info("Automatic key rotation performed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto rotation error: {e}")
                # Continue loop even on error


class SecureEncryption:
    """
    Enterprise-grade encryption system using AES-256-GCM
    """
    
    def __init__(self, key_manager: KeyRotationManager):
        """
        Initialize encryption system
        
        Args:
            key_manager: Key rotation manager instance
        """
        self.key_manager = key_manager
        
        # Initialize if no keys exist
        if not self.key_manager.current_key_id:
            self.key_manager.load_keys()
            if not self.key_manager.current_key_id:
                # Create initial key
                initial_key = self.key_manager.generate_key()
                initial_key_id = self.key_manager.create_key_id()
                self.key_manager.store_key(initial_key, initial_key_id)
                self.key_manager.current_key_id = initial_key_id
                logger.info("Initial encryption key created")
    
    def encrypt(self, data: bytes, associated_data: Optional[bytes] = None) -> Dict[str, str]:
        """
        Encrypt data using AES-256-GCM
        
        Args:
            data: Data to encrypt
            associated_data: Optional associated data for authentication
            
        Returns:
            Dictionary containing encrypted data and metadata
        """
        try:
            # Get current key
            key_id, key = self.key_manager.get_current_key()
            
            # Generate nonce
            nonce = os.urandom(12)  # 96-bit nonce for GCM
            
            # Create cipher
            aesgcm = AESGCM(key)
            
            # Encrypt
            ciphertext = aesgcm.encrypt(nonce, data, associated_data)
            
            # Return encrypted package
            encrypted_package = {
                "key_id": key_id,
                "nonce": base64.b64encode(nonce).decode('utf-8'),
                "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
                "associated_data": base64.b64encode(associated_data).decode('utf-8') if associated_data else None,
                "timestamp": datetime.utcnow().isoformat(),
                "algorithm": "AES-256-GCM"
            }
            
            logger.debug(f"Data encrypted with key {key_id}")
            return encrypted_package
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_package: Dict[str, str]) -> bytes:
        """
        Decrypt data from encrypted package
        
        Args:
            encrypted_package: Package returned by encrypt()
            
        Returns:
            Decrypted data
        """
        try:
            # Extract components
            key_id = encrypted_package["key_id"]
            nonce = base64.b64decode(encrypted_package["nonce"])
            ciphertext = base64.b64decode(encrypted_package["ciphertext"])
            associated_data = None
            if encrypted_package.get("associated_data"):
                associated_data = base64.b64decode(encrypted_package["associated_data"])
            
            # Get decryption key
            key = self.key_manager.get_key_by_id(key_id)
            
            # Create cipher
            aesgcm = AESGCM(key)
            
            # Decrypt
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data)
            
            logger.debug(f"Data decrypted with key {key_id}")
            return plaintext
            
        except InvalidTag:
            logger.error("Decryption failed: Invalid authentication tag")
            raise EncryptionError("Decryption failed: Authentication verification failed")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Decryption failed: {e}")
    
    def encrypt_string(self, text: str, associated_data: Optional[str] = None) -> Dict[str, str]:
        """
        Encrypt a string
        
        Args:
            text: String to encrypt
            associated_data: Optional associated string data
            
        Returns:
            Encrypted package
        """
        data_bytes = text.encode('utf-8')
        assoc_bytes = associated_data.encode('utf-8') if associated_data else None
        return self.encrypt(data_bytes, assoc_bytes)
    
    def decrypt_string(self, encrypted_package: Dict[str, str]) -> str:
        """
        Decrypt to string
        
        Args:
            encrypted_package: Package from encrypt_string()
            
        Returns:
            Decrypted string
        """
        plaintext = self.decrypt(encrypted_package)
        return plaintext.decode('utf-8')
    
    def encrypt_json(self, obj: Any, associated_data: Optional[str] = None) -> Dict[str, str]:
        """
        Encrypt a JSON-serializable object
        
        Args:
            obj: Object to encrypt
            associated_data: Optional associated data
            
        Returns:
            Encrypted package
        """
        json_str = json.dumps(obj, ensure_ascii=False)
        return self.encrypt_string(json_str, associated_data)
    
    def decrypt_json(self, encrypted_package: Dict[str, str]) -> Any:
        """
        Decrypt to JSON object
        
        Args:
            encrypted_package: Package from encrypt_json()
            
        Returns:
            Decrypted object
        """
        json_str = self.decrypt_string(encrypted_package)
        return json.loads(json_str)


class SecureStorage:
    """
    Secure file storage with encryption
    """
    
    def __init__(self, encryption: SecureEncryption, storage_path: str = "/tmp/secure_storage"):
        """
        Initialize secure storage
        
        Args:
            encryption: SecureEncryption instance
            storage_path: Directory for encrypted files
        """
        self.encryption = encryption
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"SecureStorage initialized at {self.storage_path}")
    
    async def store_file(self, filename: str, data: bytes, metadata: Optional[Dict] = None) -> str:
        """
        Store file with encryption
        
        Args:
            filename: Name of file
            data: File data
            metadata: Optional metadata
            
        Returns:
            Storage ID
        """
        try:
            # Create storage ID
            storage_id = f"file_{int(time.time())}_{os.urandom(4).hex()}"
            
            # Prepare metadata
            file_metadata = {
                "original_filename": filename,
                "size": len(data),
                "stored_at": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Encrypt data
            encrypted_data = self.encryption.encrypt(data, json.dumps(file_metadata).encode())
            
            # Store encrypted file
            storage_file = self.storage_path / f"{storage_id}.enc"
            with open(storage_file, 'w') as f:
                json.dump(encrypted_data, f, indent=2)
            
            # Set permissions
            os.chmod(storage_file, 0o600)
            
            logger.info(f"File {filename} stored as {storage_id}")
            return storage_id
            
        except Exception as e:
            logger.error(f"File storage failed: {e}")
            raise EncryptionError(f"File storage failed: {e}")
    
    async def retrieve_file(self, storage_id: str) -> Tuple[str, bytes, Dict]:
        """
        Retrieve and decrypt file
        
        Args:
            storage_id: Storage identifier
            
        Returns:
            Tuple of (filename, data, metadata)
        """
        try:
            storage_file = self.storage_path / f"{storage_id}.enc"
            if not storage_file.exists():
                raise EncryptionError(f"File {storage_id} not found")
            
            # Load encrypted data
            with open(storage_file, 'r') as f:
                encrypted_data = json.load(f)
            
            # Decrypt
            decrypted_data = self.encryption.decrypt(encrypted_data)
            
            # Extract metadata from associated data
            metadata_str = encrypted_data.get("associated_data")
            if metadata_str:
                metadata = json.loads(base64.b64decode(metadata_str).decode())
                filename = metadata["original_filename"]
                file_metadata = metadata.get("metadata", {})
            else:
                filename = "unknown"
                file_metadata = {}
            
            logger.info(f"File {storage_id} retrieved as {filename}")
            return filename, decrypted_data, file_metadata
            
        except Exception as e:
            logger.error(f"File retrieval failed: {e}")
            raise EncryptionError(f"File retrieval failed: {e}")


# Global encryption instance
_encryption_system: Optional[SecureEncryption] = None
_key_manager: Optional[KeyRotationManager] = None


def get_encryption_system() -> SecureEncryption:
    """Get global encryption system instance"""
    global _encryption_system, _key_manager
    
    if _encryption_system is None:
        _key_manager = KeyRotationManager()
        _encryption_system = SecureEncryption(_key_manager)
        logger.info("Global encryption system initialized")
    
    return _encryption_system


def get_key_manager() -> KeyRotationManager:
    """Get global key manager instance"""
    global _key_manager
    
    if _key_manager is None:
        get_encryption_system()  # This will initialize both
    
    return _key_manager