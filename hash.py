import uuid
import hashlib
import time
import os

def generate_unique_hash(id_value):
    """
    Generates a unique hash for the given ID each time it's called,
    without requiring persistent storage.
    
    Args:
        id_value: The base ID to hash (can be any type that can be converted to string)
        
    Returns:
        A unique hexadecimal hash string
    """
    # Convert the ID to a string if it isn't already
    id_str = str(id_value)
    
    # Combine these sources of entropy:
    # 1. The original ID
    # 2. Current timestamp with microsecond precision
    # 3. A random UUID (version 4)
    # 4. Process ID
    # 5. Random OS-supplied bytes
    
    unique_components = [
        id_str,
        str(time.time_ns()),  # Nanosecond timestamp for extra precision
        str(uuid.uuid4()),    # Random UUID
        str(os.getpid()),     # Process ID
        os.urandom(8).hex()   # 8 random bytes from OS
    ]
    
    # Join all components and hash them
    unique_string = ":".join(unique_components)
    
    # Create SHA-256 hash
    hash_object = hashlib.sha256(unique_string.encode())
    unique_hash = hash_object.hexdigest()
    
    return unique_hash