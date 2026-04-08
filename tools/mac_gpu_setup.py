"""
Mac GPU Setup for TensorFlow on Apple Silicon
This module adds configurations to enable TensorFlow to use Metal GPU acceleration on M1/M2/M3 Macs
"""

import os
import tensorflow as tf
import platform

def configure_mac_gpu():
    """
    Configure TensorFlow to use Apple's Metal API for GPU acceleration on Apple Silicon Macs
    """
    # Check if we're running on macOS
    if platform.system() != 'Darwin':
        print("Not running on macOS, skipping Metal GPU setup")
        return False
        
    # Check if we're running on Apple Silicon
    is_apple_silicon = (platform.processor() == 'arm' or 
                       'arm64' in platform.machine().lower() or
                       'M1' in platform.processor() or
                       'M2' in platform.processor() or
                       'M3' in platform.processor())
    
    if not is_apple_silicon:
        print("Not running on Apple Silicon, skipping Metal GPU setup")
        return False
    
    # Try to enable Metal plugin
    try:
        # Configure TensorFlow to use the Metal plugin
        physical_devices = tf.config.list_physical_devices()
        print(f"Available devices: {physical_devices}")
        
        # Check for GPU devices (Metal framework)
        metal_devices = tf.config.list_physical_devices('GPU')
        if not metal_devices:
            print("No Metal-capable GPU found")
            return False
            
        print(f"Found {len(metal_devices)} Metal-capable GPU(s)")
        
        # Enable memory growth to avoid allocating all GPU memory at once
        for device in metal_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Enabled memory growth for {device}")
            except Exception as e:
                print(f"Error enabling memory growth for {device}: {e}")
        
        # Print device details
        for i, device in enumerate(metal_devices):
            device_details = tf.config.experimental.get_device_details(device)
            print(f"GPU {i}: {device_details}")
            
        # Verify TensorFlow can see the Metal device
        with tf.device('/GPU:0'):
            # Simple test operation
            test_tensor = tf.random.normal([1000, 1000])
            result = tf.matmul(test_tensor, tf.transpose(test_tensor))
            # Force execution to verify GPU works
            result_sum = tf.reduce_sum(result).numpy()
            
        print("Metal GPU successfully configured for TensorFlow!")
        return True
        
    except Exception as e:
        print(f"Error configuring Metal GPU: {e}")
        print("Falling back to CPU")
        return False

# Additional environment variables that might help with Metal performance
def set_mac_environment_variables():
    """Set environment variables to optimize Metal performance"""
    os.environ['TF_METAL_DEVICE_MEMORY_FRACTION'] = '0.9'  # Use up to 90% of GPU memory
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allocate only as much GPU memory as needed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow logging (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
    os.environ['MLIR_ENABLE_THREADING'] = '1'  # Enable MLIR threading

if __name__ == "__main__":
    # Set environment variables
    set_mac_environment_variables()
    
    # Configure GPU
    success = configure_mac_gpu()
    
    if success:
        print("TensorFlow is configured to use Metal GPU acceleration")
        
        # Show TensorFlow details
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {tf.keras.__version__}")
        
        # Run a simple benchmark
        import time
        
        # Create large tensors
        start_time = time.time()
        with tf.device('/GPU:0'):
            # Matrix multiplication as a simple benchmark
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            c = tf.matmul(a, b)
            # Force execution
            result = c.numpy()
        gpu_time = time.time() - start_time
        
        start_time = time.time()
        with tf.device('/CPU:0'):
            # Same operation on CPU
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            c = tf.matmul(a, b)
            # Force execution
            result = c.numpy()
        cpu_time = time.time() - start_time
        
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"CPU time: {cpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("TensorFlow will use CPU only")
