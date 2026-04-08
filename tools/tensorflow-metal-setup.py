"""
M1/M2/M3 Mac TensorFlow Metal GPU Setup Script

This script helps set up TensorFlow with Metal GPU support on Apple Silicon Macs.
It creates a conda environment with the necessary dependencies and verifies that
the GPU is properly detected and configured.

Usage:
1. Install Miniconda for Apple Silicon if you haven't already
2. Run this script to create a new environment with TensorFlow-Metal
"""

import os
import subprocess
import sys
import platform

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def print_step(step, msg):
    """Print a step message with formatting"""
    print(f"{BLUE}[Step {step}]{ENDC} {msg}")

def print_success(msg):
    """Print a success message with formatting"""
    print(f"{GREEN}[SUCCESS]{ENDC} {msg}")

def print_warning(msg):
    """Print a warning message with formatting"""
    print(f"{YELLOW}[WARNING]{ENDC} {msg}")

def print_error(msg):
    """Print an error message with formatting"""
    print(f"{RED}[ERROR]{ENDC} {msg}")

def run_command(cmd, shell=False):
    """Run a command and return the result"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        else:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with error: {e}")
        print_error(f"Output: {e.stderr}")
        return None

def check_system():
    """Check if running on Apple Silicon Mac"""
    print_step(1, "Checking system compatibility...")
    
    if platform.system() != 'Darwin':
        print_error("This script is only for macOS systems.")
        return False
        
    # Check if running on Apple Silicon
    is_apple_silicon = (platform.processor() == 'arm' or 
                       'arm64' in platform.machine().lower())
    
    if not is_apple_silicon:
        print_error("This script is specifically for Apple Silicon Macs (M1/M2/M3).")
        return False
    
    # Get Mac model
    try:
        mac_model = run_command(['sysctl', '-n', 'hw.model'])
        print_success(f"Detected Mac model: {mac_model}")
    except (OSError, FileNotFoundError):
        print_warning("Could not determine Mac model.")
        
    return True

def check_conda():
    """Check if conda is installed and accessible"""
    print_step(2, "Checking conda installation...")
    
    try:
        conda_info = run_command(['conda', 'info'])
        if 'conda version' in conda_info:
            print_success("Conda is properly installed.")
            
            # Check if conda is configured for Apple Silicon
            if 'arm64' in conda_info or 'aarch64' in conda_info:
                print_success("Conda is configured for Apple Silicon.")
            else:
                print_warning("Your conda installation may not be optimized for Apple Silicon.")
                
            return True
    except (OSError, FileNotFoundError):
        print_error("Conda is not installed or not in PATH.")
        print("Please install Miniconda for Apple Silicon from: https://github.com/conda-forge/miniforge")
        return False
        
def create_tensorflow_environment():
    """Create a conda environment with TensorFlow and Metal support"""
    print_step(3, "Creating TensorFlow environment with Metal support...")
    
    env_name = "tensorflow-metal"
    
    # Check if environment already exists
    envs = run_command(['conda', 'env', 'list'])
    if env_name in envs:
        print_warning(f"Environment '{env_name}' already exists.")
        choice = input("Do you want to remove and recreate it? (y/n): ")
        if choice.lower() == 'y':
            run_command(['conda', 'env', 'remove', '-n', env_name])
        else:
            print_warning("Skipping environment creation.")
            return env_name
    
    # Create environment with Python 3.9 (most compatible with current TensorFlow)
    print("Creating conda environment... (this may take a few minutes)")
    result = run_command(['conda', 'create', '-n', env_name, 'python=3.9', '-y'])
    if result is None:
        print_error("Failed to create conda environment.")
        return None
        
    print_success(f"Created conda environment: {env_name}")
    
    # Install TensorFlow dependencies
    print("Installing TensorFlow dependencies... (this may take a few minutes)")
    cmd = f"conda install -n {env_name} -c apple tensorflow-deps -y"
    result = run_command(cmd, shell=True)
    if result is None:
        print_error("Failed to install TensorFlow dependencies.")
        return None
        
    # Install TensorFlow for macOS
    print("Installing TensorFlow for macOS...")
    cmd = f"conda run -n {env_name} pip install tensorflow-macos"
    result = run_command(cmd, shell=True)
    if result is None:
        print_error("Failed to install TensorFlow for macOS.")
        return None
        
    # Install Metal plugin
    print("Installing TensorFlow Metal plugin...")
    cmd = f"conda run -n {env_name} pip install tensorflow-metal"
    result = run_command(cmd, shell=True)
    if result is None:
        print_error("Failed to install TensorFlow Metal plugin.")
        return None
    
    # Install additional useful packages
    print("Installing additional packages...")
    packages = ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy", "jupyterlab", "h5py", "pillow"]
    for package in packages:
        cmd = f"conda run -n {env_name} pip install {package} --upgrade"
        run_command(cmd, shell=True)
    
    print_success(f"TensorFlow with Metal support has been installed in the '{env_name}' environment.")
    return env_name

def verify_gpu_support(env_name):
    """Verify that TensorFlow can detect and use the GPU"""
    print_step(4, "Verifying GPU support...")
    
    # Create verification script
    script = """
import tensorflow as tf
import time

# Check GPU availability
print("TensorFlow version:", tf.__version__)
print("Checking for GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\\nGPU is available:")
    for gpu in gpus:
        print("  -", gpu)
    
    # Get device details
    for i, gpu in enumerate(gpus):
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"\\nGPU {i} details:", details)
        except (RuntimeError, AttributeError):
            print(f"\\nCould not get details for GPU {i}")
    
    # Test GPU with a simple operation
    print("\\nRunning simple benchmark...")
    
    # Run on CPU
    with tf.device('/CPU:0'):
        start_time = time.time()
        # Create large tensors
        a = tf.random.normal([5000, 5000])
        b = tf.random.normal([5000, 5000])
        # Matrix multiplication
        c = tf.matmul(a, b)
        # Force execution
        result = c.numpy()
        cpu_time = time.time() - start_time
    
    # Run on GPU
    with tf.device('/GPU:0'):
        start_time = time.time()
        # Create large tensors
        a = tf.random.normal([5000, 5000])
        b = tf.random.normal([5000, 5000])
        # Matrix multiplication
        c = tf.matmul(a, b)
        # Force execution
        result = c.numpy()
        gpu_time = time.time() - start_time
    
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"\\nGPU is {speedup:.2f}x faster than CPU!")
    else:
        print(f"\\nWarning: GPU is slower than CPU. This may happen with small operations due to transfer overhead.")
else:
    print("\\nNo GPU was detected by TensorFlow.")
    print("Make sure the tensorflow-metal plugin is installed correctly.")
"""
    
    # Write script to temporary file
    script_path = "verify_gpu.py"
    with open(script_path, "w") as f:
        f.write(script)
    
    # Run verification script
    print("Running verification script...")
    cmd = f"conda run -n {env_name} python {script_path}"
    output = run_command(cmd, shell=True)
    
    # Clean up
    os.remove(script_path)
    
    # Check output
    if output is not None:
        print("\nVerification output:")
        print(output)
        
        if "GPU is available" in output and "No GPU was detected" not in output:
            print_success("GPU was successfully detected and configured!")
            return True
        else:
            print_error("GPU was not properly detected.")
            return False
    else:
        print_error("Verification script failed to run.")
        return False

def print_next_steps(env_name):
    """Print instructions for next steps"""
    print_step(5, "Next steps...")
    
    print(f"""
To use TensorFlow with GPU support:

1. Activate the environment:
   conda activate {env_name}

2. Start JupyterLab or Jupyter Notebook:
   jupyter lab
   # or
   jupyter notebook

3. In your TensorFlow code, you can verify GPU availability with:
   import tensorflow as tf
   print("GPUs Available:", tf.config.list_physical_devices('GPU'))

4. Use the GPU explicitly in your code (if needed):
   with tf.device('/GPU:0'):
       # Your TensorFlow operations here

5. For your cell edge detection project, make sure to activate the environment
   before running your code:
   conda activate {env_name}
   python cell_edge_detection.py
""")

def main():
    """Main function"""
    print(f"{BLUE}{'='*80}{ENDC}")
    print(f"{BLUE}TensorFlow with Metal GPU Setup for Apple Silicon Macs{ENDC}")
    print(f"{BLUE}{'='*80}{ENDC}")
    
    # Check system
    if not check_system():
        return
    
    # Check conda
    if not check_conda():
        return
    
    # Create environment
    env_name = create_tensorflow_environment()
    if env_name is None:
        return
    
    # Verify GPU support
    verify_gpu_support(env_name)
    
    # Print next steps
    print_next_steps(env_name)
    
    print(f"{BLUE}{'='*80}{ENDC}")
    print(f"{GREEN}Setup complete!{ENDC}")
    print(f"{BLUE}{'='*80}{ENDC}")

if __name__ == "__main__":
    main()
