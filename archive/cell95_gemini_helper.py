import os

# Define the base directory for the package
base_dir = 'grandnexus'

# Define the subdirectories
subdirs = ['core', 'graph', 'memory', 'llm_interface']

# Create the base directory
os.makedirs(base_dir, exist_ok=True)
print(f"Created directory: {base_dir}")

# Create the subdirectories
for subdir in subdirs:
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")