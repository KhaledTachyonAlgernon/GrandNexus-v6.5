import os

# Create __init__.py in the base directory
base_init_path = os.path.join('grandnexus', '__init__.py')
with open(base_init_path, 'w') as f:
    f.write("# This file makes the 'grandnexus' directory a Python package\n")
print(f"Created: {base_init_path}")

# Create __init__.py in subdirectories
subdirs = ['core', 'graph', 'memory', 'llm_interface']
for subdir in subdirs:
    init_path = os.path.join('grandnexus', subdir, '__init__.py')
    with open(init_path, 'w') as f:
        f.write(f"# This file makes the '{subdir}' directory a Python package\n")
    print(f"Created: {init_path}")

# Create __init__.py in nested llm_interface directory
llm_interface_init_path = os.path.join('grandnexus', 'llm_interface', '__init__.py')
with open(llm_interface_init_path, 'w') as f:
     f.write("# This file makes the 'llm_interface' directory a Python package\n")
print(f"Created: {llm_interface_init_path}")