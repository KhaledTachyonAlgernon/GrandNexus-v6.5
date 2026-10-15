import re

code_cells = [
    """
# graph_engine.py
import logging
import time
import uuid
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
from collections import defaultdict, deque
import math
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from neo4j import GraphDatabase, Driver
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
""",
    """
# graph_engine.py (Second version)
import logging
import time
import uuid
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, Set, Callable
from collections import defaultdict, deque
import math
import json
import os
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from neo4j import GraphDatabase, Driver
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
""",
    """
# pattern_mining.py
import logging
import time
import threading
import math
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import uuid

from grandnexus.graph.graph_engine import ( create_graph_engine,
    AbstractGraphEngine, GraphNode, GraphEdge, GraphPattern,
    GraphQuery, NodeNotFoundError, EdgeNotFoundError
)
from grandnexus.graph.graph_pattern_interface import (
    PatternMiningInterface, PatternMatchResult, FrequentPattern
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
""",
    """
# pdvSygkY4JRG
# %load_ext cudf.pandas
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
""",
    """
# graph_pattern_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field

from grandnexus.graph.graph_engine import (
    create_graph_engine, AbstractGraphEngine, GraphNode, GraphEdge, GraphPattern,
    GraphQuery, NodeNotFoundError, EdgeNotFoundError
)

import logging
""",
    """
# graph_module_manager.py
import logging
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple, Set, Type
import time

from grandnexus.core.nexus_core import NexusCore
from grandnexus.core.registry import ModuleRegistry

from grandnexus.graph.graph_engine import (
    create_graph_engine, AbstractGraphEngine, GraphNode, GraphEdge, GraphPattern,
    GraphQuery, NodeNotFoundError, EdgeNotFoundError,
    create_graph_engine
)
from grandnexus.graph.pattern_mining import (
    PatternMiningModule, PatternMatchResult, FrequentPattern,
    create_pattern_mining
)
from grandnexus.graph.graph_pattern_interface import GraphPatternBridge
""",
    """
# resource_manager.py (First version)
import logging
import psutil
import time
import queue
import threading
import heapq
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Callable, Union, Deque, Tuple, Set
)
""",
    """
# resource_manager.py (Second version)
import logging
import threading
import time
import uuid
import queue
import heapq
import psutil
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from collections import defaultdict, deque
import numpy as np

from grandnexus.core.nexus_core import NexusCore
""",
    """
# llm_engine.py
import logging
import threading
import time
import uuid
import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
from collections import defaultdict, deque
import numpy as np

from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.graph.graph_engine import GraphEngine
""",
    """
# llm_interface/prompt_templates.py
from typing import Dict, Any, List, Optional
""",
    """
# llm_interface/response_parser.py
import json
import re
from typing import Dict, Any, List, Optional, Union
""",
    """
# llm_interface/__init__.py
from .llm_engine import LLMEngine, LLMConfig, LLMRequest, LLMResponse, LLMProvider
from . import prompt_templates
from . import response_parser
"""
]

external_dependencies = set()
internal_dependencies = defaultdict(set)
unprovided_modules = set()

module_names = {
    'graph_engine.py': 'graph.graph_engine',
    'pattern_mining.py': 'graph.pattern_mining',
    'graph_pattern_interface.py': 'graph.graph_pattern_interface',
    'graph_module_manager.py': 'graph.graph_module_manager',
    'resource_manager.py': 'resource_manager',
    'llm_engine.py': 'llm_engine',
    'llm_interface/prompt_templates.py': 'llm_interface.prompt_templates',
    'llm_interface/response_parser.py': 'llm_interface.response_parser',
    'llm_interface/__init__.py': 'llm_interface'
}

provided_grandnexus_modules = {
    'grandnexus.graph.graph_engine',
    'grandnexus.graph.pattern_mining',
    'grandnexus.graph.graph_pattern_interface',
    'grandnexus.graph.graph_module_manager',
    'grandnexus.resource_manager',
    'grandnexus.llm_engine', # Note: This is referenced, but the file is llm_interface/llm_engine.py
    'grandnexus.llm_interface.prompt_templates',
    'grandnexus.llm_interface.response_parser',
    'grandnexus.llm_interface'
}


for cell_content in code_cells:
    # Extract potential module name from comment or assume based on content
    cell_name_match = re.search(r'^#\s*(.*)\.py', cell_content, re.MULTILINE)
    if cell_name_match:
        cell_file_name = cell_name_match.group(1).strip() + '.py'
    elif 'graph_pattern_interface.py' in cell_content:
         cell_file_name = 'graph_pattern_interface.py'
    elif 'graph_module_manager.py' in cell_content:
         cell_file_name = 'graph_module_manager.py'
    elif 'llm_interface/prompt_templates.py' in cell_content:
         cell_file_name = 'llm_interface/prompt_templates.py'
    elif 'llm_interface/response_parser.py' in cell_content:
         cell_file_name = 'llm_interface/response_parser.py'
    elif 'llm_interface/__init__.py' in cell_content:
         cell_file_name = 'llm_interface/__init__.py'
    elif 'psutil' in cell_content and 'threading' in cell_content and 'heapq' in cell_content:
        # Heuristic to differentiate resource_manager versions
        if 'NexusCore' in cell_content:
             cell_file_name = 'resource_manager_v2.py'
        else:
             cell_file_name = 'resource_manager_v1.py'
    elif 'graph_engine' in cell_content and 'Neo4jGraphEngine' in cell_content:
        # Heuristic to differentiate graph_engine versions
        if 'os' in cell_content and 'random' in cell_content:
             cell_file_name = 'graph_engine_v2.py'
        else:
             cell_file_name = 'graph_engine_v1.py'
    elif 'pandas' in cell_content and 'numpy' in cell_content and 'Ticket Number' in cell_content:
         cell_file_name = 'ticket_analysis.py' # Give it a descriptive name
    else:
         cell_file_name = 'unknown_cell.py'

    current_module_base = module_names.get(cell_file_name)
    if current_module_base is None:
        # Try to map based on heuristics if not in the direct map
        if 'graph_engine_v' in cell_file_name:
            current_module_base = 'graph.graph_engine'
        elif 'resource_manager_v' in cell_file_name:
             current_module_base = 'resource_manager'
        elif 'llm_engine.py' in cell_file_name:
             current_module_base = 'llm_engine' # This naming is inconsistent with file path
        elif 'ticket_analysis.py' in cell_file_name:
             current_module_base = 'analysis.ticket_analysis' # Placeholder name
        else:
             current_module_base = cell_file_name.replace('/', '.').replace('.py', '')


    # Find all import statements
    imports = re.findall(r'^(?:import\s+([\w.]+))|(?:from\s+([\w.]+)\s+import)', cell_content, re.MULTILINE)

    for imp in imports:
        external_lib = imp[0] if imp[0] else imp[1] # Get the imported name

        if external_lib.startswith('grandnexus.'):
            # Internal GrandNexus dependency
            dep_module = external_lib

            if current_module_base:
                 internal_dependencies[f'grandnexus.{current_module_base}'].add(dep_module)

            # Check if the dependency is provided
            # Normalize dependency name (remove specific imported objects)
            base_dep_module = dep_module.split('.')[0:3] # grandnexus.module.submodule
            base_dep_module_name = '.'.join(base_dep_module)

            is_provided = False
            for provided_module_base in provided_grandnexus_modules:
                if dep_module.startswith(provided_module_base):
                    is_provided = True
                    break

            if not is_provided:
                unprovided_modules.add(dep_module)

        else:
            # External library
            # Clean up potential 'as' aliases and imported submodules for external libs
            base_lib = external_lib.split('.')[0]
            if base_lib not in ['logging', 'time', 'uuid', 'threading', 'abc', 'dataclasses', 'typing', 'collections', 'math', 'json', 'os', 'random', 're', 'hashlib', 'enum', 'auto', 'queue', 'heapq', 'psutil']: # Standard libraries
                external_dependencies.add(base_lib)


print("External Dependencies:")
for dep in sorted(list(external_dependencies)):
    print(f"- {dep}")

print("\nInternal Dependencies:")
for module, deps in internal_dependencies.items():
    print(f"- {module} depends on:")
    for dep in sorted(list(deps)):
        print(f"  - {dep}")

print("\nUnprovided GrandNexus Modules/Paths (based on imports):")
for mod in sorted(list(unprovided_modules)):
    print(f"- {mod}")
