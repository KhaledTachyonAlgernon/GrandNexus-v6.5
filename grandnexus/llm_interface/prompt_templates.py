# llm_interface/prompt_templates.py

from typing import Dict, Any, List, Optional

def concept_definition_prompt(concept_name: str) -> str:
    """
    Génère un prompt pour obtenir la définition d'un concept.
    """
    return f"""Please provide a detailed definition of the concept: "{concept_name}".
Include the following:
- Primary definition
- Key attributes or properties
- Related concepts or categories
- Examples if applicable
"""

def relation_validation_prompt(source_concept: str, relation_type: str, target_concept: str) -> str:
    """
    Génère un prompt pour valider une relation entre deux concepts.
    """
    return f"""Is the following relationship valid?
Source: {source_concept}
Relation Type: {relation_type}
Target: {target_concept}

Please analyze if this relationship is factually accurate and provide:
1. Whether this relationship is valid (yes/no)
2. Confidence level (0.0 to 1.0)
3. Brief explanation
"""

def chain_of_thought_prompt(question: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Génère un prompt de raisonnement étape par étape (chain of thought).
    Inclut optionnellement du contexte.
    """
    context_str = ""
    if context:
        context_str = "\nContext Information:\n"
        for key, value in context.items():
            context_str += f"- {key}: {value}\n"

    return f"""Question: {question}
{context_str}
Let's think step by step to solve this problem:

1. """

def graph_enrichment_prompt(query: str, graph_data: Dict[str, Any]) -> str:
    """
    Génère un prompt qui incorpore des données du graphe de connaissances.
    """
    graph_context = "Graph Knowledge:\n"
    for entity, data in graph_data.items():
        graph_context += f"- {entity}:\n"
        if "properties" in data:
            graph_context += "  Properties:\n"
            for prop, value in data["properties"].items():
                graph_context += f"    - {prop}: {value}\n"
        if "relations" in data:
            graph_context += "  Relations:\n"
            for rel_type, targets in data["relations"].items():
                for target in targets:
                    graph_context += f"    - {rel_type} -> {target}\n"

    return f"""Query: {query}

{graph_context}

Based on the above graph knowledge, please provide a comprehensive answer:
"""

def learning_feedback_prompt(training_data: Dict[str, Any], model_performance: Dict[str, float]) -> str:
    """
    Génère un prompt pour obtenir du feedback sur les performances d'apprentissage.
    """
    metrics_str = "\n".join([f"- {metric}: {value}" for metric, value in model_performance.items()])

    return f"""Analyze the following learning performance:

Training Data Summary:
- Size: {training_data.get('size', 'Unknown')}
- Distribution: {training_data.get('distribution', 'Unknown')}
- Features: {', '.join(training_data.get('features', []))}

Performance Metrics:
{metrics_str}

Please provide:
1. Analysis of these results
2. Potential improvements
3. Whether this performance is satisfactory for production use
"""

def event_analysis_prompt(event_data: Dict[str, Any]) -> str:
    """
    Génère un prompt pour analyser un événement détecté par la perception.
    """
    event_str = "\n".join([f"- {key}: {value}" for key, value in event_data.items()])

    return f"""Analyze the following event:

Event Data:
{event_str}

Please provide:
1. What is happening in this event?
2. What are the key entities involved?
3. What are potential implications or next steps?
4. Confidence level in this analysis (0.0 to 1.0)
"""