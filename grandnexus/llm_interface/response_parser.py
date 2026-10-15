# llm_interface/response_parser.py

import json
import re
from typing import Dict, Any, List, Optional, Union

def parse_concept_definition(response: str) -> Dict[str, Any]:
   """
   Parse la réponse d'une définition de concept.
   Extrait la définition, les attributs, et les concepts liés.
   """
   result = {
       "definition": "",
       "attributes": [],
       "related_concepts": [],
       "examples": [],
       "confidence": 0.8  # Valeur par défaut
   }

   # Recherche de la définition principale
   definition_match = re.search(r"(definition|meaning):\s*([^\n]+)", response, re.IGNORECASE)
   if definition_match:
       result["definition"] = definition_match.group(2).strip()
   else:
       # Essayer de prendre la première phrase significative
       sentences = re.split(r'[.!?]', response)
       if sentences:
           result["definition"] = sentences[0].strip()

   # Recherche d'attributs/propriétés
   attr_match = re.search(r"(attributes|properties|characteristics):(.*?)(?:\n\n|\n[A-Z]|$)",
                          response, re.IGNORECASE | re.DOTALL)
   if attr_match:
       attr_text = attr_match.group(2).strip()
       # Extraire les éléments de liste
       attributes = re.findall(r"[-•*]\s*([^\n]+)", attr_text)
       if attributes:
           result["attributes"] = [attr.strip() for attr in attributes]
       else:
           # Séparer par points-virgules ou virgules si pas de formatage en liste
           attributes = re.split(r'[;,]', attr_text)
           result["attributes"] = [attr.strip() for attr in attributes if attr.strip()]

   # Recherche de concepts liés
   related_match = re.search(r"(related concepts|related categories|similar to):(.*?)(?:\n\n|\n[A-Z]|$)",
                             response, re.IGNORECASE | re.DOTALL)
   if related_match:
       related_text = related_match.group(2).strip()
       related = re.findall(r"[-•*]\s*([^\n]+)", related_text)
       if related:
           result["related_concepts"] = [rel.strip() for rel in related]
       else:
           related = re.split(r'[;,]', related_text)
           result["related_concepts"] = [rel.strip() for rel in related if rel.strip()]

   # Recherche d'exemples
   examples_match = re.search(r"(examples|instances|such as):(.*?)(?:\n\n|\n[A-Z]|$)",
                              response, re.IGNORECASE | re.DOTALL)
   if examples_match:
       examples_text = examples_match.group(2).strip()
       examples = re.findall(r"[-•*]\s*([^\n]+)", examples_text)
       if examples:
           result["examples"] = [ex.strip() for ex in examples]
       else:
           examples = re.split(r'[;,]', examples_text)
           result["examples"] = [ex.strip() for ex in examples if ex.strip()]

   return result

def parse_relation_validation(response: str) -> Dict[str, Any]:
   """
   Parse la réponse d'une validation de relation.
   Extrait si la relation est valide, la confiance, et l'explication.
   """
   result = {
       "is_valid": False,
       "confidence": 0.0,
       "explanation": ""
   }

   # Recherche si la relation est valide (oui/non)
   valid_match = re.search(r"(valid|relationship is valid|is valid):\s*(yes|no|true|false)",
                           response, re.IGNORECASE)
   if valid_match:
       valid_text = valid_match.group(2).lower()
       result["is_valid"] = valid_text in ["yes", "true"]

   # Recherche du niveau de confiance
   conf_match = re.search(r"(confidence|confidence level):\s*([0-9.]+)", response, re.IGNORECASE)
   if conf_match:
       try:
           conf_value = float(conf_match.group(2))
           if 0 <= conf_value <= 1:
               result["confidence"] = conf_value
       except ValueError:
           pass

   # Recherche de l'explication
   expl_match = re.search(r"(explanation|reasoning|rationale):\s*([^\n]+)", response, re.IGNORECASE)
   if expl_match:
       result["explanation"] = expl_match.group(2).strip()
   else:
       # Chercher des blocs qui pourraient contenir l'explication
       expl_section = re.search(r"(?:^|\n)([^:]+(?:\n[^:]+)+)", response)
       if expl_section:
           result["explanation"] = expl_section.group(1).strip()

   return result

def parse_chain_of_thought(response: str) -> Dict[str, Any]:
   """
   Parse une réponse de type "chaîne de raisonnement".
   Extrait les étapes, la conclusion et la confiance.
   """
   result = {
       "steps": [],
       "conclusion": "",
       "confidence": 0.0
   }

   # Extraire les étapes numérotées
   steps = re.findall(r"(?:\n|\A)(\d+)\.\s+([^\n]+)(?:\n|$)", response)
   if steps:
       result["steps"] = [step[1].strip() for step in steps]

   # Chercher la conclusion
   conclusion_match = re.search(r"(?:conclusion|therefore|thus|in summary|to summarize):\s*([^\n]+)",
                               response, re.IGNORECASE)
   if conclusion_match:
       result["conclusion"] = conclusion_match.group(1).strip()
   else:
       # Prendre la dernière phrase ou le dernier paragraphe comme conclusion
       paragraphs = response.split("\n\n")
       if paragraphs:
           last_para = paragraphs[-1].strip()
           if last_para and not any(last_para.startswith(f"{i}.") for i in range(1, 10)):
               result["conclusion"] = last_para

   # Chercher le niveau de confiance
   conf_match = re.search(r"(confidence|certainty):\s*([0-9.]+)", response, re.IGNORECASE)
   if conf_match:
       try:
           conf_value = float(conf_match.group(2))
           if 0 <= conf_value <= 1:
               result["confidence"] = conf_value
           else:
               # Normaliser si la confiance est exprimée en pourcentage
               result["confidence"] = min(conf_value / 100, 1.0)
       except ValueError:
           pass
   else:
       # Estimer la confiance en fonction des mots-clés
       confidence_words = {
           "certain": 1.0, "definitely": 1.0, "absolutely": 1.0,
           "likely": 0.8, "probably": 0.7, "most likely": 0.8,
           "possibly": 0.5, "perhaps": 0.4, "maybe": 0.4,
           "uncertain": 0.3, "unclear": 0.2, "doubtful": 0.1
       }

       for word, value in confidence_words.items():
           if re.search(r'\b' + re.escape(word) + r'\b', response, re.IGNORECASE):
               result["confidence"] = max(result["confidence"], value)

   return result

def parse_structured_json(response: str) -> Dict[str, Any]:
   """
   Tente d'extraire un objet JSON structuré de la réponse.
   Robuste aux divers formats ou textes additionnels.
   """
   # Rechercher un bloc de code JSON encadré par ```json...```
   json_block_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response)
   if json_block_match:
       json_str = json_block_match.group(1)
       try:
           return json.loads(json_str)
       except json.JSONDecodeError:
           pass

   # Chercher un bloc qui ressemble à du JSON même sans backticks
   json_obj_match = re.search(r'({[\s\S]*})', response)
   if json_obj_match:
       json_str = json_obj_match.group(1)
       try:
           return json.loads(json_str)
       except json.JSONDecodeError:
           pass

   # Dernier recours: essayer de traiter toute la réponse comme du JSON
   try:
       return json.loads(response)
   except json.JSONDecodeError:
       # Échec de l'extraction - renvoyer un dictionnaire vide
       return {}

def parse_entity_extraction(response: str) -> List[Dict[str, Any]]:
   """
   Extrait des entités nommées d'une réponse, avec leurs types et attributs.
   """
   entities = []

   # Essayer d'abord de parser un JSON structuré
   try:
       json_data = parse_structured_json(response)
       if isinstance(json_data, dict) and "entities" in json_data:
           return json_data["entities"]
       elif isinstance(json_data, list):
           return json_data
   except Exception:
       pass

   # Rechercher des sections d'entités
   entity_sections = re.findall(r"(?:^|\n)(?:Entity|Name):\s*([^\n]+)(?:\n(?:[^:\n]+:)?[^\n]*)*", response)

   for section in entity_sections:
       entity = {"name": section.strip(), "type": "UNKNOWN", "attributes": {}}

       # Chercher le type
       type_match = re.search(r"Type:\s*([^\n]+)", section)
       if type_match:
           entity["type"] = type_match.group(1).strip()

       # Chercher les attributs
       attr_matches = re.findall(r"([A-Za-z0-9_]+):\s*([^\n]+)", section)
       for attr_name, attr_value in attr_matches:
           if attr_name.lower() not in ["entity", "name", "type"]:
               entity["attributes"][attr_name] = attr_value.strip()

       entities.append(entity)

   # Si aucune entité n'a été trouvée, essayer une approche plus simple
   if not entities:
       simple_entities = re.findall(r"[-•*]\s*([^\n:]+)(?:\s*\(([^)]+)\))?", response)
       for entity_name, entity_type in simple_entities:
           entity = {
               "name": entity_name.strip(),
               "type": entity_type.strip() if entity_type else "UNKNOWN",
               "attributes": {}
           }
           entities.append(entity)

   return entities

def parse_classification(response: str) -> Dict[str, Any]:
   """
   Parse une réponse de classification.
   Extrait la classe, la probabilité et la justification.
   """
   result = {
       "class": "",
       "probability": 0.0,
       "explanation": ""
   }

   # Recherche de la classe
   class_match = re.search(r"(?:class|category|classification|label):\s*([^\n.]+)", response, re.IGNORECASE)
   if class_match:
       result["class"] = class_match.group(1).strip()

   # Recherche de la probabilité
   prob_match = re.search(r"(?:probability|confidence|certainty):\s*([0-9.]+)(?:%|)", response, re.IGNORECASE)
   if prob_match:
       try:
           prob_value = float(prob_match.group(1))
           # Normaliser si c'est un pourcentage
           if prob_value > 1 and prob_value <= 100:
               result["probability"] = prob_value / 100
           else:
               result["probability"] = prob_value
       except ValueError:
           pass

   # Recherche de l'explication
   expl_match = re.search(r"(?:explanation|reasoning|justification):\s*(.+?)(?:\n\n|$)",
                          response, re.IGNORECASE | re.DOTALL)
   if expl_match:
       result["explanation"] = expl_match.group(1).strip()

   return result

def parse_math_expression_interpretation(response: str) -> Dict[str, Any]:
   """
   Parse l'interprétation d'une expression mathématique.
   """
   result = {
       "interpretation": "",
       "variables": {},
       "simplified_form": "",
       "confidence": 0.0
   }

   # Extraire l'interprétation principale
   interp_match = re.search(r"(?:interpretation|meaning|represents):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                           response, re.IGNORECASE | re.DOTALL)
   if interp_match:
       result["interpretation"] = interp_match.group(1).strip()

   # Extraire les variables
   vars_section = re.search(r"(?:variables|where|symbols):\s*((?:[^\n]+\n?)+)", response, re.IGNORECASE)
   if vars_section:
       var_defs = re.findall(r"([a-zA-Z0-9_]+)\s*(?:is|represents|=|:)\s*([^\n,;]+)[,;]?",
                            vars_section.group(1))
       for var_name, var_meaning in var_defs:
           result["variables"][var_name.strip()] = var_meaning.strip()

   # Extraire la forme simplifiée
   simple_match = re.search(r"(?:simplified|simplifies to|reduced form):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                           response, re.IGNORECASE | re.DOTALL)
   if simple_match:
       result["simplified_form"] = simple_match.group(1).strip()

   # Extraire la confiance
   conf_match = re.search(r"(?:confidence|certainty):\s*([0-9.]+)", response, re.IGNORECASE)
   if conf_match:
       try:
           result["confidence"] = float(conf_match.group(1))
       except ValueError:
           pass

   return result

def parse_code_interpretation(response: str) -> Dict[str, Any]:
   """
   Parse l'interprétation d'un fragment de code.
   """
   result = {
       "purpose": "",
       "key_components": [],
       "potential_issues": [],
       "complexity": "",
       "confidence": 0.5
   }

   # Rechercher l'objectif principal
   purpose_match = re.search(r"(?:purpose|function|goal):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                            response, re.IGNORECASE | re.DOTALL)
   if purpose_match:
       result["purpose"] = purpose_match.group(1).strip()

   # Rechercher les composants clés
   comp_match = re.search(r"(?:components|key elements|main parts):(.*?)(?:\n\n|\n[A-Z]|$)",
                         response, re.IGNORECASE | re.DOTALL)
   if comp_match:
       comp_text = comp_match.group(1).strip()
       components = re.findall(r"[-•*]\s*([^\n]+)", comp_text)
       if components:
           result["key_components"] = [comp.strip() for comp in components]
       else:
           # Séparer par points si pas de formatage en liste
           components = re.split(r'\.', comp_text)
           result["key_components"] = [comp.strip() for comp in components if comp.strip()]

   # Rechercher les problèmes potentiels
   issues_match = re.search(r"(?:issues|problems|bugs|concerns):(.*?)(?:\n\n|\n[A-Z]|$)",
                           response, re.IGNORECASE | re.DOTALL)
   if issues_match:
       issues_text = issues_match.group(1).strip()
       issues = re.findall(r"[-•*]\s*([^\n]+)", issues_text)
       if issues:
           result["potential_issues"] = [issue.strip() for issue in issues]
       else:
           issues = re.split(r'\.', issues_text)
           result["potential_issues"] = [issue.strip() for issue in issues if issue.strip()]

   # Rechercher la complexité
   complex_match = re.search(r"(?:complexity|efficiency):\s*(.+?)(?:\n\n|\n[A-Z]|$)",
                            response, re.IGNORECASE | re.DOTALL)
   if complex_match:
       result["complexity"] = complex_match.group(1).strip()

   # Rechercher la confiance
   conf_match = re.search(r"(?:confidence|certainty):\s*([0-9.]+)", response, re.IGNORECASE)
   if conf_match:
       try:
           result["confidence"] = float(conf_match.group(1))
       except ValueError:
           pass

   return result