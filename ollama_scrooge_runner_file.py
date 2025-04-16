### Convert Extracted Entities into a Graph


import networkx as nx
import json
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Read the medical text file
with open("report_0.txt", "r") as file:
    text = file.read()

# Process text using spaCy
doc = nlp(text)

# Create a directed graph
G = nx.DiGraph()

# Add patient node
patient_id = "Patient_76M"
G.add_node(patient_id, type="Patient", age=76, gender="Male")

# Extract and add diseases
diseases = [
    "coronary artery disease",
    "congestive heart failure",
    "type 2 diabetes",
    "hypertension",
    "diverticulosis",
    "alzheimer's dementia",
    "gastrointestinal bleed",
    "hypercholesterolemia",
    "peripheral vascular disease",
]

for disease in diseases:
    G.add_node(disease, type="Disease")
    G.add_edge(patient_id, disease, relation="has_condition")

# Extract and add medications
medications = [
    "vancomycin",
    "levofloxacin",
    "metronidazole",
    "heparin",
    "simvastatin",
    "lisinopril",
    "furosemide",
    "vitamin E",
    "atenolol",
    "pantoprazole",
    "ascorbic acid",
    "insulin",
    "bisacodyl",
    "docusate",
    "percocet",
    "aspirin",
    "metoprolol",
]

for med in medications:
    G.add_node(med, type="Medication")
    G.add_edge(patient_id, med, relation="prescribed")

# Extract and add vitals
vitals = {
    "Blood Pressure": "124/42",
    "Heart Rate": 83,
    "Respiratory Rate": 24,
    "Oxygen Saturation": "100%",
    "Temperature": "96.1F",
}

for vital, value in vitals.items():
    G.add_node(vital, type="Vital", value=value)
    G.add_edge(patient_id, vital, relation="has_vital")

# Save as GraphML (XML format)
graphml_file = "neo4j_graph/medical_graph.graphml"
nx.write_graphml(G, graphml_file)

print(f"GraphML file saved as {graphml_file}")



#### Convert GraphML (XML) to JSON

# Load the GraphML file
G = nx.read_graphml("neo4j_graph/medical_graph.graphml")

# Convert to JSON format
graph_data = {
    "nodes": [{"id": node, "type": G.nodes[node]["type"]} for node in G.nodes()],
    "edges": [{"source": u, "target": v, "relation": G[u][v]["relation"]} for u, v in G.edges()]
}

# Save as JSON
json_path = "neo4j_graph/medical_graph.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(graph_data, f, indent=2)

print(f"JSON file saved as {json_path}")
