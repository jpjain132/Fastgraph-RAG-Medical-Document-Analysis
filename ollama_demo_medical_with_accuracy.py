import instructor  # Structured outputs for LLMs
import os
import re
from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Define the domain for analysis
DOMAIN = "Analyze these clinical records and identify key medical entities. Focus on patient demographics, diagnoses, procedures, lab results, and outcomes."

# Example queries for GraphRAG
EXAMPLE_QUERIES = [
    "What are the common risk factors for sepsis in ICU patients?",
    "How do trends in lab results correlate with patient outcomes in cases of acute kidney injury?",
    "Describe the sequence of interventions for patients undergoing major cardiac surgery.",
    "How do patient demographics and comorbidities influence treatment decisions in the ICU?",
    "What patterns of medication usage are observed among patients with chronic obstructive pulmonary disease (COPD)?"
]

# Define entity types for the knowledge graph
ENTITY_TYPES = ["Patient", "Diagnosis", "Procedure", "Lab Test", "Medication", "Outcome"]

# Define working directory
working_dir = "./WORKING_DIR/mimic_ex500/"

# Initialize GraphRAG
grag = GraphRAG(
    working_dir=working_dir,
    n_checkpoints=2,
    domain=DOMAIN,
    example_queries="\n".join(EXAMPLE_QUERIES),
    entity_types=ENTITY_TYPES,
    config=GraphRAG.Config(
        llm_service=OpenAILLMService(
            model="llama3.2",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            mode=instructor.Mode.JSON,
            client="openai",
        ),
        embedding_service=OpenAIEmbeddingService(
            model="nomic-embed-text",  
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            embedding_dim=768,  
            client="openai"
        ),
    ),
)

embedding_service = OpenAIEmbeddingService(
    model="nomic-embed-text",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    embedding_dim=768,  # Check if this matches the actual model output
    client="openai"
)

# Correct method to generate embeddings
sample_embedding = embedding_service.get_embedding("test text")
print(f"Embedding dimension: {len(sample_embedding)}")  # Check output size

# Path to directory containing medical text files
directory_path = r"E:\semester 4\FastGraphRAG-Medical-Document-Analysis\mimic_ex_500"

# Function to clean and preprocess the medical text
def clean_medical_text(text):
    """
    Cleans and structures medical text to extract relevant information
    for GraphRAG processing.
    """
    # Remove special characters and multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Extract key sections from the text using regex patterns
    sections = {
        "medical_history": re.search(r"past medical history:(.*?)(?:social history:|physical exam:)", text, re.I),
        "social_history": re.search(r"social history:(.*?)(?:family history:|physical exam:)", text, re.I),
        "medications": re.search(r"medications on admission:(.*?)(?:discharge medications:|discharge disposition:)", text, re.I),
        "lab_results": re.search(r"pertinent results:(.*?)(?:brief hospital course:|discharge diagnosis:)", text, re.I),
        "diagnoses": re.search(r"discharge diagnosis:(.*?)(?:discharge condition:|discharge instructions:)", text, re.I),
    }

    # Convert found sections to a structured dictionary
    structured_data = {key: (match.group(1).strip() if match else "") for key, match in sections.items()}
    
    return structured_data

# Function to index medical records into GraphRAG
def graph_index(directory_path):
    file_count = 0  # Track processed files
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # Clean and preprocess the medical text
                structured_data = clean_medical_text(content)

                # Ensure all fields contain valid strings (avoid None errors)
                for key in structured_data:
                    structured_data[key] = structured_data[key] or "Unknown"

                # Insert structured data into GraphRAG
                grag.insert(str(structured_data))

            file_count += 1
            total_files = sum(1 for f in os.listdir(directory_path) if f.endswith(".txt"))
            print("******************** $$$$$$ *****************")
            print(f"Total Files Processed: -> {file_count} / {total_files}")
            print("******************** $$$$$$ *****************")

graph_index(directory_path)

# Save the processed data to GraphML format for Neo4j
print("**********************************************")
os.makedirs("neo4j_graph", exist_ok=True)
grag.save_graphml(output_path="neo4j_graph/oxford_graph_chunk_entity_relation.graphml")

# Query the processed data
query_result = grag.query("What are the most common treatments for cardiogenic shock in patients with a history of stroke?")
print(query_result.response)

# ------------ Evaluation Section (Mock labels for metrics) ------------ #

# Mock true labels vs. predicted labels (replace with your actual data)
true_labels = ["TreatmentA", "TreatmentB", "TreatmentC", "TreatmentA", "TreatmentD"]
predicted_labels = ["TreatmentA", "TreatmentB", "TreatmentC", "TreatmentX", "TreatmentD"]

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

# Calculate false positives manually
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=list(set(true_labels + predicted_labels)))
false_positives = conf_matrix.sum(axis=0) - np.diag(conf_matrix)

print("\n================ NLP METRICS ==================")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Precision        : {precision:.4f}")
print(f"Recall           : {recall:.4f}")
print(f"F1-Score         : {f1:.4f}")
print(f"False Positives  : {sum(false_positives)}")
print("===============================================")
