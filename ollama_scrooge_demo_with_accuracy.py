import instructor  # Structured outputs for LLMs
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

# Define the story domain
DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]

ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

working_dir = "./WORKING_DIR/carol/test"

grag = GraphRAG(
    working_dir=working_dir,
    domain=DOMAIN,
    example_queries="\n".join(EXAMPLE_QUERIES),
    entity_types=ENTITY_TYPES,
    config=GraphRAG.Config(
        llm_service=OpenAILLMService(
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            mode=instructor.Mode.JSON,
            client="openai",
        ),
        embedding_service=OpenAIEmbeddingService(
            model="mxbai-embed-large",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            embedding_dim=1024,
        )
    )
)

# Inserting the book into the graph
with open("./book.txt", encoding="utf-8") as f:
    grag.insert(f.read())

# Perform queries
print("**********************************************")
response1 = grag.query("Who is Scrooge?").response
print(response1)

print("**********************************************")
response2 = grag.query("List all the characters?").response
print(response2)

print("**********************************************")
response3 = grag.query("What are the odd events in the story?").response
print(response3)

print("**********************************************")
response4 = grag.query("What is the overall theme of the story?").response
print(response4)

# ==============================
# EVALUATION METRICS SECTION
# ==============================

# Sample ground truth and predictions
# You should adapt this based on actual dataset or manual tagging
true_entities = ["Scrooge", "Tiny Tim", "Bob Cratchit", "Ghost of Christmas Past", "Ghost of Christmas Present", "Ghost of Christmas Yet to Come"]
predicted_entities = response2.split(",")  # crude entity extraction; ideally use proper NER logic

# Clean whitespace
true_entities = [e.strip().lower() for e in true_entities]
predicted_entities = [e.strip().lower() for e in predicted_entities]

# Create binary labels for evaluation
all_possible = list(set(true_entities + predicted_entities))
y_true = [1 if ent in true_entities else 0 for ent in all_possible]
y_pred = [1 if ent in predicted_entities else 0 for ent in all_possible]

# Compute NLP Evaluation Metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Calculate False Positive Rate
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

print("\n=============== NLP Evaluation Metrics ===============")
print(f"True Entities (Ground Truth): {true_entities}")
print(f"Predicted Entities: {predicted_entities}")
print("------------------------------------------------------")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")
print("======================================================")
