from neo4j import GraphDatabase

# Replace these with your Neo4j Aura details
NEO4J_URI = "neo4j+s://3bc37e79.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "WZZC0lOs9zfsMz3xRoHY96BBq69lfCBaFPc38vSzqIc"

# Function to test connection
def test_connection():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j Connection Successful!' AS message")
            print(result.single()["message"])
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        driver.close()

# Run the function
test_connection()
