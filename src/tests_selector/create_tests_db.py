"""
Create a test database from Cucumber feature files.

This script:
1. Recursively scans directories for .feature files
2. Parses the feature files into structured test data
3. Creates embeddings for each test using OpenAI's API
4. Builds a FAISS vector database for efficient similarity search
5. Saves both the vector database and test metadata
"""

import os
import json
import re
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss
from tqdm import tqdm
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_embedding(client: OpenAI, test: Dict) -> Optional[np.ndarray]:
    """
    Create an embedding for a test using OpenAI's API.
    
    Args:
        client: OpenAI client instance
        test: Test dictionary containing feature, scenario, and steps
        
    Returns:
        Numpy array containing the embedding or None if failed
    """
    # Format test as a string
    test_str = f"Feature: {test['feature']}\n"
    test_str += f"Scenario: {test['scenario']}\n"
    if test.get('tags'):
        test_str += f"Tags: {', '.join(test['tags'])}\n"
    test_str += "Steps:\n"
    for step in test['steps']:
        test_str += f"  {step}\n"

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=test_str
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        return None

def create_vector_store(tests: List[Dict], output_dir: Path) -> Tuple[List[Dict], np.ndarray]:
    """
    Create a FAISS vector store from test embeddings.
    Uses Inner Product similarity (equivalent to cosine similarity for normalized vectors)
    which is more appropriate for OpenAI's normalized embeddings than L2 distance.
    
    Args:
        tests: List of test dictionaries
        output_dir: Directory to save the vector store
        
    Returns:
        Tuple of (tests with embeddings, embedding matrix)
    """
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI()
    
    # Create embeddings for each test
    embeddings = []
    valid_tests = []
    
    logger.info("Creating embeddings for tests...")
    for test in tqdm(tests, desc="Creating embeddings"):
        embedding = create_test_embedding(client, test)
        if embedding is not None:
            embeddings.append(embedding)
            valid_tests.append(test)
    
    if not embeddings:
        raise ValueError("No valid embeddings created")
    
    # Convert to numpy array
    embedding_matrix = np.array(embeddings, dtype=np.float32)
    
    # Verify embeddings are normalized (they should be from OpenAI)
    norms = np.linalg.norm(embedding_matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4):
        logger.warning("Embeddings are not normalized. Normalizing...")
        embedding_matrix = embedding_matrix / norms[:, np.newaxis]
    
    # Create and train FAISS index
    # Using IndexFlatIP (Inner Product) which is equivalent to cosine similarity
    # for normalized vectors, better for semantic similarity than L2 distance
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)
    
    # Save the index and metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "test_embeddings.faiss"))
    
    # Save the test metadata separately
    with (output_dir / "test_metadata.json").open('w', encoding='utf-8') as f:
        json.dump(valid_tests, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created vector store with {len(valid_tests)} tests")
    return valid_tests, embedding_matrix

def parse_feature_file(filepath: Path) -> Optional[List[Dict]]:
    """
    Parse a single feature file into a list of test scenarios.
    
    Args:
        filepath: Path to the feature file
        
    Returns:
        List of parsed scenarios or None if parsing fails
    """
    try:
        with filepath.open('r', encoding='utf-8') as file:
            lines = file.readlines()

        parsed_tests = []
        current_feature = None
        current_scenario = None
        current_steps = []
        current_tags = []
        line_number = 0

        for line in lines:
            line_number += 1
            stripped = line.strip()

            if not stripped:  # Skip empty lines
                continue

            if stripped.startswith('@'):
                current_tags = stripped.split()

            elif stripped.lower().startswith('feature:'):
                current_feature = stripped[len('feature:'):].strip()

            elif stripped.lower().startswith('scenario') or stripped.lower().startswith('scenario outline'):
                if current_scenario:
                    parsed_tests.append({
                        "file": str(filepath),
                        "feature": current_feature,
                        "scenario": current_scenario,
                        "steps": current_steps,
                        "tags": current_tags,
                        "line": line_number
                    })
                    current_steps = []
                    current_tags = []

                current_scenario = re.sub(r'^scenario( outline)?:', '', stripped, flags=re.IGNORECASE).strip()

            elif any(stripped.lower().startswith(prefix) for prefix in ('given', 'when', 'then', 'and', 'but')):
                current_steps.append(stripped)

        # Catch last scenario
        if current_scenario:
            parsed_tests.append({
                "file": str(filepath),
                "feature": current_feature,
                "scenario": current_scenario,
                "steps": current_steps,
                "tags": current_tags,
                "line": line_number
            })

        return parsed_tests

    except Exception as e:
        logger.error(f"Error parsing feature file {filepath}: {str(e)}")
        return None

def scan_directory_for_features(directory: str | Path) -> List[Dict]:
    """
    Recursively scan a directory for .feature files and parse them.
    
    Args:
        directory: Directory path to scan
        
    Returns:
        List of all parsed scenarios from all feature files
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    all_tests = []
    feature_files = list(directory.rglob("*.feature"))
    total_files = len(feature_files)
    
    logger.info(f"Found {total_files} feature files in {directory}")
    
    for i, feature_file in enumerate(feature_files, 1):
        logger.info(f"Processing file {i}/{total_files}: {feature_file}")
        parsed = parse_feature_file(feature_file)
        if parsed:
            all_tests.extend(parsed)
        
    logger.info(f"Successfully parsed {len(all_tests)} scenarios from {total_files} files")
    return all_tests

def save_to_json(data: List[Dict], output_path: str | Path):
    """
    Save parsed data to a JSON file.
    
    Args:
        data: List of parsed scenarios
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved output to {output_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {output_path}: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a test database from Cucumber feature files with vector embeddings for similarity search."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing .feature files to process"
    )
    parser.add_argument(
        "--vector-store-dir", 
        default="vector_store",
        help="Directory to save the vector database (default: vector_store)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        logger.info(f"Starting scan of directory: {args.input_dir}")
        parsed_data = scan_directory_for_features(args.input_dir)
        logger.info(f"Found {len(parsed_data)} total scenarios")
        
        # Create vector store
        vector_store_dir = Path(args.vector_store_dir)
        logger.info(f"Creating vector database in {vector_store_dir}")
        valid_tests, embeddings = create_vector_store(parsed_data, vector_store_dir)
        
        logger.info(f"Successfully created test database with {len(valid_tests)} tests")
        logger.info(f"Database location: {vector_store_dir}")
        logger.info("Files created:")
        logger.info(f"  - {vector_store_dir}/test_embeddings.faiss (vector database)")
        logger.info(f"  - {vector_store_dir}/test_metadata.json (test descriptions)")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise
