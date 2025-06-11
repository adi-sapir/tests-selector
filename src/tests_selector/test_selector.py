import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_vector_store(vector_store_dir: Path) -> Tuple[faiss.Index, List[Dict]]:
    """
    Load the FAISS index and test metadata from the vector store.
    
    Args:
        vector_store_dir: Directory containing the vector store
        
    Returns:
        Tuple of (FAISS index, test metadata)
    """
    if not vector_store_dir.exists():
        raise FileNotFoundError(f"Vector store directory not found: {vector_store_dir}")
    
    index_path = vector_store_dir / "test_embeddings.faiss"
    metadata_path = vector_store_dir / "test_metadata.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Test metadata not found: {metadata_path}")
    
    # Load FAISS index
    index = faiss.read_index(str(index_path))
    
    # Load test metadata
    with metadata_path.open('r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return index, metadata

def create_change_embedding(client: OpenAI, change_description: str) -> np.ndarray:
    """
    Create an embedding for the code change description.
    
    Args:
        client: OpenAI client instance
        change_description: Description of code changes
        
    Returns:
        Numpy array containing the embedding
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=change_description
        )
        return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        raise

def find_similar_tests(index: faiss.Index, query_embedding: np.ndarray, 
                      metadata: List[Dict], k: int = 30) -> List[Dict]:
    """
    Find the k most similar tests using the FAISS index.
    Uses Inner Product similarity (equivalent to cosine similarity for normalized vectors),
    where higher scores indicate more similar items.
    
    Args:
        index: FAISS index
        query_embedding: Query embedding
        metadata: Test metadata
        k: Number of similar tests to return
        
    Returns:
        List of similar tests with distances
    """
    # Ensure k doesn't exceed the number of tests
    k = min(k, index.ntotal)
    
    # Search the index
    scores, indices = index.search(query_embedding, k)
    
    # Combine results with metadata
    similar_tests = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        test = metadata[idx].copy()
        # With IndexFlatIP, the score is already a similarity measure (higher is better)
        test['similarity_score'] = float(score)
        similar_tests.append(test)
    
    # Sort by similarity score in descending order (higher is better)
    similar_tests.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Print unique files with their highest similarity scores
    file_scores = {}
    for test in similar_tests:
        file = test['file']
        score = test['similarity_score']
        # Keep the highest score for each file
        if file not in file_scores or score > file_scores[file]:
            file_scores[file] = score
    
    # Sort files by their highest similarity score
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {k} most similar tests are from {len(sorted_files)} files:")
    for file, score in sorted_files:
        print(f"  {score:.3f} - {file}")
    
    return similar_tests

def format_tests_for_prompt(tests: List[Dict]) -> str:
    """
    Format tests into a string suitable for the LLM prompt.
    
    Args:
        tests: List of test scenarios
        
    Returns:
        Formatted string describing all tests
    """
    formatted_tests = []
    for test in tests:
        test_str = f"Feature: {test['feature']}\n"
        test_str += f"Scenario: {test['scenario']}\n"
        if test.get('tags'):
            test_str += f"Tags: {', '.join(test['tags'])}\n"
        test_str += "Steps:\n"
        for step in test['steps']:
            test_str += f"  {step}\n"
        test_str += f"File: {test['file']}\n"
        test_str += f"Similarity Score: {test.get('similarity_score', 'N/A')}\n"
        formatted_tests.append(test_str)
    
    return "\n---\n".join(formatted_tests)

def create_prompt(change_description: str, formatted_tests: str) -> str:
    """
    Create the prompt for the LLM.
    
    Args:
        change_description: Description of code changes
        formatted_tests: Formatted string of all tests
        
    Returns:
        Complete prompt for the LLM
    """
    return f"""As a test selection expert, analyze the following code change and the most relevant tests (already pre-selected based on semantic similarity) to make the final selection of which tests should be run.

Code Change Description:
{change_description}

Pre-selected Tests (ordered by relevance):
{formatted_tests}

Based on the code change description and the pre-selected tests, identify which tests should be run to verify the changes.
Consider the similarity scores but make your final selection based on your understanding of the tests' purposes and their relation to the changes.

You must respond with valid JSON only, using this exact format:
{{
    "selected_tests": [
        {{
            "file": "path/to/test.feature",
            "scenario": "name of scenario",
            "reason": "explanation of why this test should be run",
            "similarity_score": 0.85  // Include the original similarity score
        }}
    ]
}}

Only include tests that are directly relevant to verifying the described changes.
Do not include any text before or after the JSON.
The response must be parseable by json.loads()."""

def convert_similar_tests_to_results(similar_tests: List[Dict]) -> Dict:
    """
    Convert the similar tests list to the same format as LLM results.
    
    Args:
        similar_tests: List of tests with similarity scores
        
    Returns:
        Dictionary in the same format as LLM results
    """
    return {
        "selected_tests": [
            {
                "file": test["file"],
                "scenario": test["scenario"],
                "reason": "Selected based on vector similarity score",
                "similarity_score": test["similarity_score"]
            }
            for test in similar_tests
        ]
    }

def select_tests(client: OpenAI, change_description: str, vector_store_dir: Path, use_llm: bool = True, k: int = 30) -> Dict:
    """
    Select relevant tests using a combination of vector similarity and optionally LLM refinement.
    
    Args:
        client: OpenAI client instance
        change_description: Description of code changes
        vector_store_dir: Directory containing the vector store
        use_llm: Whether to use LLM for refinement
        k: Number of similar tests to select
        
    Returns:
        Dictionary containing selected tests and reasons
    """
    # Load vector store
    logger.info("Loading vector store...")
    index, metadata = load_vector_store(vector_store_dir)
    
    # Create embedding for change description
    logger.info("Creating embedding for change description...")
    query_embedding = create_change_embedding(client, change_description)
    
    # Find similar tests using FAISS
    logger.info(f"Finding top {k} similar tests...")
    similar_tests = find_similar_tests(index, query_embedding, metadata, k=k)
    logger.info(f"Found {len(similar_tests)} similar tests")
    
    if not use_llm:
        logger.info("Skipping LLM refinement as requested")
        return convert_similar_tests_to_results(similar_tests)
    
    # Format tests for the prompt
    formatted_tests = format_tests_for_prompt(similar_tests)
    prompt = create_prompt(change_description, formatted_tests)
    
    # Print prompt in debug mode
    if logger.getEffectiveLevel() <= logging.DEBUG:
        print("\n=== LLM Prompt ===")
        print(prompt)
        print("=== End Prompt ===\n")
    
    # Get refined selection from OpenAI
    logger.info("Refining test selection using LLM...")
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for a cheaper but less capable model
            messages=[
                {"role": "system", "content": "You are a test selection expert that helps identify which tests should be run for code changes. You must respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # Use deterministic output
        )
        
        # Extract the JSON response
        response_text = response.choices[0].message.content.strip()
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {response_text}")
            raise ValueError("LLM response was not valid JSON") from e
            
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        raise

def save_results(results: Dict, output_file: Path):
    """
    Save the selected tests to a file.
    
    Args:
        results: Dictionary containing selected tests
        output_file: Path to save the results
    """
    try:
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {str(e)}")
        raise

def print_selected_files(results: Dict):
    """
    Print unique list of files that contain selected tests.
    
    Args:
        results: Dictionary containing selected tests
    """
    selected_files = {test['file'] for test in results['selected_tests']}
    if selected_files:
        print("\nSelected test files:")
        for file in sorted(selected_files):
            print(f"  - {file}")
    else:
        print("No test files were selected.")

def print_selected_tests(results: Dict):
    """
    Print detailed information about selected tests.
    
    Args:
        results: Dictionary containing selected tests
    """
    tests = results['selected_tests']
    if not tests:
        print("\nNo tests were selected.")
        return
        
    print("\nSelected tests:")
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. Test:")
        print(f"   File: {test['file']}")
        print(f"   Scenario: {test['scenario']}")
        print(f"   Reason: {test['reason']}")
        if 'similarity_score' in test:
            print(f"   Similarity Score: {test['similarity_score']:.3f}")

def chat_mode(vector_store_dir: Path, use_llm: bool = True, k: int = 30):
    """
    Run the test selector in interactive chat mode.
    
    Args:
        vector_store_dir: Directory containing the vector store
        use_llm: Whether to use LLM for refinement
        k: Number of similar tests to select
    """
    try:
        # Check for OpenAI API key only if using LLM
        if use_llm:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable not set")
                return
            
        client = OpenAI()
        
        print("\nWelcome to Test Selector Chat Mode!")
        print("Enter your code change description below.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                # Get user input
                print("\nDescribe your code changes:")
                change_description = input("> ").strip()
                
                # Check for exit command
                if change_description.lower() in ['exit', 'quit']:
                    print("\nExiting chat mode. Goodbye!")
                    break
                
                if not change_description:
                    print("Please provide a description of your code changes.")
                    continue
                
                # Process the query
                print("\nAnalyzing code changes...")
                results = select_tests(client, change_description, vector_store_dir, use_llm=use_llm, k=k)
                
                # Print results
                print_selected_tests(results)
                print_selected_files(results)
                
            except KeyboardInterrupt:
                print("\nExiting chat mode. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print("Please try again with a different description.")
                
    except Exception as e:
        logger.error(f"Error in chat mode: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Select relevant tests based on code changes using vector similarity and LLM refinement."
    )
    parser.add_argument(
        "change_description",
        nargs="?",  # Make positional argument optional
        help="Description of the code changes"
    )
    parser.add_argument(
        "-V", "--vector-store-dir",
        default="vector_store",
        help="Directory containing the vector store (default: vector_store)"
    )
    parser.add_argument(
        "-o", "--output",
        default="selected_tests.json",
        help="Path to save selected tests (default: selected_tests.json)"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=30,
        help="Number of similar tests to select (default: 30)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "-chat", "--chat-mode",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "-nollm", "--no-llm",
        action="store_true",
        help="Skip LLM refinement and use vector similarity results directly"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        vector_store_dir = Path(args.vector_store_dir)
        
        if args.chat_mode:
            if args.no_llm:
                print("\nNote: Running in chat mode with LLM refinement disabled")
            chat_mode(vector_store_dir, use_llm=not args.no_llm, k=args.k)
        else:
            if not args.change_description:
                parser.error("change_description is required when not in chat mode")
            
            # Check for OpenAI API key only if using LLM
            if not args.no_llm:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error("OPENAI_API_KEY environment variable not set")
                    sys.exit(1)
            
            client = OpenAI()
            output_file = Path(args.output)
            
            logger.info("Starting test selection process...")
            results = select_tests(client, args.change_description, vector_store_dir, use_llm=not args.no_llm, k=args.k)
            
            logger.info("Saving results...")
            save_results(results, output_file)
            
            print(f"\nSelected {len(results['selected_tests'])} tests")
            print_selected_tests(results)
            print_selected_files(results)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 