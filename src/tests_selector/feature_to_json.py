import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    parser = argparse.ArgumentParser(description="Parse Cucumber .feature files into JSON.")
    parser.add_argument("input_dir", help="Directory containing .feature files")
    parser.add_argument("output_json", help="Path to save the output JSON file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        logger.info(f"Starting scan of directory: {args.input_dir}")
        parsed_data = scan_directory_for_features(args.input_dir)

        logger.info(f"Found {len(parsed_data)} total scenarios")
        save_to_json(parsed_data, args.output_json)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise
