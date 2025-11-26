#!/usr/bin/env python3
"""
Update metadata files to match the actual subfolder names.
This script reads the subfolder name and updates the metadata accordingly.
"""

import os
import json
import re
from pathlib import Path

def parse_subfolder_name(subfolder):
    """Parse subfolder name to extract parameters."""
    # Format: agents_{n}_method_{method}_mode_{mode}_questions_{n}
    pattern = r"agents_(\d+)_method_(\w+)_mode_(\w+)_questions_(\d+)"
    match = re.match(pattern, subfolder)
    if match:
        return {
            "n_agents": int(match.group(1)),
            "improvement_method": match.group(2),
            "user_mode": match.group(3),
            "n_questions": int(match.group(4))
        }
    return None

def update_metadata_file(metadata_file, subfolder, experiment_dir):
    """Update a single metadata file to match the subfolder."""
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  ❌ Error reading {metadata_file}: {e}")
        return False
    
    # Parse subfolder to get correct parameters
    params = parse_subfolder_name(subfolder)
    if not params:
        print(f"  ⚠️  Could not parse subfolder name: {subfolder}")
        return False
    
    # Update metadata
    metadata["subfolder"] = subfolder
    metadata["user_mode"] = params["user_mode"]
    
    # Update experiment_params
    if "experiment_params" not in metadata:
        metadata["experiment_params"] = {}
    metadata["experiment_params"]["user_mode"] = params["user_mode"]
    metadata["experiment_params"]["n_questions_per_pair"] = params["n_questions"]
    metadata["experiment_params"]["n_agents"] = params["n_agents"]
    metadata["experiment_params"]["improvement_method"] = params["improvement_method"]
    
    # Update file paths in metadata
    base_name = os.path.basename(metadata_file).replace("_metadata.json", "")
    metadata["files"] = {
        "population": os.path.join(experiment_dir, f"{base_name}_population.pkl"),
        "matrix": os.path.join(experiment_dir, f"{base_name}_egs_matrix.npy"),
        "preferences": os.path.join(experiment_dir, f"{base_name}_user_prefs.json")
    }
    
    # Save updated metadata
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        print(f"  ❌ Error writing {metadata_file}: {e}")
        return False

def update_info_file(info_file, subfolder, experiment_dir):
    """Update experiment info file to match the subfolder."""
    if not os.path.exists(info_file):
        return False
    
    try:
        with open(info_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ Error reading {info_file}: {e}")
        return False
    
    # Parse subfolder to get correct parameters
    params = parse_subfolder_name(subfolder)
    if not params:
        return False
    
    # Update the content
    # Update user mode
    content = re.sub(
        r"User Mode: .*",
        f"User Mode: {params['user_mode'].upper()}",
        content
    )
    
    # Update questions per pair
    content = re.sub(
        r"Questions per Agent Pair \(for EGS matrix\): .*",
        f"Questions per Agent Pair (for EGS matrix): {params['n_questions']}",
        content
    )
    
    # Update subfolder reference
    content = re.sub(
        r"Subfolder: .*",
        f"Subfolder: {subfolder}",
        content
    )
    
    # Update experiment directory
    content = re.sub(
        r"Experiment Directory: .*",
        f"Experiment Directory: {experiment_dir}",
        content
    )
    
    # Save updated content
    try:
        with open(info_file, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"  ❌ Error writing {info_file}: {e}")
        return False

def main():
    save_dir = "out/llm_competition"
    
    if not os.path.exists(save_dir):
        print(f"Directory {save_dir} does not exist.")
        return
    
    print("=" * 60)
    print("Updating Metadata Files to Match Subfolder Names")
    print("=" * 60)
    print()
    
    # Find all subfolders
    subfolders = [d for d in os.listdir(save_dir) 
                  if os.path.isdir(os.path.join(save_dir, d)) and d.startswith("agents_")]
    
    if not subfolders:
        print("No experiment subfolders found.")
        return
    
    print(f"Found {len(subfolders)} experiment subfolder(s).\n")
    
    # Find all metadata files
    pattern = re.compile(r"^(llm_competition_\d{8}_\d{6})_metadata\.json$")
    updated_count = 0
    
    for subfolder in subfolders:
        experiment_dir = os.path.join(save_dir, subfolder)
        print(f"Processing {subfolder}...")
        
        for file in os.listdir(experiment_dir):
            match = pattern.match(file)
            if match:
                base_name = match.group(1)
                metadata_file = os.path.join(experiment_dir, file)
                info_file = os.path.join(experiment_dir, f"{base_name}_experiment_info.txt")
                
                # Update metadata
                if update_metadata_file(metadata_file, subfolder, experiment_dir):
                    print(f"  ✅ Updated metadata: {file}")
                    updated_count += 1
                
                # Update info file
                if update_info_file(info_file, subfolder, experiment_dir):
                    print(f"  ✅ Updated info file: {base_name}_experiment_info.txt")
        
        print()
    
    print("=" * 60)
    print(f"✅ Complete! Updated {updated_count} metadata file(s).")
    print("=" * 60)

if __name__ == "__main__":
    main()

