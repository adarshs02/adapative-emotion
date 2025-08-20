"""
Dial Cache Module for EmoBIRD

This module provides caching functionality for CPT (Conditional Probability Table) data,
allowing neutral dials (factor-value → emotion probability) to be built offline once
and loaded from cache at runtime.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


# Cache file location
CACHE_PATH = Path("data/cpt_cache.json")


def save_cpt(cpt: Dict[str, Any]) -> None:
    """
    Save CPT data to cache file.
    
    Args:
        cpt: CPT data dictionary to save
    """
    try:
        # Ensure the data directory exists
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CPT data to cache file with pretty formatting
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cpt, f, indent=2, ensure_ascii=False)
        
        print(f"💾 CPT data saved to cache: {CACHE_PATH}")
        print(f"   Cached {len(cpt.get('combinations', {}))} factor combinations")
        print(f"   Cached {len(cpt.get('emotions', []))} emotions")
        
    except Exception as e:
        print(f"❌ Error saving CPT to cache: {e}")
        raise


def load_cpt() -> Optional[Dict[str, Any]]:
    """
    Load CPT data from cache file.
    
    Returns:
        CPT data dictionary if cache exists and is valid, None otherwise
    """
    try:
        if not CACHE_PATH.exists():
            print(f"📁 No CPT cache found at {CACHE_PATH}")
            return None
        
        # Load CPT data from cache file
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            cpt_data = json.load(f)
        
        # Basic validation
        if not isinstance(cpt_data, dict):
            print("⚠️ Invalid CPT cache format: not a dictionary")
            return None
        
        if 'combinations' not in cpt_data or 'emotions' not in cpt_data:
            print("⚠️ Invalid CPT cache format: missing required keys")
            return None
        
        print(f"✅ CPT data loaded from cache: {CACHE_PATH}")
        print(f"   Loaded {len(cpt_data.get('combinations', {}))} factor combinations")
        print(f"   Loaded {len(cpt_data.get('emotions', []))} emotions")
        
        return cpt_data
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing CPT cache JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading CPT from cache: {e}")
        return None


def cache_exists() -> bool:
    """
    Check if CPT cache file exists.
    
    Returns:
        True if cache file exists, False otherwise
    """
    return CACHE_PATH.exists()


def clear_cache() -> None:
    """
    Clear the CPT cache by removing the cache file.
    """
    try:
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
            print(f"🗑️ CPT cache cleared: {CACHE_PATH}")
        else:
            print(f"📁 No CPT cache to clear at {CACHE_PATH}")
    except Exception as e:
        print(f"❌ Error clearing CPT cache: {e}")
        raise


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache.
    
    Returns:
        Dictionary with cache information
    """
    if not cache_exists():
        return {
            "exists": False,
            "path": str(CACHE_PATH),
            "size_bytes": 0,
            "combinations": 0,
            "emotions": 0
        }
    
    try:
        # Get file size
        file_size = CACHE_PATH.stat().st_size
        
        # Load and count contents
        cpt_data = load_cpt()
        if cpt_data:
            combinations_count = len(cpt_data.get('combinations', {}))
            emotions_count = len(cpt_data.get('emotions', []))
        else:
            combinations_count = 0
            emotions_count = 0
        
        return {
            "exists": True,
            "path": str(CACHE_PATH),
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "combinations": combinations_count,
            "emotions": emotions_count
        }
        
    except Exception as e:
        return {
            "exists": True,
            "path": str(CACHE_PATH),
            "error": str(e),
            "size_bytes": 0,
            "combinations": 0,
            "emotions": 0
        }
