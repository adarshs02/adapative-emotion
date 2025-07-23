#!/usr/bin/env python3
"""
Directory Space Usage Checker

A simple script to check the disk space usage of directories.
"""

import os
import subprocess
import sys
from pathlib import Path


def get_directory_size(directory_path):
    """Get the size of a directory using du command."""
    try:
        # Use du -sh for human-readable output
        result = subprocess.run(
            ['du', '-sh', directory_path], 
            capture_output=True, 
            text=True, 
            check=True
        )
        size = result.stdout.strip().split('\t')[0]
        return size
    except subprocess.CalledProcessError:
        return "Error"
    except Exception as e:
        return f"Error: {e}"


def get_detailed_directory_sizes(directory_path, max_depth=1):
    """Get detailed breakdown of directory sizes."""
    try:
        # Use du with max-depth for breakdown
        result = subprocess.run(
            ['du', '-h', f'--max-depth={max_depth}', directory_path], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        breakdown = []
        
        for line in lines:
            if line.strip():
                parts = line.split('\t')
                if len(parts) == 2:
                    size, path = parts
                    breakdown.append((size, path))
        
        return breakdown
    except subprocess.CalledProcessError:
        return []
    except Exception as e:
        return []


def check_space_usage(target_directory=None, detailed=False, max_depth=1):
    """Check space usage for a directory."""
    
    if target_directory is None:
        target_directory = os.getcwd()
    
    # Convert to absolute path
    target_path = os.path.abspath(target_directory)
    
    if not os.path.exists(target_path):
        print(f"❌ Directory does not exist: {target_path}")
        return
    
    if not os.path.isdir(target_path):
        print(f"❌ Path is not a directory: {target_path}")
        return
    
    print(f"📁 Checking space usage for: {target_path}")
    print("=" * 60)
    
    # Get total size
    total_size = get_directory_size(target_path)
    print(f"📊 Total Size: {total_size}")
    
    if detailed:
        print(f"\n📋 Breakdown (max depth {max_depth}):")
        print("-" * 40)
        
        breakdown = get_detailed_directory_sizes(target_path, max_depth)
        
        if breakdown:
            # Sort by directory name for consistent output
            breakdown.sort(key=lambda x: x[1])
            
            for size, path in breakdown:
                # Get relative path for cleaner display
                try:
                    rel_path = os.path.relpath(path, target_path)
                    if rel_path == '.':
                        rel_path = '(current directory)'
                except ValueError:
                    rel_path = path
                
                print(f"  {size:>8} | {rel_path}")
        else:
            print("  Could not get detailed breakdown")
    
    # Show available space on the filesystem
    try:
        statvfs = os.statvfs(target_path)
        available_bytes = statvfs.f_bavail * statvfs.f_frsize
        available_gb = available_bytes / (1024**3)
        print(f"\n💾 Available space on filesystem: {available_gb:.1f} GB")
    except Exception:
        pass


def main():
    """Main function with command line argument parsing."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check directory space usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_space.py                    # Check current directory
  python check_space.py /path/to/dir      # Check specific directory
  python check_space.py -d                # Detailed breakdown
  python check_space.py -d --depth 2      # Detailed with depth 2
  python check_space.py ~/projects -d     # Check ~/projects with details
        """
    )
    
    parser.add_argument(
        'directory', 
        nargs='?', 
        default=None,
        help='Directory to check (default: current directory)'
    )
    
    parser.add_argument(
        '-d', '--detailed', 
        action='store_true',
        help='Show detailed breakdown of subdirectories'
    )
    
    parser.add_argument(
        '--depth', 
        type=int, 
        default=1,
        help='Maximum depth for detailed breakdown (default: 1)'
    )
    
    args = parser.parse_args()
    
    # If no directory specified, use current directory
    target_dir = args.directory if args.directory else os.getcwd()
    
    try:
        check_space_usage(
            target_directory=target_dir,
            detailed=args.detailed,
            max_depth=args.depth
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
