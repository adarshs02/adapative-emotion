#!/usr/bin/env python3
"""
Convert JSONL (JSON Lines) file to proper JSON array format.
"""

import json
import sys
import os


def convert_jsonl_to_json(input_file, output_file=None):
    """Convert JSONL file to JSON array format."""
    
    if not os.path.exists(input_file):
        print(f"❌ Input file does not exist: {input_file}")
        return False
    
    # Default output file name
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.json"
    
    print(f"📝 Converting {input_file} to {output_file}")
    print("📊 Reading JSONL file...")
    
    data = []
    line_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                        line_count += 1
                        
                        # Progress indicator for large files
                        if line_count % 1000 == 0:
                            print(f"  Processed {line_count} lines...")
                            
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Invalid JSON on line {line_num}: {e}")
                        continue
        
        print(f"✅ Successfully read {line_count} JSON objects")
        
        # Write as JSON array
        print("💾 Writing JSON array...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully converted to {output_file}")
        
        # Show file sizes
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        print(f"📊 File sizes:")
        print(f"  Input:  {input_size:.1f} MB")
        print(f"  Output: {output_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False


def main():
    """Main function with command line arguments."""
    
    if len(sys.argv) < 2:
        print("Usage: python convert_jsonl_to_json.py <input_file> [output_file]")
        print("\nExample:")
        print("  python convert_jsonl_to_json.py train_dataset1.json")
        print("  python convert_jsonl_to_json.py train_dataset1.json train_dataset1_fixed.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_jsonl_to_json(input_file, output_file)
    
    if success:
        print("\n🎉 Conversion completed successfully!")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
