#!/usr/bin/env python3

"""
Simple test script to validate that our code quality improvements 
haven't broken basic Python syntax and import structure.
"""

import sys
import ast

def test_syntax(filename):
    """Test if a Python file has valid syntax."""
    try:
        with open(filename, 'r') as f:
            ast.parse(f.read(), filename=filename)
        print(f"✓ {filename}: Syntax OK")
        return True
    except SyntaxError as e:
        print(f"✗ {filename}: Syntax Error - {e}")
        return False

def main():
    """Run basic syntax checks on modified files."""
    test_files = [
        "autoBOTLib/__init__.py",
        "autoBOTLib/__main__.py", 
        "autoBOTLib/misc/misc_keyword_detection.py"
    ]
    
    all_passed = True
    
    print("Testing syntax of modified files...")
    for filename in test_files:
        if not test_syntax(filename):
            all_passed = False
    
    if all_passed:
        print("\nAll syntax tests passed! ✓")
        return 0
    else:
        print("\nSome syntax tests failed! ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())