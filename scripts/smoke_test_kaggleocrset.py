"""
Smoke test runner for randomly selected images from kaggleocrset.
Runs OCREngine.extract() and displays raw_ocr, fields, labs results.
"""

import sys
import json
import random
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.ocr.engine import OCREngine

def main():
    kaggle_path = project_root / "data" / "kaggleocrset"
    images = sorted(list(kaggle_path.glob("*.png")))
    
    if not images:
        print("No images found in data/kaggleocrset/")
        return
    
    # Select 5 random images
    random.seed(42)
    sample_images = random.sample(images, min(5, len(images)))
    
    print(f"Selected {len(sample_images)} random images from {len(images)} available\n")
    print("=" * 100)
    
    results = []
    
    for idx, img_path in enumerate(sample_images, 1):
        print(f"\n[{idx}/{len(sample_images)}] Testing: {img_path.name}")
        print("-" * 100)
        
        try:
            engine = OCREngine()
            result = engine.extract(str(img_path))
            
            # Extract key info
            raw_ocr_count = len(result.get("raw_ocr", []))
            fields = result.get("fields", {})
            sections = result.get("sections", {})
            labs = result.get("labs", [])
            warnings = result.get("warnings", [])
            
            print(f"Status: ✓ OK")
            print(f"  Raw OCR items: {raw_ocr_count}")
            print(f"  Fields found: {len(fields)}")
            print(f"  Sections found: {len(sections)}")
            print(f"  Lab values found: {len(labs)}")
            
            if fields:
                print(f"  Fields: {list(fields.keys())[:5]}{'...' if len(fields) > 5 else ''}")
            
            if labs:
                labs_keys = list(labs.keys())[:3] if isinstance(labs, dict) else labs[:3]
                print(f"  Labs sample: {labs_keys}")
            
            if warnings:
                print(f"  Warnings: {len(warnings)}")
                for w in warnings[:2]:
                    print(f"    - {w}")
            
            # Prepare labs sample
            if isinstance(labs, dict):
                labs_sample = list(labs.items())[:5]
            else:
                labs_sample = labs[:5] if labs else []
            
            results.append({
                "image": img_path.name,
                "status": "OK",
                "raw_ocr_count": raw_ocr_count,
                "fields_count": len(fields),
                "sections_count": len(sections),
                "labs_count": len(labs),
                "field_keys": list(fields.keys()),
                "labs_sample": labs_sample,
                "warnings_count": len(warnings)
            })
            
        except Exception as e:
            print(f"Status: ✗ FAILED")
            print(f"  Error: {type(e).__name__}: {str(e)[:100]}")
            results.append({
                "image": img_path.name,
                "status": "FAILED",
                "error": str(e)[:100]
            })
    
    print("\n" + "=" * 100)
    print("\nSMOKE TEST SUMMARY")
    print("-" * 100)
    passed = sum(1 for r in results if r.get("status") == "OK")
    failed = len(results) - passed
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} tests")
    
    # Save results to JSON
    results_file = project_root / "test_results_kaggleocrset.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
