# Cell
# ========================================
# Step 2: DDXPlus Data Preprocessing (HuggingFace Format from Drive)
# ========================================


# ========================================
# 1. SETUP & INSTALLATION
# ========================================

!pip install -q pandas tqdm datasets

import json
import pandas as pd
import ast
from pathlib import Path
from tqdm.notebook import tqdm
import os
from datasets import load_from_disk

print("✅ Libraries imported successfully!")

# ========================================
# 2. MOUNT GOOGLE DRIVE
# ========================================

from google.colab import drive
drive.mount('/content/drive')

print("\n✅ Google Drive mounted!")

# ========================================
# 3. SET YOUR PATHS
# ========================================

# 🔴 IMPORTANT: Update this to match YOUR Drive structure
# Your structure: ddx/raw/ddxplus_hf/
DRIVE_BASE = '/content/drive/MyDrive/DDX'  # ← Change this if different!

BASE_DIR = Path(DRIVE_BASE)
RAW_DIR = BASE_DIR / "raw" / "ddxplus_hf"
PROCESSED_DIR = BASE_DIR / "processed"

# Create processed directory
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Directory Setup:")
print(f"   Base: {BASE_DIR}")
print(f"   Raw data (HF format): {RAW_DIR}")
print(f"   Processed data: {PROCESSED_DIR}")

# ========================================
# 4. VERIFY DATASET STRUCTURE
# ========================================

print("\n🔍 Checking dataset structure...")

# Check if HuggingFace dataset exists
if not RAW_DIR.exists():
    print(f"❌ ERROR: {RAW_DIR} does not exist!")
    print(f"\n📋 Please check your folder structure:")
    print(f"   Expected: {DRIVE_BASE}/raw/ddxplus_hf/")
    print(f"   Should contain: train/, test/, validation/ folders")
    raise FileNotFoundError("Dataset not found!")

# List contents
print(f"\n📂 Contents of {RAW_DIR}:")
if RAW_DIR.exists():
    for item in RAW_DIR.iterdir():
        if item.is_dir():
            print(f"   📁 {item.name}/")
        else:
            print(f"   📄 {item.name}")

# Check for split folders
expected_splits = ['train', 'test', 'validate']
found_splits = []

for split in expected_splits:
    split_path = RAW_DIR / split
    if split_path.exists():
        print(f"   ✅ Found: {split}/")
        found_splits.append(split)
    else:
        print(f"   ❌ Missing: {split}/")

if not found_splits:
    print("\n❌ No split folders found!")
    print("Expected structure:")
    print("  ddx/raw/ddxplus_hf/train/")
    print("  ddx/raw/ddxplus_hf/test/")
    print("  ddx/raw/ddxplus_hf/validation/")
    raise FileNotFoundError("Dataset splits not found!")

print(f"\n✅ Found {len(found_splits)} splits: {found_splits}")

# ========================================
# 5. LOAD HUGGINGFACE DATASET
# ========================================

print("\n📥 Loading dataset from disk...")

try:
    # Load the entire dataset
    dataset = load_from_disk(str(RAW_DIR))

    print("✅ Dataset loaded successfully!")
    print(f"\n📊 Dataset Info:")
    print(f"   Available splits: {list(dataset.keys())}")

    for split_name in dataset.keys():
        print(f"   • {split_name}: {len(dataset[split_name]):,} samples")

    # Show sample
    if 'train' in dataset:
        print(f"\n📋 Sample record structure:")
        sample = dataset['train'][0]
        for key in sample.keys():
            value = str(sample[key])
            if len(value) > 50:
                value = value[:50] + "..."
            print(f"   • {key}: {value}")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("\nTrying alternative method...")

    # Alternative: Load each split separately
    dataset = {}
    for split in found_splits:
        split_path = RAW_DIR / split
        try:
            from datasets import Dataset
            # Try to load as arrow dataset
            split_dataset = Dataset.load_from_disk(str(split_path))
            dataset[split] = split_dataset
            print(f"✅ Loaded {split}: {len(split_dataset):,} samples")
        except Exception as e2:
            print(f"❌ Could not load {split}: {e2}")

# ========================================
# 6. EVIDENCE MAPPER
# ========================================

class EvidenceMapper:
    """Maps evidence codes to human-readable text"""

    def __init__(self):
        # We'll extract unique evidence keys from the dataset
        self.code_to_text = {}
        print("   ℹ️ Using dynamic evidence mapping from dataset")

    def get_text(self, code):
        """Get human-readable text for evidence code"""
        # Clean the code
        code_clean = str(code).replace('_', ' ').replace('-', ' ')

        # Remove common prefixes
        code_clean = code_clean.replace('E ', '').replace('e ', '')

        # Capitalize words
        code_clean = ' '.join(word.capitalize() for word in code_clean.split())

        return code_clean if code_clean else str(code)

# ========================================
# 7. MAIN PREPROCESSING PIPELINE
# ========================================

class DDXPreprocessor:
    """Main preprocessing class for HuggingFace format"""

    def __init__(self):
        self.evidence_mapper = EvidenceMapper()
        print("✅ Preprocessor initialized")

    def parse_evidences(self, evidences_data):
        """Parse evidences to extract symptoms - Fixed for DDXPlus format"""
        try:
            if evidences_data is None:
                return []

            # ── DDXPlus الصيغة الأساسية: list of strings زي ['fever', 'cough'] ──
            if isinstance(evidences_data, list):
                symptoms = []
                for item in evidences_data:
                    if isinstance(item, str) and item.strip():
                        symptom_text = self.evidence_mapper.get_text(item)
                        symptoms.append(symptom_text)
                    elif isinstance(item, dict):
                        # لو كل عنصر dict فيه key اسمه 'name' أو 'code'
                        for key in ['name', 'code', 'symptom']:
                            if key in item:
                                symptoms.append(self.evidence_mapper.get_text(item[key]))
                                break
                return symptoms

            # ── صيغة dict: {'fever': 1, 'cough': 'Y'} ──
            if isinstance(evidences_data, dict):
                symptoms = []
                for code, value in evidences_data.items():
                    if value in [1, 'Y', True, 'yes', '1', 1.0]:
                        symptoms.append(self.evidence_mapper.get_text(code))
                return symptoms

            # ── لو string: حاول تحوله ──
            if isinstance(evidences_data, str):
                try:
                    parsed = ast.literal_eval(evidences_data)
                    return self.parse_evidences(parsed)  # استدعاء نفسه بعد التحويل
                except Exception:
                    # لو فشل التحويل، رجّع الـ string كعرض واحد
                    return [evidences_data.strip()] if evidences_data.strip() else []

            return []

        except Exception as e:
            print(f"   ⚠️ parse_evidences error: {type(e).__name__}: {e}")  # ← logging بدل صمت
            return []

    def parse_differential(self, diff_data):
        """Parse differential diagnosis"""
        try:
            if diff_data is None or pd.isna(diff_data):
                return "Not available"

            # Handle different formats
            if isinstance(diff_data, str):
                try:
                    diff_list = ast.literal_eval(diff_data)
                except:
                    try:
                        diff_list = json.loads(diff_data)
                    except:
                        return "Not available"
            elif isinstance(diff_data, list):
                diff_list = diff_data
            else:
                return "Not available"

            # Format top 3
            diff_text = []
            for item in diff_list[:3]:
                if isinstance(item, dict):
                    # Format: {'disease': 'name', 'probability': 0.85}
                    disease = item.get('disease', item.get('condition', 'Unknown'))
                    prob = item.get('probability', item.get('prob', 0))
                    diff_text.append(f"{disease} ({prob*100:.0f}%)")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # Format: ['disease', 0.85]
                    disease, prob = item[0], item[1]
                    diff_text.append(f"{disease} ({prob*100:.0f}%)")

            return ", ".join(diff_text) if diff_text else "Not available"

        except:
            return "Not available"

    def create_combined_text(self, row):
        """Create rich text for embedding"""
        text_parts = []

        # Patient demographics
        if 'age' in row and 'sex' in row:
            text_parts.append(f"Patient: {row['age']} year old {row['sex']}")

        # Symptoms
        if row.get('symptoms_text') and row['symptoms_text'] != "None reported":
            text_parts.append(f"Presenting symptoms: {row['symptoms_text']}")

        # Diagnosis
        if 'pathology' in row:
            text_parts.append(f"Diagnosed condition: {row['pathology']}")

        # Differential diagnosis
        if row.get('differential_diagnosis') and row['differential_diagnosis'] != "Not available":
            text_parts.append(f"Differential diagnosis: {row['differential_diagnosis']}")

        return ". ".join(text_parts) if text_parts else "No information available"

    def process_hf_dataset(self, hf_dataset, split_name):
        """Process HuggingFace dataset format"""
        print(f"\n🔄 Processing {split_name} set...")
        print(f"   Input: {len(hf_dataset):,} samples")

        processed_data = []
        errors = 0

        # Convert to pandas for easier processing
        df = pd.DataFrame(hf_dataset[:])  # Load all data

        # Process with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            try:
                # Parse symptoms
                symptoms_list = []

                # Try different column names for evidences
                for col_name in ['EVIDENCES', 'evidences', 'symptoms', 'evidence']:
                    if col_name in row and row[col_name] is not None:
                        symptoms_list = self.parse_evidences(row[col_name])
                        if symptoms_list:
                            break

                symptoms_text = ", ".join(symptoms_list) if symptoms_list else "None reported"

                # Get pathology (try different column names)
                pathology = "Unknown"
                for col_name in ['PATHOLOGY', 'pathology', 'diagnosis', 'condition']:
                    if col_name in row and row[col_name] is not None:
                        pathology = str(row[col_name])
                        break

                # Get age and sex
                age = 0
                for col_name in ['AGE', 'age']:
                    if col_name in row and not pd.isna(row[col_name]):
                        age = int(row[col_name])
                        break

                sex = "Unknown"
                for col_name in ['SEX', 'sex', 'gender']:
                    if col_name in row and row[col_name] is not None:
                        sex = str(row[col_name])
                        break

                # Parse differential
                differential = "Not available"
                for col_name in ['DIFFERENTIAL_DIAGNOSIS', 'differential_diagnosis', 'differential']:
                    if col_name in row and row[col_name] is not None:
                        differential = self.parse_differential(row[col_name])
                        if differential != "Not available":
                            break

                # Create processed record
                processed_record = {
                    'patient_id': f"{split_name}_{idx}",
                    'age': age,
                    'sex': sex,
                    'symptoms_text': symptoms_text,
                    'symptom_count': len(symptoms_list),
                    'pathology': pathology,
                    'differential_diagnosis': differential,
                }

                # Create combined text
                processed_record['combined_text'] = self.create_combined_text(processed_record)

                processed_data.append(processed_record)

            except Exception as e:
                errors += 1
                continue

        processed_df = pd.DataFrame(processed_data)

        print(f"   ✅ Processed: {len(processed_df):,} samples")
        if errors > 0:
            print(f"   ⚠️ Skipped: {errors} records due to errors")

        return processed_df

    def process_all_splits(self, dataset, sample_size=None):
        """Process all dataset splits"""
        print("\n" + "="*60)
        print("🏥 DDXPlus Data Preprocessing Pipeline")
        print("="*60)

        processed_splits = {}

        for split_name in dataset.keys():
            try:
                hf_dataset = dataset[split_name]

                # Sample if requested
                if sample_size and len(hf_dataset) > sample_size:
                    print(f"\n📊 Sampling {sample_size:,} records from {split_name}...")
                    # Random sample
                    import random
                    indices = random.sample(range(len(hf_dataset)), sample_size)
                    hf_dataset = hf_dataset.select(indices)

                # Process
                processed_df = self.process_hf_dataset(hf_dataset, split_name)

                # Save
                output_path = PROCESSED_DIR / f"{split_name}_processed.csv"
                processed_df.to_csv(output_path, index=False)
                file_size = output_path.stat().st_size / (1024*1024)
                print(f"   💾 Saved: {output_path.name} ({file_size:.1f} MB)")

                processed_splits[split_name] = processed_df

            except Exception as e:
                print(f"   ❌ Error processing {split_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return processed_splits

    def show_statistics(self, processed_splits):
        """Display dataset statistics"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)

        for split_name, df in processed_splits.items():
            print(f"\n{split_name.upper()} SET:")
            print(f"  Total samples: {len(df):,}")
            print(f"  Unique diseases: {df['pathology'].nunique()}")
            print(f"  Avg symptoms: {df['symptom_count'].mean():.1f} per patient")

            if 'age' in df.columns:
                age_stats = df['age'][df['age'] > 0]
                if len(age_stats) > 0:
                    print(f"  Age range: {age_stats.min()}-{age_stats.max()} years")

            if 'sex' in df.columns:
                sex_dist = df['sex'].value_counts()
                print(f"  Sex distribution:")
                for sex, count in sex_dist.head(3).items():
                    print(f"    • {sex}: {count:,} ({count/len(df)*100:.1f}%)")

            print(f"\n  Top 5 diseases:")
            top_diseases = df['pathology'].value_counts().head(5)
            for disease, count in top_diseases.items():
                disease_short = disease[:40] + "..." if len(disease) > 40 else disease
                print(f"    • {disease_short}: {count:,} ({count/len(df)*100:.1f}%)")

        # Show sample
        if processed_splits:
            print("\n" + "="*60)
            print("📋 SAMPLE PROCESSED RECORDS")
            print("="*60)
            first_split = list(processed_splits.values())[0]

            for i in range(min(2, len(first_split))):
                print(f"\n--- Sample {i+1} ---")
                sample = first_split.iloc[i]
                for col in ['patient_id', 'age', 'sex', 'symptoms_text', 'pathology']:
                    if col in sample.index:
                        value = str(sample[col])
                        if len(value) > 80:
                            value = value[:80] + "..."
                        print(f"  {col}: {value}")

# ========================================
# 8. MAIN EXECUTION
# ========================================

print("\n" + "="*60)
print("🚀 STARTING PREPROCESSING")
print("="*60)

# Processing mode
print("\n⚙️ Processing mode:")
print("  Quick test: 1,000 samples per split")
print("  Full dataset: All samples")

USE_SAMPLE_SIZE = 1000  # Change to None for full dataset

if USE_SAMPLE_SIZE:
    print(f"\n📊 Running with {USE_SAMPLE_SIZE:,} samples per split...")
else:
    print(f"\n📊 Running with FULL dataset...")

# Initialize preprocessor
preprocessor = DDXPreprocessor()

# Process dataset
processed_splits = preprocessor.process_all_splits(dataset, sample_size=USE_SAMPLE_SIZE)

# Show statistics
if processed_splits:
    preprocessor.show_statistics(processed_splits)

    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"📁 Processed data saved in: {PROCESSED_DIR}")
    print("\n📌 Files created:")
    for split in processed_splits.keys():
        file_path = PROCESSED_DIR / f"{split}_processed.csv"
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024*1024)
            records = len(processed_splits[split])
            print(f"  ✅ {split}_processed.csv - {records:,} records ({file_size:.1f} MB)")

    print("\n✨ Ready for Step 3: ClinicalBERT Embeddings!")
else:
    print("\n❌ No data was processed. Please check errors above.")

# Cell
# ========================================
# Step 2: Generate ClinicalBERT Embeddings
# ========================================

"""
✅ Load ClinicalBERT model
✅ Generate embeddings for all processed data
✅ Save embeddings efficiently
✅ Progress tracking with tqdm

What are embeddings?
- Convert text → 768-dimensional vector
- Similar texts → similar vectors
- Used for semantic search
"""

# ========================================
# 1. INSTALL & IMPORT
# ========================================

!pip install -q transformers torch pandas numpy tqdm

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModel
import pickle
import json

print("✅ Libraries imported!")

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ========================================
# 2. SETUP PATHS
# ========================================

from google.colab import drive
try:
    drive.mount('/content/drive')
except:
    print("Drive already mounted")

DRIVE_BASE = '/content/drive/MyDrive/DDX'
BASE_DIR = Path(DRIVE_BASE)
PROCESSED_DIR = BASE_DIR / "processed"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Directories:")
print(f"   Processed data: {PROCESSED_DIR}")
print(f"   Embeddings output: {EMBEDDINGS_DIR}")

# ========================================
# 3. VERIFY PROCESSED DATA EXISTS
# ========================================

print("\n🔍 Checking for processed data...")

processed_files = list(PROCESSED_DIR.glob("*_processed.csv"))

if not processed_files:
    print("❌ No processed files found!")
    print(f"   Expected location: {PROCESSED_DIR}")
    print("\n⚠️ Please run Step 1 (preprocessing) first!")
    raise FileNotFoundError("Processed data not found")

print(f"✅ Found {len(processed_files)} processed files:")
for file in processed_files:
    file_size = file.stat().st_size / (1024*1024)
    print(f"   • {file.name} ({file_size:.1f} MB)")

# ========================================
# 4. LOAD CLINICALBERT MODEL
# ========================================

print("\n" + "="*60)
print("🧠 LOADING CLINICALBERT MODEL")
print("="*60)

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
print(f"\nModel: {MODEL_NAME}")
print("This model is specifically trained on clinical text!")
print("\n⏳ Downloading model... (first time only, ~400MB)")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    print("✅ ClinicalBERT loaded successfully!")
    print(f"   Embedding dimension: 768")
    print(f"   Max sequence length: {tokenizer.model_max_length}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# ========================================
# 5. EMBEDDING GENERATOR CLASS
# ========================================

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling على كل الـ tokens — أدق من CLS token
    لأنها بتاخد متوسط كل الكلمات مش أول token بس
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ClinicalBERTEmbedder:
    """
    Generate embeddings using ClinicalBERT with Mean Pooling
    """

    def __init__(self, model, tokenizer, device, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

        print(f"⚙️ Embedder initialized")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device}")
        print(f"   Pooling: Mean Pooling ✅")

    def encode_text(self, text):
        """
        Encode single text to embedding using mean pooling
        """
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # ✅ Mean pooling بدل CLS token
        embedding = mean_pooling(outputs, inputs['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0]

    def encode_batch(self, texts):
        """
        Encode batch of texts using mean pooling (more efficient)
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # ✅ Mean pooling بدل CLS token
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def encode_dataframe(self, df, text_column='combined_text'):
        """
        Encode entire dataframe with progress bar
        """
        print(f"\n🔄 Encoding {len(df):,} texts...")
        print(f"   Text column: {text_column}")

        embeddings = []
        num_batches = (len(df) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(0, len(df), self.batch_size),
                     total=num_batches,
                     desc="Generating embeddings"):

            batch_texts = df[text_column].iloc[i:i+self.batch_size].tolist()
            batch_texts = [str(text) if pd.notna(text) else "" for text in batch_texts]

            batch_embeddings = self.encode_batch(batch_texts)
            embeddings.extend(batch_embeddings)

        embeddings_array = np.array(embeddings)

        print(f"✅ Generated embeddings shape: {embeddings_array.shape}")
        print(f"   ({len(df)} texts × 768 dimensions)")

        return embeddings_array

# ========================================
# 6. INITIALIZE EMBEDDER
# ========================================

BATCH_SIZE = 32 if device.type == 'cuda' else 8

embedder = ClinicalBERTEmbedder(
    model=model,
    tokenizer=tokenizer,
    device=device,
    batch_size=BATCH_SIZE
)

# ========================================
# 7. TEST EMBEDDING GENERATION
# ========================================

print("\n" + "="*60)
print("🧪 TESTING EMBEDDINGS")
print("="*60)

test_texts = [
    "Patient: 45 year old male. Presenting symptoms: fever, cough, fatigue. Diagnosed condition: Influenza",
    "Patient: 30 year old female. Presenting symptoms: chest pain, shortness of breath. Diagnosed condition: Pneumonia",
]

print("\n📝 Test texts:")
for i, text in enumerate(test_texts, 1):
    print(f"   {i}. {text[:80]}...")

print("\n⏳ Generating test embeddings...")
test_embeddings = embedder.encode_batch(test_texts)

print(f"✅ Test successful!")
print(f"   Shape: {test_embeddings.shape}")

from numpy.linalg import norm
similarity = np.dot(test_embeddings[0], test_embeddings[1]) / (
    norm(test_embeddings[0]) * norm(test_embeddings[1])
)
print(f"   Similarity between texts: {similarity:.3f}")
print(f"   (0 = completely different, 1 = identical)")

# ========================================
# 8. PROCESS ALL SPLITS
# ========================================

print("\n" + "="*60)
print("🚀 PROCESSING ALL DATA SPLITS")
print("="*60)

all_embeddings = {}
all_metadata = {}

for processed_file in processed_files:
    split_name = processed_file.stem.replace('_processed', '')

    print(f"\n{'='*60}")
    print(f"📊 Processing: {split_name.upper()}")
    print(f"{'='*60}")

    try:
        print(f"\n📂 Loading {processed_file.name}...")
        df = pd.read_csv(processed_file)
        print(f"   Loaded: {len(df):,} records")

        if 'combined_text' not in df.columns:
            print(f"❌ Error: 'combined_text' column not found!")
            print(f"   Available columns: {list(df.columns)}")
            continue

        embeddings = embedder.encode_dataframe(df, text_column='combined_text')

        embeddings_file = EMBEDDINGS_DIR / f"{split_name}_embeddings.npy"
        np.save(embeddings_file, embeddings)

        file_size = embeddings_file.stat().st_size / (1024*1024)
        print(f"   💾 Saved embeddings: {embeddings_file.name} ({file_size:.1f} MB)")

        metadata = {
            'patient_ids': df['patient_id'].tolist(),
            'pathologies': df['pathology'].tolist(),
            'symptoms': df['symptoms_text'].tolist(),
            'num_samples': len(df),
            'embedding_dim': 768,
            'pooling_method': 'mean_pooling',  # ✅ توثيق الطريقة
        }

        metadata_file = EMBEDDINGS_DIR / f"{split_name}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        metadata_size = metadata_file.stat().st_size / (1024*1024)
        print(f"   💾 Saved metadata: {metadata_file.name} ({metadata_size:.1f} MB)")

        all_embeddings[split_name] = embeddings
        all_metadata[split_name] = metadata

        print(f"   ✅ {split_name} complete!")

    except Exception as e:
        print(f"   ❌ Error processing {split_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ========================================
# 9. GENERATE SUMMARY
# ========================================

print("\n" + "="*60)
print("📊 EMBEDDINGS GENERATION SUMMARY")
print("="*60)

total_samples = 0
total_size = 0

for split_name in all_embeddings.keys():
    embeddings_file = EMBEDDINGS_DIR / f"{split_name}_embeddings.npy"
    metadata_file = EMBEDDINGS_DIR / f"{split_name}_metadata.pkl"

    num_samples = all_embeddings[split_name].shape[0]
    file_size = embeddings_file.stat().st_size / (1024*1024)

    print(f"\n{split_name.upper()}:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Embedding shape: {all_embeddings[split_name].shape}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Files:")
    print(f"    • {embeddings_file.name}")
    print(f"    • {metadata_file.name}")

    total_samples += num_samples
    total_size += file_size

print(f"\n{'='*60}")
print(f"TOTAL:")
print(f"  Total samples: {total_samples:,}")
print(f"  Total size: {total_size:.1f} MB")
print(f"  Location: {EMBEDDINGS_DIR}")

# ========================================
# 10. VERIFICATION TEST
# ========================================

print("\n" + "="*60)
print("🔍 VERIFICATION TEST")
print("="*60)

if all_embeddings:
    test_split = list(all_embeddings.keys())[0]

    print(f"\n✅ Testing load for: {test_split}")

    loaded_embeddings = np.load(EMBEDDINGS_DIR / f"{test_split}_embeddings.npy")
    with open(EMBEDDINGS_DIR / f"{test_split}_metadata.pkl", 'rb') as f:
        loaded_metadata = pickle.load(f)

    print(f"   Embeddings shape: {loaded_embeddings.shape}")
    print(f"   Metadata samples: {loaded_metadata['num_samples']}")
    print(f"   Pooling method: {loaded_metadata.get('pooling_method', 'N/A')}")
    print(f"   First patient ID: {loaded_metadata['patient_ids'][0]}")
    print(f"   First pathology: {loaded_metadata['pathologies'][0]}")

    assert loaded_embeddings.shape[0] == loaded_metadata['num_samples']
    assert loaded_embeddings.shape[1] == 768

    print("\n✅ Verification passed! Files are valid.")

# ========================================
# 11. SAVE GENERATION INFO
# ========================================

generation_info = {
    'model_name': MODEL_NAME,
    'device': str(device),
    'batch_size': BATCH_SIZE,
    'embedding_dim': 768,
    'pooling_method': 'mean_pooling',  # ✅ توثيق
    'total_samples': total_samples,
    'splits': list(all_embeddings.keys()),
    'generation_date': pd.Timestamp.now().isoformat(),
}

info_file = EMBEDDINGS_DIR / "generation_info.json"
with open(info_file, 'w') as f:
    json.dump(generation_info, f, indent=2)

print(f"\n💾 Generation info saved: {info_file.name}")

# ========================================
# COMPLETION
# ========================================

print("\n" + "="*60)
print("✅ STEP 2 COMPLETE!")
print("="*60)
print("\n📌 What we created:")
print("   • ClinicalBERT embeddings (768-dim vectors)")
print("   • Pooling method: Mean Pooling ✅")
print("   • Metadata files (patient IDs, diagnoses)")
print("   • Generation info (model details)")
print(f"\n📁 All files saved in: {EMBEDDINGS_DIR}")
print("\n🎯 NEXT STEP: Build FAISS vector database for fast search!")

# Cell
# ========================================
# Step 3: Build FAISS Vector Database
# ========================================

"""
✅ Load embeddings from Step 2
✅ Build FAISS index for fast similarity search
✅ Test retrieval with sample queries
✅ Save index for later use

FAISS = Facebook AI Similarity Search
- Ultra-fast vector search
- Millions of vectors in milliseconds
- Perfect for RAG systems
"""

# ========================================
# 1. INSTALL & IMPORT
# ========================================
# ✅ Load model if not already loaded from Cell 3
try:
    model
    tokenizer
    device
    print("✅ Model already loaded from Cell 3")
except NameError:
    print("⚠️ Loading ClinicalBERT (Cell 3 was not run)...")
    from transformers import AutoTokenizer, AutoModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    model.eval()
    print("✅ ClinicalBERT loaded!")

!pip install -q faiss-cpu numpy pandas

import faiss
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm.notebook import tqdm
import json

print("✅ Libraries imported!")

# ========================================
# 2. SETUP PATHS
# ========================================

# Mount Drive (if not already mounted)
from google.colab import drive
try:
    drive.mount('/content/drive')
except:
    print("Drive already mounted")

# Set paths
DRIVE_BASE = '/content/drive/MyDrive/DDX'
BASE_DIR = Path(DRIVE_BASE)
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
FAISS_DIR = BASE_DIR / "faiss_index"
PROCESSED_DIR = BASE_DIR / "processed"

# Create FAISS directory
FAISS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Directories:")
print(f"   Embeddings: {EMBEDDINGS_DIR}")
print(f"   FAISS Index: {FAISS_DIR}")
print(f"   Processed data: {PROCESSED_DIR}")

# ========================================
# 3. VERIFY EMBEDDINGS EXIST
# ========================================

print("\n🔍 Checking for embeddings...")

embedding_files = list(EMBEDDINGS_DIR.glob("*_embeddings.npy"))

if not embedding_files:
    print("❌ No embedding files found!")
    print(f"   Expected location: {EMBEDDINGS_DIR}")
    print("\n⚠️ Please run Step 2 (Generate Embeddings) first!")
    raise FileNotFoundError("Embeddings not found")

print(f"✅ Found {len(embedding_files)} embedding files:")
for file in embedding_files:
    file_size = file.stat().st_size / (1024*1024)
    print(f"   • {file.name} ({file_size:.1f} MB)")

# ========================================
# 4. LOAD ALL EMBEDDINGS & METADATA
# ========================================

print("\n" + "="*60)
print("📥 LOADING EMBEDDINGS & METADATA")
print("="*60)

all_embeddings = []
all_metadata = []
split_info = {}

for emb_file in embedding_files:
    split_name = emb_file.stem.replace('_embeddings', '')

    print(f"\n📂 Loading: {split_name}")

    # Load embeddings
    embeddings = np.load(emb_file)
    print(f"   Embeddings: {embeddings.shape}")

    # Load metadata
    metadata_file = EMBEDDINGS_DIR / f"{split_name}_metadata.pkl"
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    print(f"   Metadata: {metadata['num_samples']} samples")

    # Store
    all_embeddings.append(embeddings)
    all_metadata.append(metadata)

    split_info[split_name] = {
        'start_idx': len(all_embeddings) - 1,
        'num_samples': len(embeddings),
        'split': split_name
    }

# Combine all embeddings
print(f"\n🔗 Combining embeddings...")
combined_embeddings = np.vstack(all_embeddings)
print(f"   Combined shape: {combined_embeddings.shape}")
print(f"   Total vectors: {combined_embeddings.shape[0]:,}")
print(f"   Dimension: {combined_embeddings.shape[1]}")

# Combine metadata
print(f"\n🔗 Combining metadata...")
combined_patient_ids = []
combined_pathologies = []
combined_symptoms = []
combined_splits = []

for i, metadata in enumerate(all_metadata):
    split_name = list(split_info.keys())[i]
    combined_patient_ids.extend(metadata['patient_ids'])
    combined_pathologies.extend(metadata['pathologies'])
    combined_symptoms.extend(metadata['symptoms'])
    combined_splits.extend([split_name] * metadata['num_samples'])

print(f"   Total records: {len(combined_patient_ids):,}")

# ========================================
# 5. BUILD FAISS INDEX
# ========================================

print("\n" + "="*60)
print("🔨 BUILDING FAISS INDEX")
print("="*60)

# Get embedding dimension
dimension = combined_embeddings.shape[1]
print(f"\n📐 Vector dimension: {dimension}")

# Normalize embeddings for cosine similarity
print(f"\n🔄 Normalizing vectors for cosine similarity...")
faiss.normalize_L2(combined_embeddings)
print(f"   ✅ Vectors normalized")

# Build FAISS index
print(f"\n🏗️ Building FAISS index...")
print(f"   Index type: IndexFlatIP (Inner Product = Cosine Similarity)")

# Create IVF index — أسرع مع بيانات كبيرة
nlist = 100
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# Train أولاً (مطلوب مرة واحدة بس)
print(f"   Training index on {len(combined_embeddings):,} vectors...")
index.train(combined_embeddings)

# Add vectors
print(f"   Adding {len(combined_embeddings):,} vectors...")
index.add(combined_embeddings)

print(f"\n✅ FAISS index built successfully!")
print(f"   Total vectors in index: {index.ntotal:,}")
print(f"   Index is trained: {index.is_trained}")

# ========================================
# 6. SAVE FAISS INDEX
# ========================================

print("\n" + "="*60)
print("💾 SAVING FAISS INDEX")
print("="*60)

# Save FAISS index
index_file = FAISS_DIR / "medical_cases.index"
faiss.write_index(index, str(index_file))
index_size = index_file.stat().st_size / (1024*1024)
print(f"\n✅ FAISS index saved: {index_file.name} ({index_size:.1f} MB)")

# Save metadata mapping
metadata_mapping = {
    'patient_ids': combined_patient_ids,
    'pathologies': combined_pathologies,
    'symptoms': combined_symptoms,
    'splits': combined_splits,
    'num_vectors': len(combined_embeddings),
    'dimension': dimension,
    'split_info': split_info,
}

mapping_file = FAISS_DIR / "metadata_mapping.pkl"
with open(mapping_file, 'wb') as f:
    pickle.dump(metadata_mapping, f)
mapping_size = mapping_file.stat().st_size / (1024*1024)
print(f"✅ Metadata mapping saved: {mapping_file.name} ({mapping_size:.1f} MB)")

# Save index info
index_info = {
    'index_type': 'IndexFlatIP',
    'dimension': dimension,
    'num_vectors': int(index.ntotal),
    'similarity_metric': 'cosine',
    'splits': list(split_info.keys()),
    'created_date': pd.Timestamp.now().isoformat(),
}

info_file = FAISS_DIR / "index_info.json"
with open(info_file, 'w') as f:
    json.dump(index_info, f, indent=2)
print(f"✅ Index info saved: {info_file.name}")

# ========================================
# 7. CREATE SEARCH FUNCTION
# ========================================

class MedicalCaseSearcher:
    """
    Search similar medical cases using FAISS
    """

    def __init__(self, index, metadata_mapping):
        self.index = index
        self.metadata = metadata_mapping
        print("✅ Medical Case Searcher initialized")
        print(f"   Index size: {self.index.ntotal:,} cases")

    def search(self, query_embedding, k=5):
        """
        Search for k most similar cases

        Args:
            query_embedding: 768-dim vector (from ClinicalBERT)
            k: number of results to return

        Returns:
            List of similar cases with scores
        """
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Get results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            result = {
                'rank': i + 1,
                'similarity_score': float(score),
                'patient_id': self.metadata['patient_ids'][idx],
                'pathology': self.metadata['pathologies'][idx],
                'symptoms': self.metadata['symptoms'][idx],
                'split': self.metadata['splits'][idx],
                'index': int(idx),
            }
            results.append(result)

        return results

    def print_results(self, results):
        """Pretty print search results"""
        print(f"\n{'='*60}")
        print(f"🔍 SEARCH RESULTS (Top {len(results)})")
        print(f"{'='*60}")

        for result in results:
            print(f"\n#{result['rank']} - Similarity: {result['similarity_score']:.3f} ({result['similarity_score']*100:.1f}%)")
            print(f"   Diagnosis: {result['pathology']}")
            print(f"   Symptoms: {result['symptoms'][:100]}...")
            print(f"   Patient: {result['patient_id']} ({result['split']})")

# Initialize searcher
searcher = MedicalCaseSearcher(index, metadata_mapping)

# ========================================
# 8. TEST SEARCH WITH EXAMPLES
# ========================================

print("\n" + "="*60)
print("🧪 TESTING SEARCH FUNCTIONALITY")
print("="*60)

# Test 1: Search by index (direct embedding lookup)
print("\n📊 TEST 1: Find similar cases to a specific patient")
print("-" * 60)

test_idx = 42  # Random patient
test_embedding = combined_embeddings[test_idx]

print(f"\nQuery patient:")
print(f"  ID: {combined_patient_ids[test_idx]}")
print(f"  Diagnosis: {combined_pathologies[test_idx]}")
print(f"  Symptoms: {combined_symptoms[test_idx][:100]}...")

results = searcher.search(test_embedding, k=5)
searcher.print_results(results)

# Test 2: Search by pathology
print("\n\n📊 TEST 2: Find patients with similar pathology")
print("-" * 60)

target_pathology = combined_pathologies[100]
print(f"\nSearching for cases similar to: {target_pathology}")

test_embedding_2 = combined_embeddings[100]
results_2 = searcher.search(test_embedding_2, k=5)
searcher.print_results(results_2)

# ========================================
# 9. STATISTICS & ANALYSIS
# ========================================

print("\n" + "="*60)
print("📊 INDEX STATISTICS")
print("="*60)

print(f"\n🔢 Overall:")
print(f"   Total cases: {index.ntotal:,}")
print(f"   Vector dimension: {dimension}")
print(f"   Index size: {index_size:.1f} MB")
print(f"   Metadata size: {mapping_size:.1f} MB")

print(f"\n📁 By Split:")
for split_name, info in split_info.items():
    print(f"   {split_name}: {info['num_samples']:,} cases")

print(f"\n🏥 Disease Distribution (Top 10):")
pathology_counts = pd.Series(combined_pathologies).value_counts()
for disease, count in pathology_counts.head(10).items():
    percentage = count / len(combined_pathologies) * 100
    disease_short = disease[:40] + "..." if len(disease) > 40 else disease
    print(f"   • {disease_short}: {count:,} ({percentage:.1f}%)")

# ========================================
# 10. CREATE SIMPLE QUERY INTERFACE
# ========================================

print("\n" + "="*60)
print("🎯 INTERACTIVE SEARCH INTERFACE")
print("="*60)

def search_by_symptoms(symptoms_query, top_k=5):
    """
    Semantic search باستخدام ClinicalBERT — مش keyword matching
    """
    print(f"\n🔍 Searching for: '{symptoms_query}'")
    print(f"   (Semantic search via ClinicalBERT + FAISS)")

    # ── Encode النص بـ ClinicalBERT ──
    inputs = tokenizer(
        symptoms_query,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    token_embeddings = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    query_embedding = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    query_np = query_embedding.cpu().numpy().astype('float32')

    # ── FAISS search ──
    faiss.normalize_L2(query_np)
    scores, indices = index.search(query_np, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            'rank': len(results) + 1,
            'similarity_score': float(score),
            'pathology': combined_pathologies[idx],
            'symptoms': combined_symptoms[idx],
            'patient_id': combined_patient_ids[idx],
        })

    # Print results
    print(f"\n{'='*60}")
    print(f"🔍 SEARCH RESULTS (Top {len(results)})")
    print(f"{'='*60}")
    for r in results:
        print(f"\n#{r['rank']} - Similarity: {r['similarity_score']*100:.1f}%")
        print(f"   Diagnosis: {r['pathology']}")
        print(f"   Symptoms:  {r['symptoms'][:100]}...")

    return results
# Test interactive search
print("\n📝 Example queries:")
example_queries = [
    "fever cough",
    "chest pain",
    "headache nausea",
]

for query in example_queries:
    print(f"\n{'='*60}")
    search_by_symptoms(query, top_k=3)

# ========================================
# 11. SAVE SEARCHER FUNCTION
# ========================================

print("\n" + "="*60)
print("💾 SAVING SEARCH UTILITIES")
print("="*60)

# Save searcher code for later use
searcher_code = '''# Quick load and search utility
import faiss
import numpy as np
import pickle

# Load index
index = faiss.read_index('faiss_index/medical_cases.index')

# Load metadata
with open('faiss_index/metadata_mapping.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Loaded {index.ntotal:,} medical cases")

# Search function
def search_cases(query_embedding, k=5):
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            'score': float(score),
            'pathology': metadata['pathologies'][idx],
            'symptoms': metadata['symptoms'][idx],
        })
    return results
'''

utils_file = FAISS_DIR / "search_utils.py"
with open(utils_file, 'w') as f:
    f.write(searcher_code)

print(f"✅ Search utilities saved: {utils_file.name}")


# ========================================
# EVALUATION METRICS
# ========================================

print("\n" + "="*60)
print("📊 EVALUATION METRICS")
print("="*60)

def evaluate_retrieval(searcher, test_embeddings, test_pathologies):
    top1_correct = 0
    top5_correct = 0
    n = len(test_embeddings)

    print(f"\n🔄 Evaluating on {n:,} test cases...")

    for embedding, true_label in zip(test_embeddings, test_pathologies):
        results = searcher.search(embedding, k=5)
        predicted_labels = [r['pathology'] for r in results]

        if predicted_labels[0] == true_label:
            top1_correct += 1
        if true_label in predicted_labels:
            top5_correct += 1

    print(f"\n✅ Results:")
    print(f"   Top-1 Accuracy: {top1_correct/n:.2%}")
    print(f"   Top-5 Accuracy: {top5_correct/n:.2%}")
    print(f"   Total evaluated: {n:,} cases")

    return {
        'top1_accuracy': top1_correct/n,
        'top5_accuracy': top5_correct/n,
        'total_cases': n
    }

# شغّل الـ evaluation على الـ test split
test_split_name = 'test'  # أو 'validate'

if test_split_name in split_info:
    # جيب الـ embeddings الخاصة بالـ test split بس
    test_start = 0
    for name in list(split_info.keys()):
        if name == test_split_name:
            break
        test_start += split_info[name]['num_samples']

    test_end = test_start + split_info[test_split_name]['num_samples']
    test_embs = combined_embeddings[test_start:test_end]
    test_paths = combined_pathologies[test_start:test_end]

    eval_results = evaluate_retrieval(searcher, test_embs, test_paths)

    # Save evaluation results
    eval_file = FAISS_DIR / "evaluation_results.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n💾 Evaluation results saved: {eval_file.name}")
else:
    print(f"⚠️ '{test_split_name}' split not found, skipping evaluation")
# ========================================
# COMPLETION
# ========================================

print("\n" + "="*60)
print("✅ STEP 3 COMPLETE!")
print("="*60)

print("\n📌 What we created:")
print("   • FAISS vector index (fast similarity search)")
print("   • Metadata mapping (patient info)")
print("   • Search utilities (ready to use)")

print(f"\n📁 Files saved in: {FAISS_DIR}")
print("   • medical_cases.index")
print("   • metadata_mapping.pkl")
print("   • index_info.json")
print("   • search_utils.py")

print(f"\n📊 Index Statistics:")
print(f"   Total cases: {index.ntotal:,}")
print(f"   Search speed: Milliseconds for millions of vectors")
print(f"   Ready for: RAG pipeline integration")

print("\n🎯 NEXT STEP: LLM Integration (RAG)")
print("   • Connect to Claude/GPT API")
print("   • Build prompt with retrieved cases")
print("   • Generate medical responses")

print("\n💡 You can now:")
print("   1. Search similar medical cases in milliseconds")
print("   2. Retrieve relevant context for any symptoms")
print("   3. Build RAG-powered medical assistant")

# Cell
# ========================================
# Step 4: LLM Integration - RAG with Google Gemini (FREE)
# ========================================

"""
✅ Load FAISS index & embeddings
✅ Connect to Google Gemini (FREE API)
✅ Build RAG pipeline (Retrieve + Generate)
✅ Create medical assistant chatbot
✅ Add safety disclaimers
✅ Arabic language support

RAG Flow:
User Query → [Arabic? → Gemini Translate] → ClinicalBERT → FAISS Search → Top Cases → Gemini → Response
"""

# ========================================
# 1. INSTALL & IMPORT
# ========================================

!pip install -q google-generativeai transformers torch faiss-cpu numpy pandas

import google.generativeai as genai
import faiss
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import json

print("✅ Libraries imported!")

# ========================================
# 2. SETUP PATHS
# ========================================

from google.colab import drive
try:
    drive.mount('/content/drive')
except:
    print("Drive already mounted")

DRIVE_BASE = '/content/drive/MyDrive/DDX'
BASE_DIR = Path(DRIVE_BASE)
FAISS_DIR = BASE_DIR / "faiss_index"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

print(f"\n📁 Directories:")
print(f"   FAISS: {FAISS_DIR}")
print(f"   Embeddings: {EMBEDDINGS_DIR}")

# ========================================
# 3. SETUP GOOGLE GEMINI API (FREE)
# ========================================

print("\n" + "="*60)
print("🔑 SETUP GOOGLE GEMINI API (FREE)")
print("="*60)

print("\n📝 Get your FREE API key:")
print("   1. Go to: https://makersuite.google.com/app/apikey")
print("   2. Click 'Create API Key'")
print("   3. Copy the key")
print("\n   (It's FREE - no credit card needed!)")

import getpass
GOOGLE_API_KEY = getpass.getpass("\n🔑 Paste your Gemini API Key: ")

genai.configure(api_key=GOOGLE_API_KEY)

try:
    model_test = genai.GenerativeModel('gemini-2.5-flash')
    response_test = model_test.generate_content("Say hello in one word")
    print(f"\n✅ Gemini API connected successfully!")
    print(f"   Test response: {response_test.text}")
    print(f"   Model: gemini-2.5-flash (FREE & FAST)")
except Exception as e:
    print(f"\n❌ API Error: {e}")
    print("\n💡 Make sure:")
    print("   1. API key is correct")
    print("   2. You have internet connection")
    print("   3. API is enabled at https://makersuite.google.com")

# ========================================
# 4. LOAD CLINICALBERT
# ========================================

print("\n" + "="*60)
print("🧠 LOADING CLINICALBERT")
print("="*60)

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

print(f"\n⏳ Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = bert_model.to(device)
bert_model.eval()

print(f"✅ ClinicalBERT loaded on {device}")

# ========================================
# 5. LOAD FAISS INDEX
# ========================================

print("\n" + "="*60)
print("📚 LOADING FAISS INDEX")
print("="*60)

index_file = FAISS_DIR / "medical_cases.index"
index = faiss.read_index(str(index_file))
print(f"✅ FAISS index loaded: {index.ntotal:,} cases")

metadata_file = FAISS_DIR / "metadata_mapping.pkl"
with open(metadata_file, 'rb') as f:
    metadata_mapping = pickle.load(f)
print(f"✅ Metadata loaded: {metadata_mapping['num_vectors']:,} records")

# ========================================
# 6. ARABIC TRANSLATION LAYER  ← ✅ جديد
# ========================================

class ArabicToEnglishTranslator:
    """
    ترجمة طبية ذكية باستخدام Gemini
    بتحول الأعراض العربية لمصطلحات طبية إنجليزية صح
    مش ترجمة حرفية
    """

    def __init__(self, gemini_model):
        self.gemini_model = gemini_model

    def translate(self, arabic_text: str) -> str:
        try:
            prompt = f"""Translate the following Arabic medical symptoms to English.
Use proper medical terminology, not literal translation.
Return ONLY the English translation, nothing else.

Arabic symptoms: {arabic_text}

English translation:"""

            response = self.gemini_model.generate_content(prompt)
            translated = response.text.strip()
            print(f"   🔄 Translated: {arabic_text}")
            print(f"   → {translated}")
            return translated

        except Exception as e:
            print(f"   ⚠️ Translation failed: {e} — using original text")
            return arabic_text

    def is_arabic(self, text: str) -> bool:
        """كشف تلقائي إذا كان النص عربي"""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return arabic_chars / max(len(text), 1) > 0.3


# ========================================
# 7. CREATE RAG MEDICAL ASSISTANT
# ========================================

class MedicalRAGAssistant:
    """
    Complete RAG-powered Medical Assistant with Google Gemini
    يدعم العربية والإنجليزية ✅
    """

    def __init__(self, bert_model, tokenizer, faiss_index, metadata, gemini_api_key, device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.index = faiss_index
        self.metadata = metadata
        self.device = device

        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

        # ✅ Arabic translator — بيستخدم نفس الـ Gemini model
        self.translator = ArabicToEnglishTranslator(self.gemini_model)

        print("✅ Medical RAG Assistant initialized")
        print(f"   Knowledge base: {self.index.ntotal:,} medical cases")
        print(f"   LLM: Google Gemini 2.5 Flash (FREE & FAST)")
        print(f"   Encoder: ClinicalBERT + Mean Pooling ✅")
        print(f"   Arabic support: ✅ Enabled")

    def encode_query(self, text):
        """Encode user query using ClinicalBERT with mean pooling"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # ✅ Mean pooling
        token_embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0]

    def retrieve_similar_cases(self, query_embedding, k=5):
        """Search FAISS for similar cases"""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # ✅ تنظيف الـ symptoms لو كانت أرقام
            symptoms = self.metadata['symptoms'][idx]
            if not symptoms or symptoms == "None reported" or symptoms.strip() == "":
                symptoms = f"Patient with {self.metadata['pathologies'][idx]}"
            elif all(c.isdigit() or c in ', @V' for c in symptoms.replace(' ', '')):
                # لو الـ symptoms أرقام codes → حولها لنص مفهوم
                symptoms = f"Patient with {self.metadata['pathologies'][idx]} presenting with multiple symptoms"

            results.append({
                'similarity': float(score),
                'pathology': self.metadata['pathologies'][idx],
                'symptoms': symptoms,
                'patient_id': self.metadata['patient_ids'][idx],
            })

        return results

    def build_rag_prompt(self, user_symptoms, retrieved_cases):
        """Build prompt with retrieved context"""

        context = "SIMILAR MEDICAL CASES FROM DATABASE:\n\n"
        for i, case in enumerate(retrieved_cases, 1):
            context += f"Case {i} (Similarity: {case['similarity']*100:.1f}%):\n"
            context += f"- Diagnosis: {case['pathology']}\n"
            context += f"- Symptoms: {case['symptoms']}\n\n"

        prompt = f"""You are a professional medical AI assistant.
A patient has described their symptoms. Your job is to:
1. Analyze what they said
2. Ask 2-3 smart follow-up questions to narrow down the diagnosis
3. Give a preliminary assessment based on what you know so far

PATIENT'S SYMPTOMS:
{user_symptoms}

{context}

Respond in this exact format:

**Preliminary Assessment:**
Based on your symptoms, this could be related to [mention top 1-2 possibilities from similar cases].

**To give you a more accurate diagnosis, I need to ask:**
1. [Question about duration/onset]
2. [Question about associated symptoms]
3. [Question about medical history or severity]

**Important:** [Any red flag warning if symptoms are serious]

Keep your tone warm, professional, and concise like a real doctor."""

        return prompt

    def generate_response(self, prompt,retry=3):
        """Generate response using Gemini"""
        import time
        try:
            generation_config = {
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 4096,
            }

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            if "429" in str(e) and retry > 0:
                wait_time = 60
                print(f"   ⏳ Rate limit — waiting {wait_time}s then retrying... ({retry} attempts left)")
                time.sleep(wait_time)
                return self.generate_response(prompt, retry=retry-1)
            return f"Error generating response: {e}"

    def add_medical_disclaimer(self, response):
        """Add safety disclaimer"""
        disclaimer = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ IMPORTANT MEDICAL DISCLAIMER

This response is generated by AI based on pattern matching with medical cases.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.

✋ Always seek the advice of your physician or other qualified health provider
   with any questions you may have regarding a medical condition.

🚨 EMERGENCY: If you are experiencing a medical emergency, call emergency
   services immediately (911, 123, or your local emergency number).

This AI assistant is for informational purposes only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        return response + disclaimer

    def chat(self, user_symptoms, top_k=5, verbose=True):
        """
        Main chat function - complete RAG pipeline
        يدعم العربية والإنجليزية تلقائياً
        """
        if verbose:
            print("\n" + "="*70)
            print("🏥 MEDICAL RAG ASSISTANT (Powered by Google Gemini)")
            print("="*70)
            print(f"\n👤 Patient Query: {user_symptoms}")

        # ✅ Step 0: Arabic detection & translation
        if self.translator.is_arabic(user_symptoms):
            if verbose:
                print("\n🌍 Arabic input detected — translating to English...")
            user_symptoms = self.translator.translate(user_symptoms)

        # Step 1: Encode query
        if verbose:
            print("\n🔄 Step 1: Encoding symptoms with ClinicalBERT (Mean Pooling)...")
        query_embedding = self.encode_query(user_symptoms)
        if verbose:
            print("   ✅ Symptoms encoded to 768-dim vector")

        # Step 2: Retrieve similar cases
        if verbose:
            print(f"\n🔍 Step 2: Searching {self.index.ntotal:,} medical cases...")
        retrieved_cases = self.retrieve_similar_cases(query_embedding, k=top_k)
        if verbose:
            print(f"   ✅ Found {len(retrieved_cases)} similar cases")
            print("\n📋 Top Similar Cases:")
            for i, case in enumerate(retrieved_cases[:3], 1):
                symptoms_preview = case['symptoms'][:60] + "..." if len(case['symptoms']) > 60 else case['symptoms']
                print(f"   {i}. {case['pathology']}")
                print(f"      Similarity: {case['similarity']*100:.1f}%")
                print(f"      Symptoms: {symptoms_preview}")

        # Step 3: Build prompt
        if verbose:
            print("\n📝 Step 3: Building RAG prompt with context...")
        prompt = self.build_rag_prompt(user_symptoms, retrieved_cases)
        if verbose:
            print("   ✅ Prompt ready")

        # Step 4: Generate response
        if verbose:
            print("\n🤖 Step 4: Generating response with Gemini...")
        response = self.generate_response(prompt)
        if verbose:
            print("   ✅ Response generated")

        # Step 5: Add disclaimer
        final_response = self.add_medical_disclaimer(response)

        if verbose:
            print("\n" + "="*70)
            print("💬 AI ASSISTANT RESPONSE:")
            print("="*70)
            print(final_response)
            print("\n" + "="*70)

        return {
            'query': user_symptoms,
            'retrieved_cases': retrieved_cases,
            'response': final_response,
            'raw_response': response,
            'needs_followup': True
        }

# ========================================
# 8. INITIALIZE ASSISTANT
# ========================================

print("\n" + "="*60)
print("🚀 INITIALIZING MEDICAL RAG ASSISTANT")
print("="*60)

assistant = MedicalRAGAssistant(
    bert_model=bert_model,
    tokenizer=tokenizer,
    faiss_index=index,
    metadata=metadata_mapping,
    gemini_api_key=GOOGLE_API_KEY,
    device=device
)

# ========================================
# 9. TEST WITH EXAMPLES
# ========================================

print("\n" + "="*60)
print("🧪 TESTING WITH EXAMPLE QUERIES")
print("="*60)

test_queries = [
    "I have high fever, severe cough, and body aches for 3 days",       # إنجليزي
    "عندي حمى شديدة وسعال وألم في الجسم من 3 أيام",                    # ✅ عربي
    "Sudden chest pain with difficulty breathing and sweating",          # إنجليزي
    "عندي صداع شديد مع غثيان وحساسية للضوء",                           # ✅ عربي
]

print(f"\n📝 Running {len(test_queries)} test queries (Arabic + English)...")
print("   (Each query takes ~10-15 seconds)")

test_results = []

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"TEST QUERY {i}/{len(test_queries)}")
    print(f"{'='*70}")

    result = assistant.chat(query, top_k=5, verbose=True)
    test_results.append(result)

    result_file = BASE_DIR / f"test_result_{i}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json_result = {
            'query': result['query'],
            'response': result['raw_response'],
            'top_cases': [
                {
                    'pathology': c['pathology'],
                    'similarity': f"{c['similarity']*100:.1f}%"
                }
                for c in result['retrieved_cases'][:3]
            ]
        }
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Result saved to: {result_file.name}")

# ========================================
# 10. INTERACTIVE MODE
# ========================================

print("\n" + "="*60)
print("💬 INTERACTIVE CHAT MODE")
print("="*60)

def interactive_chat():
    """Interactive chat loop — يدعم العربية والإنجليزية"""
    print("\n🏥 Welcome to Medical RAG Assistant!")
    print("   Powered by ClinicalBERT + FAISS + Google Gemini")
    print("\n💡 Describe your symptoms in Arabic or English.")
    print("   اكتب أعراضك بالعربي أو الإنجليزي")
    print("   Type 'quit' to stop.\n")

    conversation_history = []

    while True:
        try:
            user_input = input("👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q', 'stop']:
                print("\n👋 Thank you for using Medical RAG Assistant!")
                print(f"   Total queries in session: {len(conversation_history)}")
                break

            if not user_input:
                continue

            print("\n⏳ Processing... (this may take 10-15 seconds)")
            result = assistant.chat(user_input, top_k=5, verbose=False)

            print(f"\n🤖 Assistant:\n")
            print(result['response'])
            print("\n" + "-"*70 + "\n")

            conversation_history.append({
                'user': user_input,
                'assistant': result['raw_response'],
                'top_diagnosis': result['retrieved_cases'][0]['pathology'] if result['retrieved_cases'] else 'N/A'
            })

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

print("\n🎯 Starting interactive chat in 3 seconds...")
print("   (Press Ctrl+C to skip)")

import time
try:
    time.sleep(3)
    interactive_chat()
except KeyboardInterrupt:
    print("\n⏭️ Skipped interactive mode")

# ========================================
# 11. SAVE SYSTEM INFO
# ========================================

print("\n" + "="*60)
print("💾 SAVING SYSTEM INFORMATION")
print("="*60)

system_info = {
    'bert_model': MODEL_NAME,
    'llm_model': 'Google Gemini 2.5 Flash',
    'pooling_method': 'mean_pooling',
    'arabic_support': True,
    'translation_method': 'Gemini Medical Translation',
    'index_size': int(index.ntotal),
    'created_date': pd.Timestamp.now().isoformat(),
    'components': {
        'encoder': 'ClinicalBERT (768-dim) + Mean Pooling',
        'vector_db': 'FAISS IndexFlatIP',
        'llm': 'Google Gemini 2.5 Flash (FREE)',
        'translation': 'Gemini Medical Translation ✅',
        'safety': 'Medical disclaimer included'
    },
    'test_queries_run': len(test_results),
    'api_cost': 'FREE (Gemini API)'
}

system_file = BASE_DIR / "rag_system_info.json"
with open(system_file, 'w') as f:
    json.dump(system_info, f, indent=2)

print(f"✅ System info saved: {system_file.name}")

print("\n" + "="*60)
print("📊 SYSTEM SUMMARY")
print("="*60)
print(f"\n✅ Components Initialized:")
print(f"   • ClinicalBERT + Mean Pooling: ✅ Loaded")
print(f"   • FAISS Index: ✅ {index.ntotal:,} cases")
print(f"   • Google Gemini 2.5 Flash: ✅ Connected (FREE)")
print(f"   • Arabic Support: ✅ Enabled")
print(f"   • Medical Disclaimer: ✅ Included")

print(f"\n📁 Generated Files:")
print(f"   • Test results: {len(test_results)} queries saved")
print(f"   • System info: rag_system_info.json")
print(f"   • Location: {BASE_DIR}")

# ========================================
# COMPLETION
# ========================================

print("\n" + "="*60)
print("✅ STEP 4 COMPLETE - RAG SYSTEM READY!")
print("="*60)

print("\n🎉 Congratulations! Your Medical RAG Assistant is fully functional!")

print("\n🎯 Examples:")
print("\n   # English:")
print("   result = assistant.chat('I have fever and cough')")
print("\n   # Arabic:")
print("   result = assistant.chat('عندي حمى وسعال')")

print("\n💰 Cost: $0 (100% FREE with Gemini API)")
print("\n🚀 Your Medical RAG Assistant is READY!")