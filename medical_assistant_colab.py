# ==============================================================================
# ██████████████████████████████████████████████████████████████████████████████
#
#   INTELLIGENT MEDICAL ASSISTANT — Google Colab Script
#   Author  : AI & Machine Learning Postgraduate Project
#   Purpose : Verify medicine packaging via TensorFlow, extract active
#             ingredients via EasyOCR, and check for drug interactions
#             against a local CSV database.
#
#   PIPELINE OVERVIEW
#   ─────────────────
#   [Image Upload]
#        │
#        ▼
#   PHASE 1 — TensorFlow Filter (MobileNetV2 on ImageNet)
#        │  Is this image a medicine / pill bottle / packet?
#        │  NO  ──► Stop. Show red border. Print message.
#        │  YES ──► Continue.
#        ▼
#   PHASE 2 — EasyOCR Reader
#        │  Extract all visible text (Polish + English).
#        │  Clean / normalise the text.
#        ▼
#   PHASE 3 — Database Matching
#        │  Search for the active ingredient in meds_database.csv.
#        │  FOUND ──► Print a WARNING REPORT.
#        │  NOT FOUND ──► Print a safe-pass message.
#        ▼
#   [Display image with GREEN (safe) or RED (warning) border]
#
# ==============================================================================


# ==============================================================================
# SECTION 0 — LIBRARY INSTALLATION
# ==============================================================================
# These commands install the libraries that are NOT pre-installed in Colab.
# Run this cell first; Colab may ask you to restart the runtime afterwards —
# that is normal.  Just press "Restart Runtime" and re-run from the top.

# The exclamation mark (!) tells Colab to run the command in the terminal,
# not in Python itself.
# ==============================================================================

# Uncomment the lines below and run once when starting a fresh Colab session.
# After installation, you can comment them out again to speed up future runs.

# !pip install easyocr          # OCR library that supports Polish and English
# !pip install tensorflow       # Deep-learning framework (MobileNetV2 lives here)
# !pip install pandas           # Tabular data — used to read the CSV database
# !pip install opencv-python    # Image processing (reading, resizing, colour ops)
# !pip install matplotlib       # Plotting — used to display the final image


# ==============================================================================
# SECTION 1 — IMPORTS
# ==============================================================================
# We import every library we need at the top of the script.
# This is a Python best-practice: keep all imports in one visible place.
# ==============================================================================

import os                          # For file-path operations (os.path.exists, etc.)
import re                          # Regular expressions — used for text cleaning
import warnings                    # To suppress non-critical library warnings

import cv2                         # OpenCV — image reading & colour manipulation
import easyocr                     # OCR engine supporting multiple languages
import matplotlib.pyplot as plt    # Plotting / visualisation library
import matplotlib.patches as mpatches  # Used to draw coloured rectangles on plots
import numpy as np                 # Numerical arrays (images are numpy arrays)
import pandas as pd                # DataFrames — used to load and search the CSV
import tensorflow as tf            # Deep-learning framework

# Keras lives inside TensorFlow since TF 2.x
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,              # Scales pixel values to the range MobileNetV2 expects
    decode_predictions,            # Converts raw model output to human-readable labels
)
from tensorflow.keras.preprocessing import image as keras_image

# Suppress noisy deprecation warnings so the output stays readable
warnings.filterwarnings("ignore")

print("✅ All libraries imported successfully.")


# ==============================================================================
# SECTION 2 — GOOGLE DRIVE MOUNT
# ==============================================================================
# Colab runs on a virtual machine. To access YOUR files (the CSV database,
# the test image), we must first "mount" your Google Drive, which makes your
# Drive appear as a folder at /content/drive/.
# ==============================================================================

from google.colab import drive

print("🔗 Mounting Google Drive …")
drive.mount("/content/drive")
print("✅ Google Drive mounted at /content/drive/")


# ==============================================================================
# SECTION 3 — CONFIGURATION  ← CHANGE YOUR PATHS HERE
# ==============================================================================
# All user-configurable settings live in this one section.
# You should ONLY need to edit values here — not anywhere else in the script.
# This makes the script easy to reuse for different exams or projects.
# ==============================================================================

# ── 3.1  Database path ────────────────────────────────────────────────────────
# Full path to your CSV file inside Google Drive.
# Make sure the file exists at exactly this location.
DATABASE_PATH = "/content/drive/MyDrive/AI_Project/meds_database.csv"

# ── 3.2  Image path ───────────────────────────────────────────────────────────
# Full path to the medicine-package image you want to analyse.
# Supported formats: .jpg, .jpeg, .png, .bmp
IMAGE_PATH = "/content/drive/MyDrive/AI_Project/medicine_package.jpg"

# ── 3.3  OCR languages ────────────────────────────────────────────────────────
# List of language codes EasyOCR will try to recognise.
# 'pl' = Polish,  'en' = English.
# Add more codes if your packaging uses other languages (e.g., 'de', 'fr').
OCR_LANGUAGES = ["pl", "en"]

# ── 3.4  TensorFlow confidence threshold ──────────────────────────────────────
# MobileNetV2 returns a confidence score (0.0 → 1.0) for each predicted label.
# If the score of any medicine-related label exceeds this threshold, Phase 1 passes.
# Lower value = more permissive filter.  Higher value = stricter filter.
TF_CONFIDENCE_THRESHOLD = 0.05     # 5 % — intentionally permissive for prototypes

# ── 3.5  Medicine-related ImageNet labels ─────────────────────────────────────
# MobileNetV2 was trained on ImageNet which has 1,000 categories.
# None of them is exactly "medicine box", so we list the CLOSEST ones.
# The filter passes if any of these substrings appears in the top-5 predictions.
MEDICINE_RELATED_LABELS = [
    "pill",          # pill organiser / tablet
    "packet",        # blister pack
    "bottle",        # pill bottle
    "container",     # generic container
    "envelope",      # flat packet
    "box",           # cardboard box
    "carton",        # carton packaging
    "label",         # labelled item
    "medicine",      # direct hit (rare but possible)
    "Band_Aid",      # medical supply — close enough
    "rubber_eraser", # ImageNet sometimes maps small boxes here
    "wallet",        # sometimes flat blister packs match this
]

print("✅ Configuration loaded.")
print(f"   DATABASE : {DATABASE_PATH}")
print(f"   IMAGE    : {IMAGE_PATH}")


# ==============================================================================
# SECTION 4 — HELPER FUNCTIONS
# ==============================================================================
# We define all our reusable functions here, BEFORE using them.
# Functions are like named "recipes" — define once, call many times.
# Each function has a docstring (the text in triple quotes) that explains
# what it does, what it receives as input, and what it returns.
# ==============================================================================


# ── 4.1  Load and validate the CSV database ───────────────────────────────────

def load_database(path: str) -> pd.DataFrame | None:
    """
    Load the medication database from a CSV file and validate its structure.

    Parameters
    ----------
    path : str
        Absolute file-system path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        A pandas DataFrame if loading succeeds; None if any error occurs.
    """
    print(f"\n📂 Loading database from:\n   {path}")

    try:
        # Check the file actually exists before trying to read it
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Database not found at '{path}'.\n"
                "Please check SECTION 3 and verify the path is correct."
            )

        # Read the CSV into a DataFrame
        df = pd.read_csv(path)

        # Validate that the expected columns are present
        required_columns = {"active_ingredient", "interactions_to_avoid", "interaction_side_effects"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"The CSV is missing these required columns: {missing}\n"
                f"Columns found: {list(df.columns)}"
            )

        # Normalise the active_ingredient column to lowercase for reliable matching
        df["active_ingredient"] = df["active_ingredient"].str.lower().str.strip()

        print(f"✅ Database loaded successfully — {len(df)} records found.")
        print(f"   Columns : {list(df.columns)}")
        return df

    except FileNotFoundError as e:
        print(f"❌ FILE ERROR: {e}")
        return None
    except ValueError as e:
        print(f"❌ DATA ERROR: {e}")
        return None
    except Exception as e:
        # Catch-all for unexpected errors (e.g., malformed CSV)
        print(f"❌ UNEXPECTED ERROR while loading database: {e}")
        return None


# ── 4.2  Load the image from disk ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray | None:
    """
    Load an image from disk using OpenCV and convert it to RGB colour order.

    Why convert to RGB?
    - OpenCV reads images in BGR (Blue-Green-Red) order by default.
    - TensorFlow and matplotlib both expect RGB (Red-Green-Blue).
    - If we skip this step, colours appear wrong and predictions suffer.

    Parameters
    ----------
    path : str
        Absolute path to the image file.

    Returns
    -------
    numpy.ndarray or None
        The image as a NumPy array (height × width × 3 channels) or None on error.
    """
    print(f"\n🖼️  Loading image from:\n   {path}")

    try:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Image not found at '{path}'.\n"
                "Please check SECTION 3 and make sure the image is uploaded."
            )

        # cv2.imread returns a NumPy array; None if the file cannot be decoded
        img_bgr = cv2.imread(path)

        if img_bgr is None:
            raise ValueError(
                "OpenCV could not decode the image. "
                "Check that the file is a valid JPEG, PNG, or BMP."
            )

        # Convert colour order BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h, w, c = img_rgb.shape
        print(f"✅ Image loaded — size: {w}×{h} px, channels: {c}")
        return img_rgb

    except FileNotFoundError as e:
        print(f"❌ FILE ERROR: {e}")
        return None
    except ValueError as e:
        print(f"❌ IMAGE ERROR: {e}")
        return None
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR while loading image: {e}")
        return None


# ── 4.3  Phase 1 — TensorFlow filter ─────────────────────────────────────────

def phase1_tensorflow_filter(img_rgb: np.ndarray) -> tuple[bool, list]:
    """
    Use MobileNetV2 (pre-trained on ImageNet) to check whether the image
    looks like a medicine package.

    HOW MOBILENETV2 WORKS (for your exam)
    ──────────────────────────────────────
    MobileNetV2 is a Convolutional Neural Network (CNN) designed to be
    lightweight enough for mobile devices while still being highly accurate.
    It was pre-trained on ImageNet — a dataset of 1.2 million images across
    1,000 categories — so it already "knows" what millions of objects look like.
    We use it in *inference mode* (no training): we just pass our image through
    the network and read the top-5 predicted categories.

    Steps inside this function:
    1. Resize the image to 224×224 px (the exact input size MobileNetV2 expects).
    2. Expand dimensions: the model expects a BATCH of images, shape (N, 224, 224, 3).
       We have 1 image, so we add a batch dimension → (1, 224, 224, 3).
    3. Call preprocess_input() to scale pixel values from [0, 255] to [-1, 1].
    4. Run model.predict() to get 1,000 confidence scores.
    5. decode_predictions() converts those scores to (class_id, label, score) tuples.
    6. Check if any top-5 label contains a medicine-related keyword.

    Parameters
    ----------
    img_rgb : np.ndarray
        The image in RGB order.

    Returns
    -------
    tuple[bool, list]
        - bool  : True if the image passes the medicine filter, False otherwise.
        - list  : The top-5 (label, score) pairs (useful for debugging / exam).
    """
    print("\n" + "="*60)
    print("  PHASE 1 — TensorFlow Medicine Filter")
    print("="*60)

    try:
        # ── Step 1: Load MobileNetV2 ──────────────────────────────────────────
        # include_top=True  → include the final classification layer (1,000 classes)
        # weights='imagenet' → use the pre-trained weights (downloads ~14 MB once)
        print("⏳ Loading MobileNetV2 model (pre-trained on ImageNet) …")
        model = MobileNetV2(weights="imagenet", include_top=True)
        print("✅ Model loaded.")

        # ── Step 2: Prepare the image ─────────────────────────────────────────
        # Resize to exactly 224×224 pixels — MobileNetV2's required input shape
        img_resized = cv2.resize(img_rgb, (224, 224))

        # Convert the NumPy array to a Keras-friendly format
        img_array = keras_image.img_to_array(img_resized)  # shape: (224, 224, 3)

        # Add the batch dimension → shape: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        # Scale pixels from [0, 255] to [-1, 1] (what MobileNetV2 was trained with)
        img_preprocessed = preprocess_input(img_batch)

        # ── Step 3: Run inference ─────────────────────────────────────────────
        print("⏳ Running inference …")
        predictions = model.predict(img_preprocessed, verbose=0)

        # decode_predictions returns the top-k (class_id, human_label, score) tuples
        top5 = decode_predictions(predictions, top=5)[0]

        # Reformat to a cleaner list of (label, score) for display
        top5_clean = [(label, float(score)) for (_, label, score) in top5]

        # ── Step 4: Display results ───────────────────────────────────────────
        print("\n  Top-5 ImageNet predictions:")
        print(f"  {'Label':<25} {'Confidence':>10}")
        print("  " + "-"*37)
        for label, score in top5_clean:
            print(f"  {label:<25} {score:>9.2%}")

        # ── Step 5: Check against medicine-related keywords ───────────────────
        passed = False
        for label, score in top5_clean:
            label_lower = label.lower().replace("_", " ")
            for keyword in MEDICINE_RELATED_LABELS:
                if keyword.lower() in label_lower and score >= TF_CONFIDENCE_THRESHOLD:
                    print(f"\n  ✅ FILTER PASSED — '{label}' matched keyword "
                          f"'{keyword}' with confidence {score:.2%}")
                    passed = True
                    break
            if passed:
                break

        if not passed:
            print(
                "\n  ❌ FILTER FAILED — No medicine-related category detected "
                f"above the {TF_CONFIDENCE_THRESHOLD:.0%} confidence threshold.\n"
                "  The uploaded image does not appear to be a medicine package.\n"
                "  Pipeline stopped."
            )

        return passed, top5_clean

    except Exception as e:
        print(f"❌ ERROR in Phase 1 (TensorFlow): {e}")
        return False, []


# ── 4.4  Phase 2 — EasyOCR text extraction ────────────────────────────────────

def phase2_easyocr_extraction(img_rgb: np.ndarray) -> str | None:
    """
    Use EasyOCR to extract all visible text from the image.

    HOW EASYOCR WORKS (for your exam)
    ───────────────────────────────────
    EasyOCR uses a deep-learning pipeline with two main stages:
    1. DETECTOR  — finds where text regions are in the image (bounding boxes).
    2. RECOGNISER — reads the characters inside each bounding box.
    The model supports 80+ languages; we use Polish ('pl') and English ('en').

    After extraction we *clean* the raw text:
    - Convert to lowercase  (so 'Ibuprofen' matches 'ibuprofen' in the database)
    - Remove special characters with a regular expression
    - Collapse multiple spaces into one

    Parameters
    ----------
    img_rgb : np.ndarray
        The image in RGB order.

    Returns
    -------
    str or None
        A single cleaned string of all detected text, or None on error.
    """
    print("\n" + "="*60)
    print("  PHASE 2 — EasyOCR Text Extraction")
    print("="*60)

    try:
        # ── Step 1: Initialise the OCR reader ─────────────────────────────────
        # gpu=False → use CPU (safer for Colab; set True if GPU runtime is active)
        print(f"⏳ Initialising EasyOCR for languages: {OCR_LANGUAGES} …")
        reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
        print("✅ EasyOCR reader ready.")

        # ── Step 2: Run OCR ───────────────────────────────────────────────────
        # detail=1 returns (bounding_box, text, confidence) for each detection.
        # detail=0 would return text strings only — we use 1 for richer output.
        print("⏳ Extracting text from image …")
        results = reader.readtext(img_rgb, detail=1)

        if not results:
            print("⚠️  No text detected in the image. "
                  "Try a higher-resolution or better-lit photo.")
            return None

        # ── Step 3: Display raw OCR output ────────────────────────────────────
        print(f"\n  Raw OCR detections ({len(results)} found):")
        print(f"  {'Text':<30} {'Confidence':>10}")
        print("  " + "-"*42)
        for (_, text, confidence) in results:
            print(f"  {text:<30} {confidence:>9.2%}")

        # ── Step 4: Combine all detected text into one string ─────────────────
        raw_combined = " ".join([text for (_, text, _) in results])

        # ── Step 5: Clean the text ────────────────────────────────────────────
        # a) Lowercase everything
        cleaned = raw_combined.lower()
        # b) Remove anything that is NOT a letter, digit, or space
        #    The regex [^a-ząćęłńóśźż0-9\s] matches unwanted characters
        #    (Polish letters are explicitly listed so they are kept)
        cleaned = re.sub(r"[^a-ząćęłńóśźż0-9\s]", " ", cleaned)
        # c) Collapse multiple consecutive whitespace characters into one space
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        print(f"\n  ✅ Cleaned text:\n  \"{cleaned}\"")
        return cleaned

    except Exception as e:
        print(f"❌ ERROR in Phase 2 (EasyOCR): {e}")
        return None


# ── 4.5  Phase 3 — Database matching ─────────────────────────────────────────

def phase3_database_matching(cleaned_text: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Search the medication database for active ingredients mentioned in the OCR text.

    MATCHING STRATEGY
    ─────────────────
    For each active_ingredient in the database we check whether that ingredient
    name appears as a *substring* of the cleaned OCR text.
    Example: if the OCR text contains "ibuprofen 400 mg tablets" and the database
    has an entry with active_ingredient = "ibuprofen", the match succeeds.

    Parameters
    ----------
    cleaned_text : str
        The normalised text returned by Phase 2.
    df : pd.DataFrame
        The medication database loaded in load_database().

    Returns
    -------
    pd.DataFrame or None
        Rows from the database where a match was found, or None if no match.
    """
    print("\n" + "="*60)
    print("  PHASE 3 — Database Matching")
    print("="*60)

    try:
        matched_rows = []

        print("🔍 Searching for active ingredients in detected text …")
        for _, row in df.iterrows():
            ingredient = row["active_ingredient"]   # already lowercased in load_database()
            if ingredient in cleaned_text:
                matched_rows.append(row)
                print(f"   ✅ MATCH FOUND: '{ingredient}'")

        if matched_rows:
            return pd.DataFrame(matched_rows)
        else:
            print("   ℹ️  No known active ingredients were detected in the text.")
            return None

    except Exception as e:
        print(f"❌ ERROR in Phase 3 (Database Matching): {e}")
        return None


# ── 4.6  Print the Warning Report ────────────────────────────────────────────

def print_warning_report(matches: pd.DataFrame) -> None:
    """
    Print a clearly formatted WARNING REPORT for every matched active ingredient.

    Parameters
    ----------
    matches : pd.DataFrame
        The subset of the database that matched the OCR text.
    """
    print("\n" + "█"*60)
    print("  ⚠️   DRUG INTERACTION WARNING REPORT   ⚠️")
    print("█"*60)
    print(f"  {len(matches)} active ingredient(s) identified in this package.\n")

    for i, (_, row) in enumerate(matches.iterrows(), start=1):
        print(f"  ── Entry {i} " + "─"*46)
        print(f"  Active Ingredient   : {row['active_ingredient'].title()}")
        print(f"  Interactions to Avoid:\n    {row['interactions_to_avoid']}")
        print(f"  Possible Side Effects if Combined:\n    {row['interaction_side_effects']}")
        print()

    print("█"*60)
    print("  ‼️  Consult a licensed pharmacist or physician before")
    print("  combining medications. This tool is NOT medical advice.")
    print("█"*60 + "\n")


# ── 4.7  Display the image with a coloured border ─────────────────────────────

def display_image_with_border(img_rgb: np.ndarray, match_found: bool,
                               top5: list | None = None) -> None:
    """
    Use matplotlib to display the analysed image with:
    - GREEN border + title if a drug interaction was found (warning state).
    - RED  border + title if the image failed Phase 1 (not a medicine package).
    - ORANGE border if Phase 1 passed but no interaction match was found.

    Wait — green for a WARNING?
    Yes: green means "the system worked and found something important".
    The colour signals pipeline success / finding, not danger level.
    You can swap colours in the variables below if your course prefers otherwise.

    Parameters
    ----------
    img_rgb : np.ndarray
        The image to display.
    match_found : bool | str
        True → interaction found (green border)
        False → no interaction found (orange border)
        "failed" → Phase 1 filter failed (red border)
    top5 : list or None
        Top-5 TF predictions for display in the plot subtitle.
    """
    # Choose border colour and title text
    if match_found == "failed":
        border_color = "red"
        title = "❌ FILTER FAILED — Not recognised as a medicine package"
    elif match_found:
        border_color = "green"
        title = "⚠️  INTERACTION DETECTED — See WARNING REPORT above"
    else:
        border_color = "orange"
        title = "✅ No Known Interactions Found in Database"

    # Build a subtitle from the top-5 TF predictions (if available)
    subtitle = ""
    if top5:
        top3_str = " | ".join([f"{lbl} ({sc:.1%})" for lbl, sc in top5[:3]])
        subtitle = f"Top-3 TF predictions: {top3_str}"

    # ── Matplotlib figure setup ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))

    # Display the image
    ax.imshow(img_rgb)
    ax.axis("off")   # Hide axis ticks and labels — cleaner look

    # Title and subtitle
    ax.set_title(title, fontsize=13, fontweight="bold", color=border_color, pad=12)
    if subtitle:
        fig.text(0.5, 0.02, subtitle, ha="center", fontsize=9, color="grey")

    # Draw a thick coloured rectangle AROUND the image to act as a border
    # The rectangle is placed in figure-fraction coordinates (0 to 1).
    rect = plt.Rectangle(
        (0.01, 0.05),   # (x, y) of bottom-left corner
        0.98,           # width
        0.93,           # height
        linewidth=6,
        edgecolor=border_color,
        facecolor="none",
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.add_artist(rect)

    plt.tight_layout()
    plt.show()
    print(f"🖼️  Image displayed with {border_color.upper()} border.")


# ==============================================================================
# SECTION 5 — MAIN PIPELINE
# ==============================================================================
# This is the orchestration function that calls every phase in order.
# Keeping the "director" logic separate from the "worker" functions
# (defined above) is a software-engineering best practice called
# Separation of Concerns — it makes each part easier to test and explain.
# ==============================================================================

def run_medical_assistant_pipeline():
    """
    Execute the full intelligent medical assistant pipeline:
      0. Load database and image.
      1. TensorFlow filter (Phase 1).
      2. EasyOCR text extraction (Phase 2).
      3. Database matching (Phase 3).
      4. Display result image with coloured border.
    """

    print("\n" + "="*60)
    print("  🏥 INTELLIGENT MEDICAL ASSISTANT — Starting Pipeline")
    print("="*60 + "\n")

    # ── Step 0a: Load the database ────────────────────────────────────────────
    df = load_database(DATABASE_PATH)
    if df is None:
        print("\n🛑 Pipeline aborted: database could not be loaded.")
        return   # Exit the function early — nothing more we can do

    # ── Step 0b: Load the image ───────────────────────────────────────────────
    img_rgb = load_image(IMAGE_PATH)
    if img_rgb is None:
        print("\n🛑 Pipeline aborted: image could not be loaded.")
        return

    # ── Step 1: TensorFlow filter ─────────────────────────────────────────────
    passed_filter, top5 = phase1_tensorflow_filter(img_rgb)

    if not passed_filter:
        # Image does not look like a medicine package — stop here
        display_image_with_border(img_rgb, match_found="failed", top5=top5)
        print("\n🏁 Pipeline finished — image rejected by TensorFlow filter.")
        return

    # ── Step 2: EasyOCR text extraction ──────────────────────────────────────
    cleaned_text = phase2_easyocr_extraction(img_rgb)

    if cleaned_text is None:
        # OCR returned nothing useful
        print("\n⚠️  No text could be extracted from the image.")
        display_image_with_border(img_rgb, match_found=False, top5=top5)
        print("\n🏁 Pipeline finished — no text detected.")
        return

    # ── Step 3: Database matching ─────────────────────────────────────────────
    matches = phase3_database_matching(cleaned_text, df)

    # ── Step 4: Report & display ──────────────────────────────────────────────
    if matches is not None and not matches.empty:
        # At least one interaction was found
        print_warning_report(matches)
        display_image_with_border(img_rgb, match_found=True, top5=top5)
    else:
        # No interaction found in the database
        print("\n✅ RESULT: No drug interactions found in the database "
              "for the detected active ingredients.")
        print("   Remember: absence in the database does NOT guarantee safety.")
        display_image_with_border(img_rgb, match_found=False, top5=top5)

    print("\n🏁 Pipeline completed successfully.")


# ==============================================================================
# SECTION 6 — ENTRY POINT
# ==============================================================================
# In Python, the block below runs ONLY when this file is executed directly
# (not when it is imported as a module by another script).
# In Colab, simply running the cell executes everything, so this block will
# always trigger — it is still good practice to include it.
# ==============================================================================

if __name__ == "__main__":
    run_medical_assistant_pipeline()


# ==============================================================================
# SECTION 7 — SAMPLE CSV DATABASE (for testing)
# ==============================================================================
# Paste the content below into a new file called meds_database.csv and upload
# it to /content/drive/MyDrive/AI_Project/ in your Google Drive.
#
# active_ingredient,interactions_to_avoid,interaction_side_effects
# ibuprofen,aspirin; warfarin; lithium,Increased bleeding risk; kidney damage; elevated lithium toxicity
# paracetamol,alcohol; warfarin,Liver damage; enhanced anticoagulant effect
# amoxicillin,methotrexate; warfarin,Methotrexate toxicity; increased bleeding risk
# metformin,alcohol; iodine contrast agents,Lactic acidosis; kidney failure risk
# omeprazole,clopidogrel; methotrexate,Reduced antiplatelet effect; methotrexate toxicity
# atorvastatin,clarithromycin; erythromycin; gemfibrozil,Myopathy; rhabdomyolysis risk
# aspirin,warfarin; ibuprofen; naproxen,Severe gastrointestinal bleeding
# simvastatin,amiodarone; verapamil; amlodipine,Muscle damage (rhabdomyolysis)
# warfarin,aspirin; ibuprofen; paracetamol; amoxicillin,Life-threatening bleeding
# diazepam,alcohol; opioids; antihistamines,Severe CNS and respiratory depression
#
# ==============================================================================
