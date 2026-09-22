"""
Shared protocol constants for Tasks 2 & 3. Both tasks MUST import from here so
splits, class ordering, and the seed stay identical across the two tasks.
"""
SEED = 6304
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]  # official PACS order
N_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 224
RESIZE_SIZE = 256  # resize-then-crop, per spec

BATCH_SIZE_PER_SOURCE_DOMAIN = 8   # 8 x 3 source domains = 24 source examples/batch
BATCH_SIZE_TARGET = 24              # Task 2 only: matches total source batch size

MAX_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
