import os
import shutil
import random

# --- Config ---
ORIG_DIR = "dataset/OriginalDataset"
AUG_DIR  = "dataset/AugmentedAlzheimerDataset"
TARGET_DIR = "dataset/Processed"
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
SPLIT = {'train':0.7, 'val':0.15, 'test':0.15}
random.seed(42)

# --- Cria pastas Processed/train|val|test/CLASSE ---
def create_dirs():
    for split in SPLIT.keys():
        for cls in CLASSES:
            path = os.path.join(TARGET_DIR, split, cls)
            os.makedirs(path, exist_ok=True)

# --- Copia imagens para as pastas Processed ---
def copy_images(src_dir):
    for cls in CLASSES:
        cls_path = os.path.join(src_dir, cls)
        if not os.path.exists(cls_path):
            print(f"Atenção: pasta {cls_path} não existe, pulando.")
            continue
        images = [f for f in os.listdir(cls_path) if f.lower().endswith('.jpg')]
        random.shuffle(images)
        n = len(images)
        train_end = int(SPLIT['train']*n)
        val_end   = train_end + int(SPLIT['val']*n)

        for i, img_name in enumerate(images):
            if i < train_end:
                split = 'train'
            elif i < val_end:
                split = 'val'
            else:
                split = 'test'
            src_path = os.path.join(cls_path, img_name)
            dst_path = os.path.join(TARGET_DIR, split, cls, img_name)
            shutil.copy2(src_path, dst_path)

# --- Execução ---
create_dirs()
copy_images(ORIG_DIR)
copy_images(AUG_DIR)

print("Distribuição completa em:", TARGET_DIR)
