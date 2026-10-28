KAGGLE_ROOT = r"D:\New research fog work\kaggle_fog\Competition Dataset"
FOGSTAR_CSV = r"D:\New research fog work\FoG STAR\sensor_data.csv"
FOGSTAR_AXES = (("back_acc_x", 1.0), ("back_acc_y", 1.0), ("back_acc_z", 1.0))
DAPHNET_ROOT = r"D:\Daphnet dataset\dataset"
DAPHNET_AXES = (("tr_y", -1.0), ("tr_z", 1.0), ("tr_x", 1.0))
MULTIMODAL_ROOT = r"D:\New research fog work\Multimodal dataset 2021\r8gmbtv7w2-3\Filtered Data"
MULTIMODAL_AXES = ((46, -1.0), (47, 1.0), (48, 1.0))
MULTIMODAL_LSB_PER_G = 8192.0
PREP = "prepared"

DATASETS = {"tdcsfog": False, "defog": False, "Daphnet": False, "Multimodal": False, "FoG-STAR": True}
DEVELOPMENT = ["tdcsfog", "defog"]
EXTERNAL = ["Daphnet", "Multimodal", "FoG-STAR"]
SELECTION_DATASET = "defog"
CONTEXTS = {"home": "defog", "lab": "tdcsfog"}
CONTEXT_OF = {"defog": "home", "tdcsfog": "lab", "Daphnet": "lab", "Multimodal": "lab", "FoG-STAR": "lab"}
OUT = "results"

FS = 100
WIN = 50
NFFT = 256
MODE_RES = 0.01
FOG_MIN_SAMPLES = 30
PRE_W = 8
POST_W = 6
MERGE_GAP_S = 2.0
BAND = (0.5, 20.0)
FILT_ORDER = 4
G = 9.80665
SEED = 42

CANDIDATES = {"ctx8_w1": (8, 1.0), "ctx8_w2": (8, 2.0), "ctx16_w1": (16, 1.0), "ctx16_w2": (16, 2.0), "ctx32_w1": (32, 1.0), "ctx32_w2": (32, 2.0)}
N_FOLDS = 5
VAL_FRACTION = 0.2
MAX_EPOCHS = 12
PATIENCE = 3
BATCH = 256
FA_TARGET = 15.0
SPEC_FLOOR = 0.70
N_SHIFTS = 2000
GRID = [(round(a, 2), round(b, 2)) for a in [x / 100 for x in range(30, 96, 5)] for b in [y / 10 for y in range(1, 10)] if b <= a + 1e-9]
