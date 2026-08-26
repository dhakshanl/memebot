import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
AUG_DIR = os.path.join(BASE_DIR, 'data', 'augmented')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FEATURES_DIR = BASE_DIR

LABELS = ['backgroundnoise', 'eppudra', 'rajini']

SR = 22050
DURATION = 4.0                       # seconds, fixed window every clip is padded/cropped to
N_SAMPLES = int(SR * DURATION)
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
N_FRAMES = 1 + N_SAMPLES // HOP_LENGTH   # frames produced by librosa for N_SAMPLES samples

EMBEDDING_DIM = 128

EMBED_MODEL_PATH = os.path.join(MODELS_DIR, 'embedding_model.keras')
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, 'classifier_model.keras')
INDEX_PATH = os.path.join(MODELS_DIR, 'reference_index.npz')
X_PATH = os.path.join(FEATURES_DIR, 'X.npy')
Y_PATH = os.path.join(FEATURES_DIR, 'y.npy')
