"""Explicit local data locations; dataset files are obtained from their authors."""
import os, json
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
GH=REPO
RES=REPO/'results'
config_path=os.environ.get('LLIE_STUDY_CONFIG')
CONFIG=json.loads(Path(config_path).read_text()) if config_path else {}
BASE=Path(config_path).resolve().parent if config_path else Path.cwd()
def location(key, default):
    value=Path(CONFIG.get(key,default)).expanduser()
    return value if value.is_absolute() else BASE/value
OUT=location('work_dir','study_work')
PREPROCESSED=location('preprocessed','preprocessed')
LOL_ROOT=location('lol_root','data/our485')
HUMAN_ROOTS={ds:location(ds,'data/'+name) for ds,name in [('squarelol','SQUARE-LOL'),('lieq','LIEQ'),('rlie','RLIE')]}
THIRD_PARTY={ds:location(ds+'_source','third_party/'+ds) for ds in ['bpri','bmpri','stairiqa']}
STAIR_WEIGHTS=location('stairiqa_weights','third_party/stairiqa/ResNet_staircase_50-EXP1-Koniq10k.pkl')
MATLAB=CONFIG.get('matlab','matlab')
