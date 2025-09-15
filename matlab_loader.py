import matlab.engine
from paths import matlab_dir

def activate_matlab():
    eng = matlab.engine.start_matlab()
    eng.addpath(matlab_dir)
    return eng