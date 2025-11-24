import vapoursynth as vs

core = vs.core
try:
    print(f"vs-dovi available: {hasattr(core, 'dovi')}")
except Exception as e:
    print(f"Error checking vs-dovi: {e}")
