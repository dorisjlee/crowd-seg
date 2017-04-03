# Create indicator matrix, can either start from scratch or use existing tiles
python -i runIndicatorMat.py
# run Tile experiments (average, median, exhaustive, Local) to get the gamma values
python -i runExperiments.py
# Compute Random Tile Subset Training data
python -i runRandTileTraining.py
# Run PR Curves
python -i runPR.py postprocess
python -i runPR.py T-search
python -i runDualPR.py all
