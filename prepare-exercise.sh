# Run black on .py files
# black solution.py

jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update solution.py
jupytext --to ipynb --update-metadata '{"jupytext":{"cell_metadata_filter":"all"}}' --update solution.py --output exercise.ipynb

jupyter nbconvert solution.ipynb --ClearOutputPreprocessor.enabled=True --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags task --to notebook --output solution.ipynb
jupyter nbconvert exercise.ipynb --ClearOutputPreprocessor.enabled=True --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags solution --to notebook --output exercise.ipynb
# Convert the cleaned-up exercise.ipynb back to a Python script
jupytext --to py:percent exercise.ipynb