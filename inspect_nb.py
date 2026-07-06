import nbformat

with open('iris_logistic_regression.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for i, cell in enumerate(nb.cells):
    source_preview = cell.source[:60].replace('\n', ' ')
    print(f"Cell {i} [{cell.cell_type}]: {source_preview}...")
