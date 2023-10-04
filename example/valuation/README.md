# Valuation Functions
Each relation (like `closeby`) has a corresponding **valuation function** which maps the game state to a probability that the relation is true.

Each environment needs to define its own valuation functions. For example, the valuation functions for Freeway can be found in `freeway.py`. There, each valuation function is defined as a simple Python function. The function's name must match the name of the corresponding relation.