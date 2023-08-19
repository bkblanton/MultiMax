# MultiMax
## State-of-the-Art Game Tree Search for Arbitrary Games
MultiMax is a project which demonstrates general state-of-the-art game tree search algorithms that can be easily implemented for arbitrary games in Python.

## Game and Action Interfaces

- Implement the simple Game and Action interfaces to make use of MultiMax's evaluators.
- See [examples](multimax/examples) for an example chess implementation.

## Negamax Evaluator

- Fail-soft alpha-beta pruning
- Iterative deepening
- Principal variation search
- Quiescence search

## MCTS Evaluator

- Monte Carlo Tree Search (MCTS)
- Upper Confidence Bound 1 applied to trees (UCT)

## Caching

- Two-tier evaluation and transposition caching
