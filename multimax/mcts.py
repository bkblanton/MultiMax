import random
from dataclasses import dataclass, replace
from math import sqrt, log
from typing import *

from evaluation import BaseEvaluator, BaseEvaluation, Branch
from game import Game, ActionProtocol


@dataclass
class MCTSEvaluation(BaseEvaluation):
    """
    Monte Carlo Tree Search (MCTS) evaluation of a game state
    Attributes:
        wins: Number of simulations where the score was > 0
        losses: Number of simulations where the score was < 0
    """
    wins: float = 0
    losses: float = 0

    @property
    def total(self):
        """
        :return: Total number of simulations. Equal to wins + losses.
        """
        return self.wins + self.losses

    @property
    def value(self):
        """
        :return: Proportion of wins to total number of simulations
        """
        return self.wins / self.total

    def __itruediv__(self, other: float) -> Self:
        """
        Divide the wins and losses by a value, swapping them if negative.
        :return: self
        """
        self.wins /= abs(other)
        self.losses /= abs(other)

        if other < 0:
            self.wins, self.losses = self.losses, self.wins

        return self


@dataclass
class MCTSConfig:
    """
    Configuration for a MCTSEvaluator
    Attributes:
        exploration_factor: Weight assigned to the exploration term
            in the Upper Confidence Bound 1 for trees (UCT) formula
        max_depth: Maximum simulation depth
    """
    exploration_factor = sqrt(2)
    max_depth = 100


class MCTSEvaluator(BaseEvaluator[MCTSEvaluation]):
    """
    Monte Carlo Tree Search (MCTS) evaluator using Upper Confidence Trees (UCT)
    """

    def __init__(self, game: Game, config: MCTSConfig = MCTSConfig()):
        super().__init__(game)
        self.config = config

    def iter_evaluations(self, *sequence: ActionProtocol) -> Iterator[MCTSEvaluation]:
        """
        Iterate over evaluations of a sequence of actions, running a simulation for each iteration.
        :param sequence: Sequence of actions
        :return: MCTSEvaluation iterator
        """
        self._set_state()

        try:
            while True:
                yield self._evaluate_sequence(sequence, self.config.max_depth)[0]
        except TimeoutError:
            pass

    def _evaluate_sequence(self, sequence: Sequence[Union[ActionProtocol, Branch]], depth: int
                           ) -> tuple[MCTSEvaluation, MCTSEvaluation]:
        """
        Evaluate a sequence of actions or branches from the current state.
        :param sequence: Sequence of actions or branches
        :param depth: Maximum depth of the simulation
        :return: Tuple of current state's evaluation and leaf node's evaluation
        """

        # If the sequence is empty, evaluate from the current state.
        if not sequence:
            return self._evaluate_state(None, depth)

        # Get the action and cached transposition.
        if isinstance(sequence[0], Branch):
            action = sequence[0].action
            transposition = sequence[0].transposition
        else:
            action = sequence[0]
            transposition = None

        # Apply the action, get the evaluation from the resulting state, and finally revert the action.
        self.game.apply(action)

        try:
            if len(sequence) > 1:
                evaluation, leaf_evaluation = self._evaluate_sequence(sequence[1:], depth)
            else:
                evaluation, leaf_evaluation = self._evaluate_state(transposition, depth)
        finally:
            self.game.revert(action)

        # Adjust the evaluations for the action's alignment.
        evaluation /= action.alignment
        leaf_evaluation /= action.alignment

        return evaluation, leaf_evaluation

    def _evaluate_state(self, state: Optional[int], depth: int) -> tuple[MCTSEvaluation, MCTSEvaluation]:
        """
        Evaluate the current game state using Monte Carlo Tree Search
        :param state: Optional current game state
        :param depth: Maximum depth of the simulation
        :return: Tuple of current state's evaluation and leaf node's evaluation
        """
        if state is None:
            state = self.game.get_state()

        evaluation = self.evaluation_cache.get(state)

        # Expand and simulate from the current state if the state is un-cached.
        if not evaluation:
            leaf_evaluation = self._simulate(depth)
            evaluation = replace(leaf_evaluation)
            evaluation.state = state
            self.evaluation_cache[state] = evaluation
            return evaluation, leaf_evaluation

        # Get and cache the possible branches.
        branches = self._get_branches(state)
        self.transposition_cache[state] = branches

        # Return the evaluation if at a leaf node.
        if not branches or depth <= 0:
            score = self.game.get_score()
            leaf_evaluation = MCTSEvaluation(
                state=state,
                is_final=True,
                wins=1 if score > 0 else 0.5 if score == 0 else 0,
                losses=1 if score < 0 else 0.5 if score == 0 else 0,
            )
            evaluation.wins += leaf_evaluation.wins
            evaluation.losses += leaf_evaluation.losses
            self.evaluation_cache[state] = evaluation
            return evaluation, leaf_evaluation

        # Select and evaluate the branch with the highest UCT.
        ucts = [self._get_uct(branch.evaluation, evaluation) for branch in branches]
        branch = branches[max(range(len(branches)), key=ucts.__getitem__)]
        branch.evaluation, leaf_evaluation = self._evaluate_sequence([branch], depth - 1)
        branch.transposition = branch.evaluation.state

        # Back-propagate the simulation result.
        evaluation.wins += leaf_evaluation.wins
        evaluation.losses += leaf_evaluation.losses

        # Cache the state and branches.
        self.evaluation_cache[state] = evaluation
        self.transposition_cache[state] = branches

        return evaluation, leaf_evaluation

    def _simulate(self, depth: int) -> MCTSEvaluation:
        """
        Simulate a random playout up to a maximum depth from the current state
        :param depth: Maximum depth of the simulation
        """

        # Get the possible branches.
        branches = self._get_branches()

        # Return the evaluation if at a leaf node or maximum depth.
        if not branches or depth <= 0:
            score = self.game.get_score()
            return MCTSEvaluation(
                state=0,
                is_final=not branches,
                wins=1 if score > 0 else 0.5 if score == 0 else 0,
                losses=1 if score < 0 else 0.5 if score == 0 else 0,
            )

        # Select and simulate a random branch.
        return self._simulate_sequence([random.choice(branches)], depth - 1)

    def _simulate_sequence(self, sequence: Sequence[Branch], depth: int) -> MCTSEvaluation:
        """
        Simulate a random playout up to a maximum depth starting with the given sequence of branches
        :param sequence: Sequence of branches to prefix the simulation
        :param depth: Maximum depth of the simulation
        """

        # If the sequence is empty, simulate from the current state.
        if not sequence:
            return self._simulate(depth)

        # Apply an action, get the evaluation by simulating from the resulting state, and finally revert the action.
        self.game.apply(sequence[0].action)

        try:
            evaluation = self._simulate_sequence(sequence[1:], depth)
        finally:
            self.game.revert(sequence[0].action)

        # Adjust the wins and losses for the action's alignment.
        if sequence[0].action.alignment < 0:
            evaluation.wins, evaluation.losses = evaluation.losses, evaluation.wins

        evaluation.is_final = False
        return evaluation

    def _get_uct(self, child: Optional[MCTSEvaluation], parent: MCTSEvaluation) -> float:
        """
        Get the Upper Confidence Bound 1 for trees (UCT) of an evaluation and its parent evaluation
        :param child: Optional child evaluation
        :parm parent: Parent evaluation
        """
        if child is None:
            return 0.5 + self.config.exploration_factor * sqrt(log(parent.total))
        else:
            return child.value + self.config.exploration_factor * sqrt(log(parent.total) / child.total)
