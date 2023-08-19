from collections import deque
from dataclasses import dataclass, replace
from itertools import count
from math import inf
from typing import *

from evaluation import BaseEvaluator, Branch, BaseEvaluation, EvaluationCache
from game import ActionProtocol


@dataclass
class NegamaxEvaluation(BaseEvaluation):
    """
    Negamax evaluation of a game state
    Attributes:
        value: Value of the evaluation
        depth: Depth of the evaluation
    """
    value: float
    depth: int

    def __itruediv__(self, other: float) -> Self:
        """
        Divide the value by another value
        :return: self
        """
        self.value /= other
        return self


@dataclass
class SequenceEvaluation:
    """
    Evaluation of a sequence of actions
    Attributes:
        sequence: Sequence of actions
        evaluation: Evaluation of the sequence of actions
    """
    sequence: Deque[ActionProtocol]
    evaluation: NegamaxEvaluation


@dataclass
class NegamaxAnalysis:
    """
    Evaluation along with sequence evaluations
    Attributes:
        evaluation: Evaluation of the game state
        variations: List of evaluations of sequences of actions from the game state
    """
    evaluation: NegamaxEvaluation
    variations: list[SequenceEvaluation]


class NegamaxEvaluator(BaseEvaluator[NegamaxEvaluation]):
    """
    Negamax evaluator with:
    - Alpha-beta pruning
    - Iterative deepening
    - Principal variation search
    - Quiescence search
    """
    evaluation_cache: EvaluationCache[NegamaxEvaluation]

    def iter_evaluations(self, *sequence: ActionProtocol) -> Iterator[NegamaxAnalysis]:
        """
        Iterate over evaluations of a sequence of actions at increasing depth.
        :param sequence: Sequence of actions to evaluate
        :return: Analysis iterator
        """
        self._set_state()

        try:
            for depth in count():
                yield self._evaluate_sequence(sequence, depth)
        except TimeoutError:
            pass

    def _evaluate_sequence(self,
                           sequence: Sequence[Union[ActionProtocol, Branch]],
                           depth: int,
                           alpha: float = -inf,
                           beta: float = inf) -> NegamaxAnalysis:
        """
        Evaluate a sequence of actions or branches from the current state.
        :param sequence: Sequence of actions or branches
        :param depth: Depth of the evaluation
        :param alpha: Minimum score assured to the maximizing player
        :param beta: Maximum score assured to the minimizing player
        :return: Analysis of the sequence
        """

        # If the sequence is empty, evaluate from the current state.
        if not sequence:
            return self._evaluate_state(None, depth, alpha, beta)

        # Get the action and cached transposition.
        if isinstance(sequence[0], Branch):
            action = sequence[0].action
            transposition = sequence[0].transposition
        else:
            action = sequence[0]
            transposition = None

        # Adjust alpha and beta for the action's alignment.
        alpha *= action.alignment
        beta *= action.alignment

        if action.alignment < 0:
            alpha, beta = beta, alpha

        # Apply the action, get the evaluation from the resulting state, and finally revert the action.
        self.game.apply(action)

        try:
            if len(sequence) > 1:
                analysis = self._evaluate_sequence(sequence[1:], depth, alpha, beta)
            else:
                analysis = self._evaluate_state(transposition, depth, alpha, beta)
        finally:
            self.game.revert(action)

        # Adjust the analysis for the action's alignment.
        analysis.evaluation /= action.alignment

        # Append the action to all the variations and adjust them for action's alignment.
        for sequence_evaluation in analysis.variations:
            sequence_evaluation.sequence.appendleft(action)
            sequence_evaluation.evaluation /= action.alignment

        return analysis

    def _evaluate_state(self,
                        state: Optional[int],
                        depth: int,
                        alpha: float = -inf,
                        beta: float = inf) -> NegamaxAnalysis:
        """
        Evaluate the current game state using Principal Variation Search with quiescence.
        :param state: Optional current game state
        :param depth: Depth of the evaluation
        :param alpha: Minimum score assured to the maximizing player
        :param beta: Maximum score assured to the minimizing player
        :return: Analysis of the current game state
        """

        # Get the current state.
        if state is None:
            state = self.game.get_state()

            if not isinstance(state, int):
                raise TypeError(f"State must be int, not {type(state)}")

        # Return the cached evaluation if it has sufficient depth.
        cached_evaluation = self.evaluation_cache.get(state)

        if cached_evaluation and (cached_evaluation.is_final or cached_evaluation.depth >= depth):
            return NegamaxAnalysis(cached_evaluation, [SequenceEvaluation(deque(), replace(cached_evaluation))])

        branches = self._get_branches(state)
        self.transposition_cache[state] = branches

        if branches and depth > 0:
            # If there are possible actions and the depth is greater than 0, start by assuming the worst.
            analysis = NegamaxAnalysis(NegamaxEvaluation(
                state=state,
                is_final=True,
                value=-inf,
                depth=depth
            ), [])

            # All actions are candidates for evaluation.
            evaluation_branches = branches
        else:
            # If there are no actions or the depth is 0, start by assuming current state's score.
            analysis = NegamaxAnalysis(NegamaxEvaluation(
                state=state,
                is_final=not branches,
                value=self.game.get_score(),
                depth=depth
            ), [])

            # Allow the player to stand pat.
            analysis.variations.append(SequenceEvaluation(deque(), replace(analysis.evaluation)))
            alpha = max(alpha, analysis.evaluation.value)

            # Cache the evaluation and return immediately if the evaluation is final or if a beta-cutoff occurred.
            if analysis.evaluation.is_final or alpha >= beta:
                self.evaluation_cache[state] = analysis.evaluation
                return analysis

            # Only evaluate forceful actions during quiescence search.
            evaluation_branches = [branch for branch in branches if getattr(branch.action, "force", 0) > 0]

        # Sort the actions by their cached values and forces to achieve the earliest possible beta-cutoff.
        evaluation_branches.sort(
            key=lambda branch: (
                -branch.evaluation.value if branch.evaluation else inf,
                -branch.action.force if hasattr(branch.action, "force") else inf,
            )
        )

        for i, branch in enumerate(evaluation_branches):
            def pvs(a=alpha, b=beta) -> NegamaxAnalysis:
                """Search the action with a window of (a, b)."""
                return self._evaluate_sequence([branch], max(0, depth - 1), a, b)

            if i == 0:
                # Fully search the principal variation.
                branch_analysis = pvs()
            else:
                # Search with a null window around alpha to test if the new move can do better.
                branch_analysis = pvs(b=alpha + 1)

                # If so, do a full re-search.
                if alpha < branch_analysis.evaluation.value < beta:
                    branch_analysis = pvs()

            # Update the analysis.
            analysis.variations.append(branch_analysis.variations[0])
            branch.evaluation = branch_analysis.evaluation
            branch.transposition = branch.evaluation.state
            analysis.evaluation.value = max(analysis.evaluation.value, branch.evaluation.value)
            analysis.evaluation.is_final = analysis.evaluation.is_final and branch.evaluation.is_final
            alpha = max(alpha, analysis.evaluation.value)

            # Stop searching if a beta-cutoff occurs.
            if alpha >= beta:
                break

        # Sort the variations by evaluation value.
        analysis.variations.sort(key=lambda sequence_evaluation: -sequence_evaluation.evaluation.value)

        # Cache the evaluation and branches.
        self.evaluation_cache[state] = analysis.evaluation
        self.transposition_cache[state] = branches
        return analysis
