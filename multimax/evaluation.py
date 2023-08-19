import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from sys import getsizeof
from types import MappingProxyType
from typing import *

from game import Game, ActionProtocol


@dataclass
class BaseEvaluation(ABC):
    """
    Base evaluation class shared by evaluators.
    Attributes:
        state: Evaluated game state
        is_final: Whether the evaluation is final
    """
    state: int
    is_final: bool

    @abstractmethod
    def __itruediv__(self, other: float) -> Self:
        pass


Evaluation = TypeVar("Evaluation", bound=BaseEvaluation)


@dataclass
class Branch(Generic[Evaluation]):
    """
    An branch from a state through a transposition to a relative evaluation
    Attributes:
        action: Action in a game state
        transposition: Game state after applying the action
        evaluation: Evaluation relative to the original game state
    """
    action: ActionProtocol
    transposition: Optional[int] = None
    evaluation: Optional[Evaluation] = None


GameCacheValue = TypeVar("GameCacheValue")


class GameCache(Generic[GameCacheValue]):
    """
    Two-tier game cache
    Attributes:
        _capacity: Capacity of the cache in bytes
        _primary_cache: Primary cache that fills up first
        _secondary_cache: Least recently used (LRU) cache that fills up second
        _size: Current size of the cache in bytes
        _hits: Number of cache hits
        _misses: Number of cache misses
    """

    def __init__(self, capacity=2 ** 25):
        self._capacity = capacity
        self._primary_cache = dict[int, GameCacheValue]()
        self._secondary_cache = OrderedDict[int, GameCacheValue]()
        self._size = self._hits = self._misses = 0

    @property
    def hit_rate(self) -> float:
        """
        Ratio of cache hits to total cache accesses
        """
        return self._hits / (self._hits + self._misses)

    def __contains__(self, state: int) -> bool:
        """
        Check if the cache contains a game state.
        :param state: Game state to check
        :return: Bool representing whether the state is present in the cache
        """
        return state in self._primary_cache or state in self._secondary_cache

    def __len__(self) -> int:
        """
        Get the number of values in the cache.
        :return: Number of values in the cache
        """
        return len(self._primary_cache) + len(self._secondary_cache)

    def __getitem__(self, state: int) -> GameCacheValue:
        """
        Retrieve the cached value of the game state from the cache.
        :param state: Cached game state to retrieve
        :return: Cached value
        """
        primary = self._primary_cache.get(state)
        secondary = self._secondary_cache.get(state)

        if secondary is not None:
            self._secondary_cache.move_to_end(state)
        elif primary is None:
            raise KeyError(state)

        self._hits += 1
        return primary or secondary

    def get(self, state: int) -> Optional[GameCacheValue]:
        """
        Get the value of the game state from the cache or None if not present.
        :param state: Cached game state to retrieve
        :return: Cached value or None
        """
        if state in self:
            return self[state]
        else:
            self._misses += 1
            return None

    def clear(self):
        """
        Clear the cache.
        """
        self._primary_cache.clear()
        self._secondary_cache.clear()
        self._size = self._hits = self._misses = 0


class EvaluationCache(GameCache[Evaluation]):
    """
    Cache of game states to evaluations
    """

    def __getitem__(self, state: int) -> Evaluation:
        """
        Get a copy of the cached evaluation.
        :param state: Cached game state to retrieve
        :return: Cached evaluation of the game state
        """
        return replace(super().__getitem__(state))

    def __setitem__(self, state: int, evaluation: Evaluation) -> None:
        """
        Cache a copy of the game state's evaluation.
        :param state: Game state to cache
        :param evaluation: Game state's evaluation to cache
        """
        evaluation = replace(evaluation)
        primary = self._primary_cache.get(state)
        secondary = self._secondary_cache.get(state)

        # Update the evaluation if it already exists.
        if primary:
            self._primary_cache[state] = evaluation
            self._size += getsizeof(evaluation) - getsizeof(primary)
        elif secondary:
            self._secondary_cache[state] = evaluation
            self._secondary_cache.move_to_end(state)
            self._size += getsizeof(evaluation) - getsizeof(secondary)
        else:
            # Otherwise, add the evaluation.
            if self._size < self._capacity / 2:
                self._primary_cache[state] = evaluation
            else:
                self._secondary_cache[state] = evaluation

            self._size += getsizeof(state) + getsizeof(evaluation)

        # Make room by evicting LRU items.
        while self._size >= self._capacity:
            evicted_state, evicted_evaluation = self._secondary_cache.popitem(last=False)
            self._size -= getsizeof(evicted_state) + getsizeof(evicted_evaluation)


class TranspositionCache(GameCache[Mapping[ActionProtocol, Optional[int]]]):
    """
    Cache of game states to actions to game state transpositions
    """
    _primary_cache: dict[int, OrderedDict[ActionProtocol, Optional[int]]]
    _secondary_cache: OrderedDict[int, OrderedDict[ActionProtocol, Optional[int]]]

    def __getitem__(self, state: int) -> Mapping[ActionProtocol, Optional[int]]:
        """
        Get a read-only mapping of a game state's actions to game state transpositions.
        :param state: Cached game state
        :return: Cached read-only mapping of the game state's actions to game state transpositions
        """
        return MappingProxyType(super().__getitem__(state))

    def put(self, state: int, action: ActionProtocol, transposition: Optional[int]) -> None:
        """
        Cache a game state to action to game state transposition.
        :param state: Game state to cache
        :param action: Action to cache
        :param transposition: Game state transposition to cache
        """
        primary = self._primary_cache.get(state)
        secondary = self._secondary_cache.get(state)

        # Create the mapping if it doesn't exist.
        if primary is None and secondary is None:
            if self._size < self._capacity / 2:
                primary = self._primary_cache[state] = OrderedDict[ActionProtocol, int]()
            else:
                secondary = self._secondary_cache[state] = OrderedDict[ActionProtocol, int]()

            self._size += getsizeof(state)

        # Add the transposition to the mapping.
        if primary is not None:
            if action in primary:
                self._size -= getsizeof(primary[action])
            else:
                self._size += getsizeof(action)

            primary[action] = transposition
        else:
            if action in secondary:
                self._size -= getsizeof(secondary[action])
            else:
                self._size += getsizeof(action)

            secondary[action] = transposition
            self._secondary_cache.move_to_end(state)

        self._size += getsizeof(transposition)

        # Make room by evicting LRU items.
        while self._size >= self._capacity:
            evicted_key, evicted_dict = self._secondary_cache.popitem(last=False)
            self._size -= getsizeof(evicted_key)
            for evicted_action, evicted_transposition in evicted_dict.items():
                self._size -= getsizeof(evicted_action) + getsizeof(evicted_transposition)

    def __setitem__(self, state: int, branches: Iterable[Branch]) -> None:
        """
        Transactionally put the branches into the cache.
        :param state: Game state to cache
        :param branches: Iterable of branches to cache
        """
        try:
            for branch in branches:
                self.put(state, branch.action, branch.transposition)
        except (KeyboardInterrupt, TimeoutError):
            for branch in branches:
                self.put(state, branch.action, branch.transposition)
            raise


class BaseEvaluator(Generic[Evaluation], ABC):
    """
    Base evaluator class
    Attributes:
        game: Game
        evaluation_cache: Evaluation cache
        transposition_cache: Transposition cache
        _state: Last evaluated state
    """
    def __init__(self, game: Game):
        self.game = game
        self.evaluation_cache = EvaluationCache()
        self.transposition_cache = TranspositionCache()
        self._state: Optional[int] = None

    @abstractmethod
    def iter_evaluations(self, *sequence: ActionProtocol) -> Iterator[Any]:
        pass

    def _set_state(self):
        """
        Set the internal state to the current game state. Clear the caches if the state changed.
        """
        current_state = self.game.get_state()

        if current_state != self._state:
            self.evaluation_cache.clear()
            self.transposition_cache.clear()
            self._state = current_state

    def _get_branches(self, state: Optional[int] = None) -> list[Branch[Evaluation]]:
        """
        Get the branches from the current game state.
        :param state: Current game state
        :return: Edges
        """
        if not (state is None or (cached_branches := self._get_cached_branches(state)) is None):
            return cached_branches

        branches = []

        for action in self.game.get_actions():
            if action.alignment == 0:
                raise ValueError("Action alignment cannot be 0")

            branches.append(Branch(action))

        return branches

    def _get_cached_branches(self, state: int) -> Optional[list[Branch[Evaluation]]]:
        """
        Get the cached branches from the given game state.
        :param state: Game state
        :return: Cached branches
        """
        cached_branches = self.transposition_cache.get(state)

        if cached_branches is None:
            return None

        branches = []

        for action, transposition in cached_branches.items():
            if transposition is None:
                branches.append(Branch(action))
                continue
            evaluation = self.evaluation_cache.get(transposition)
            evaluation /= action.alignment
            branches.append(Branch(action, transposition, evaluation))

        return branches


def start_timer(seconds: int) -> None:
    """
    Start a timer that stops evaluation after a given amount of time.
    :param seconds: Seconds to ponder for. 0 to ponder forever.
    """
    if seconds <= 0:
        raise ValueError("Seconds must be > 0")

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(seconds)


def _alarm_handler(signum, frame):
    """
    Handler for the alarm signal received when the timer finishes
    """
    raise TimeoutError
