from typing import *


class ActionProtocol(Protocol):
    """
    Game action
    Attributes:
        alignment:
            Number representing how the next player is aligned with the current player.
            1 if they are the same player or are on the same team. -1 if they are opponents.
        force:
            Heuristic representing how forceful an action is.
            Non-zero forceful actions are always evaluated to quiescence.
    """
    alignment: float
    force: float

    def __eq__(self, other: "ActionProtocol") -> bool:
        pass

    def __hash__(self) -> int:
        pass


Action = TypeVar("Action", bound=ActionProtocol)


class Game(Protocol):
    def get_state(self) -> int:
        """
        :return: Unique id of the current game state
        """
        pass

    def get_score(self) -> float:
        """
        :return: Score the current player is maximizing
        """
        pass

    def get_actions(self) -> Iterable[Action]:
        """
        :return: Iterable of possible actions
        """
        pass

    def apply(self, action: Action) -> None:
        """
        Apply an action.
        :param action: Action to apply
        """
        pass

    def revert(self, action: Action) -> None:
        """
        Revert an action.
        :param action: Action to revert
        """
        pass
