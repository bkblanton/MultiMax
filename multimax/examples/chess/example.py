import cProfile
from dataclasses import dataclass
from math import inf
from pprint import pprint
from typing import Optional

import chess

from game import Game, ActionProtocol
from mcts import MCTSEvaluator
from negamax import NegamaxEvaluator


@dataclass(frozen=True)
class Move(ActionProtocol):
    move: chess.Move
    force: float
    alignment = -1


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


class Chess(Game):

    def __init__(self, fen: Optional[str] = chess.STARTING_FEN):
        self.board = chess.Board(fen)

    def get_state(self):
        return hash(self.board.fen())

    def get_score(self):
        outcome = self.board.outcome()

        if outcome:
            if outcome.termination == chess.Termination.CHECKMATE:
                return -inf
            else:
                return 0

        score = 0

        for piece, value in PIECE_VALUES.items():
            for _ in self.board.pieces(piece, self.board.turn):
                score += value
            for _ in self.board.pieces(piece, not self.board.turn):
                score -= value

        return score

    def get_actions(self):
        for move in self.board.legal_moves:
            if self.board.gives_check(move):
                force = inf
            elif self.board.is_capture(move):
                if self.board.is_en_passant(move):
                    force = PIECE_VALUES[chess.PAWN]
                else:
                    force = PIECE_VALUES[self.board.piece_at(move.to_square).piece_type]
            else:
                force = 0

            yield Move(move, force)

    def apply(self, move: Move):
        self.board.push(move.move)

    def revert(self, _):
        self.board.pop()


if __name__ == "__main__":
    Evaluator = NegamaxEvaluator
    # Evaluator = MCTSEvaluator
    evaluator = Evaluator(Chess())
    # evaluator = Evaluator(Chess("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"))  # scholar's mate
    # evaluator = Evaluator(Chess("r1bqkb1r/ppp2ppp/2n2n2/3Pp1N1/2B5/8/PPPP1PPP/RNBQK2R b KQkq - 0 5"))  # knight attack
    with cProfile.Profile() as pr:
        try:
            for depth, evaluation in enumerate(evaluator.iter_evaluations()):
                print(f"{depth}:")
                pprint(evaluation)
                print()
        except KeyboardInterrupt:
            pass
        print()
        pr.print_stats(sort="time")