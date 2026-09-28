"""Short-range bishops: slide diagonally but at most two squares per ray (blocking as usual).

Queens, rooks, knights, pawns, and kings use standard chess rules.
"""

from __future__ import annotations

import chess
from chess import BB_DIAG_ATTACKS, BB_DIAG_MASKS, BB_FILE_ATTACKS, BB_FILE_MASKS
from chess import BB_KING_ATTACKS, BB_KNIGHT_ATTACKS, BB_PAWN_ATTACKS, BB_RANK_ATTACKS
from chess import BB_RANK_MASKS, BB_SQUARES, square_file, square_rank
from chess import between, ray, scan_reversed


def old_bishop_attack_mask(from_square: chess.Square, occupied: chess.Bitboard) -> chess.Bitboard:
    r0 = square_rank(from_square)
    f0 = square_file(from_square)
    out: chess.Bitboard = 0
    for dr, df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        r, f = r0, f0
        for _ in range(2):
            r += dr
            f += df
            if not (0 <= r < 8 and 0 <= f < 8):
                break
            sq = chess.square(f, r)
            bb = BB_SQUARES[sq]
            out |= bb
            if occupied & bb:
                break
    return out


class OldBishopBoard(chess.Board):
    halluworld_variant = "old_bishop"

    def attacks_mask(self, square: chess.Square) -> chess.Bitboard:
        bb_square = BB_SQUARES[square]

        if bb_square & self.pawns:
            color = bool(bb_square & self.occupied_co[chess.WHITE])
            return BB_PAWN_ATTACKS[color][square]
        if bb_square & self.knights:
            return BB_KNIGHT_ATTACKS[square]
        if bb_square & self.kings:
            return BB_KING_ATTACKS[square]
        if bb_square & self.bishops and not (bb_square & self.queens):
            return old_bishop_attack_mask(square, self.occupied)

        attacks: chess.Bitboard = 0
        if bb_square & self.bishops or bb_square & self.queens:
            attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied]
        if bb_square & self.rooks or bb_square & self.queens:
            attacks |= (
                BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]
                | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied]
            )
        return attacks

    def attackers_mask(
        self,
        color: chess.Color,
        square: chess.Square,
        occupied: chess.Bitboard | None = None,
    ) -> chess.Bitboard:
        occupied = self.occupied if occupied is None else occupied

        rank_pieces = BB_RANK_MASKS[square] & occupied
        file_pieces = BB_FILE_MASKS[square] & occupied
        diag_pieces = BB_DIAG_MASKS[square] & occupied

        queens_and_rooks = self.queens | self.rooks

        attackers = (
            (BB_KING_ATTACKS[square] & self.kings)
            | (BB_KNIGHT_ATTACKS[square] & self.knights)
            | (BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks)
            | (BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks)
            | (BB_DIAG_ATTACKS[square][diag_pieces] & self.queens)
            | (BB_PAWN_ATTACKS[not color][square] & self.pawns)
        )

        enemy_bishops = self.bishops & ~self.queens & self.occupied_co[color]
        for bsq in scan_reversed(enemy_bishops):
            if BB_SQUARES[square] & old_bishop_attack_mask(bsq, occupied):
                attackers |= BB_SQUARES[bsq]

        return attackers & self.occupied_co[color]

    def _slider_blockers(self, king: chess.Square) -> chess.Bitboard:
        rooks_and_queens = self.rooks | self.queens

        snipers = (BB_RANK_ATTACKS[king][0] & rooks_and_queens) | (
            BB_FILE_ATTACKS[king][0] & rooks_and_queens
        )
        snipers |= BB_DIAG_ATTACKS[king][0] & self.queens

        enemy_bishops = self.bishops & ~self.queens & self.occupied_co[not self.turn]
        for bsq in scan_reversed(BB_DIAG_ATTACKS[king][0] & enemy_bishops):
            if BB_SQUARES[king] & old_bishop_attack_mask(bsq, self.occupied):
                snipers |= BB_SQUARES[bsq]

        blockers: chess.Bitboard = 0
        for sniper in scan_reversed(snipers & self.occupied_co[not self.turn]):
            b = between(sniper, king) & self.occupied
            if b and BB_SQUARES[chess.msb(b)] == b:
                blockers |= b

        return blockers & self.occupied_co[self.turn]

    def pin_mask(self, color: chess.Color, square: chess.Square) -> chess.Bitboard:
        king = self.king(color)
        if king is None:
            return chess.BB_ALL

        square_mask = BB_SQUARES[square]

        for attacks, sliders in (
            (BB_FILE_ATTACKS, self.rooks | self.queens),
            (BB_RANK_ATTACKS, self.rooks | self.queens),
            (BB_DIAG_ATTACKS, self.queens | self.bishops),
        ):
            rays = attacks[king][0]
            if not (rays & square_mask):
                continue
            snipers = rays & sliders & self.occupied_co[not color]
            for sniper in scan_reversed(snipers):
                if not (BB_SQUARES[king] & self.attacks_mask(sniper)):
                    continue
                if between(sniper, king) & (self.occupied | square_mask) == square_mask:
                    return ray(king, sniper)
            break

        return chess.BB_ALL

    def _generate_evasions(
        self,
        king: chess.Square,
        checkers: chess.Bitboard,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ):
        sliders = checkers & (self.bishops | self.rooks | self.queens)

        attacked: chess.Bitboard = 0
        for checker in scan_reversed(sliders):
            if BB_SQUARES[checker] & self.bishops & ~self.queens:
                attacked |= self.attacks_mask(checker) & ~BB_SQUARES[checker]
            else:
                attacked |= ray(king, checker) & ~BB_SQUARES[checker]

        if BB_SQUARES[king] & from_mask:
            for to_square in scan_reversed(
                BB_KING_ATTACKS[king] & ~self.occupied_co[self.turn] & ~attacked & to_mask
            ):
                yield chess.Move(king, to_square)

        checker = chess.msb(checkers)
        if BB_SQUARES[checker] == checkers:
            target = between(king, checker) | checkers

            yield from self.generate_pseudo_legal_moves(~self.kings & from_mask, target & to_mask)

            if self.ep_square and not BB_SQUARES[self.ep_square] & target:
                last_double = self.ep_square + (-8 if self.turn == chess.BLACK else 8)
                if last_double == checker:
                    yield from self.generate_pseudo_legal_ep(from_mask, to_mask)
