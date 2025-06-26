import copy
import numpy as np

class Move:
    def __init__(self, from_x, from_y, to_x, to_y, promote=False, piece_to_drop=None):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        self.promote = promote
        self.piece_to_drop = piece_to_drop

    def __str__(self):
        if self.piece_to_drop:
            return f"Drop {self.piece_to_drop.name()} at ({self.to_x}, {self.to_y})"
        else:
            return f"Move from ({self.from_x}, {self.from_y}) to ({self.to_x}, {self.to_y}) {'(promote)' if self.promote else ''}"


class Player:
    def __init__(self, name, owner_id):
        self.name = name
        self.owner = owner_id
        self.captured_pieces = []


class Piece:
    def __init__(self, owner, promoted=False):
        self.owner = owner
        self.promoted = promoted

    def legal_moves(self, x, y, board):
        raise NotImplementedError("This method should be implemented by subclasses")

    @property
    def name(self):
        raise NotImplementedError("各駒で name プロパティを定義してください")

    def short_name(self):
        return self.name[-1]

    def clone(self):
        return self.__class__(self.owner, self.promoted)

    def _get_moves_in_directions(self, x, y, directions, distance, board):
        moves = []
        for dx, dy in directions:
            for i in range(1, distance + 1):
                nx, ny = x + dx * i, y + dy * i
                if not (0 <= nx < 5 and 0 <= ny < 5):
                    break

                target = board.get(nx, ny)
                if not target or target.owner != self.owner:
                    moves.append((nx, ny))

                if target:
                    break
        return moves

    def encode_index(self, current_player):
        piece_dict = {
            '歩': 0, '香': 1, '桂': 2, '銀': 3, '金': 4, '角': 5, '飛': 6, '王': 7,
            'と': 8, '成香': 9, '成桂': 10, '成銀': 11, '馬': 12, '竜': 13
        }
        base = piece_dict.get(self.name, -1)
        if base == -1:
            raise ValueError(f"Unknown piece: {self.name}")
        return base if self.owner == current_player else base + 14


class Ohsho(Piece):
    @property
    def name(self):
        return "王"

    def legal_moves(self, x, y, board):
        directions = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
        return self._get_moves_in_directions(x, y, directions, 1, board)


class Hisha(Piece):
    @property
    def name(self):
        return "竜" if self.promoted else "飛"

    def legal_moves(self, x, y, board):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        moves = self._get_moves_in_directions(x, y, directions, 5, board)
        if self.promoted:
            diag_directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
            moves.extend(self._get_moves_in_directions(x, y, diag_directions, 1, board))
        return moves


class Kakugyo(Piece):
    @property
    def name(self):
        return "馬" if self.promoted else "角"

    def legal_moves(self, x, y, board):
        directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        moves = self._get_moves_in_directions(x, y, directions, 5, board)
        if self.promoted:
            straight_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            moves.extend(self._get_moves_in_directions(x, y, straight_directions, 1, board))
        return moves


def _get_kinsho_moves(x, y, owner, board):
    f_dy = -1 if owner == 0 else 1
    directions = [(-1, f_dy), (0, f_dy), (1, f_dy), (-1, 0), (1, 0), (0, -f_dy)]

    moves = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < 5 and 0 <= ny < 5):
            continue
        target = board.get(nx, ny)
        if not target or target.owner != owner:
            moves.append((nx, ny))
    return moves


class Kinsho(Piece):
    @property
    def name(self):
        return "金"

    def legal_moves(self, x, y, board):
        return _get_kinsho_moves(x, y, self.owner, board)


class Ginsho(Piece):
    @property
    def name(self):
        return "金" if self.promoted else "銀"

    def legal_moves(self, x, y, board):
        if self.promoted:
            return _get_kinsho_moves(x, y, self.owner, board)

        f_dy = -1 if self.owner == 0 else 1
        directions = [(-1, f_dy), (0, f_dy), (1, f_dy), (-1, -f_dy), (1, -f_dy)]
        return self._get_moves_in_directions(x, y, directions, 1, board)


class Hohei(Piece):
    @property
    def name(self):
        return "と" if self.promoted else "歩"

    def short_name(self):
        return "と" if self.promoted else "歩"

    def legal_moves(self, x, y, board):
        if self.promoted:
            return _get_kinsho_moves(x, y, self.owner, board)

        f_dy = -1 if self.owner == 0 else 1
        directions = [(0, f_dy)]
        return self._get_moves_in_directions(x, y, directions, 1, board)


class Board:
    def __init__(self):
        self.grid = [[None for _ in range(5)] for _ in range(5)]
        self.setup()

    def setup(self):
        self.grid[0][0] = Hisha(1)
        self.grid[0][1] = Kakugyo(1)
        self.grid[0][2] = Ginsho(1)
        self.grid[0][3] = Kinsho(1)
        self.grid[0][4] = Ohsho(1)
        self.grid[1][4] = Hohei(1)

        self.grid[4][0] = Ohsho(0)
        self.grid[4][1] = Kinsho(0)
        self.grid[4][2] = Ginsho(0)
        self.grid[4][3] = Kakugyo(0)
        self.grid[4][4] = Hisha(0)
        self.grid[3][0] = Hohei(0)

    def apply(self, move: "Move", players):
        if move.piece_to_drop:
            piece = move.piece_to_drop
            self.set(move.to_x, move.to_y, piece)
        else:
            piece = self.get(move.from_x, move.from_y)
            if piece is None:
                return

            target = self.get(move.to_x, move.to_y)
            if target:
                captured_piece = target.clone()
                captured_piece.owner = piece.owner
                captured_piece.promoted = False
                players[piece.owner].captured_pieces.append(captured_piece)

            if move.promote:
                piece.promoted = True

            self.set(move.to_x, move.to_y, piece)
            self.set(move.from_x, move.from_y, None)

    def get(self, x, y):
        return self.grid[y][x]

    def set(self, x, y, piece: "Piece"):
        self.grid[y][x] = piece

    def clone(self):
        return copy.deepcopy(self)

    def display(self, players):
        print(
            "\n後手(Player 1) 持ち駒:",
            " ".join([p.short_name() for p in players[1].captured_pieces]) or "なし",
        )
        print("   0　1　2　3　4")
        print("------------------")
        for y, row in enumerate(self.grid):
            print(f"{y}|", end="")
            for piece in row:
                if piece is None:
                    print(" ・", end="")
                else:
                    print(
                        f"{'v' if piece.owner == 1 else 'm'}{piece.short_name()}",
                        end="",
                    )
            print()
        print("------------------")
        print(
            "先手(Player 0) 持ち駒:",
            " ".join([p.short_name() for p in players[0].captured_pieces]) or "なし",
        )


class Rule:
    @staticmethod
    def is_legal(board, move, player, players):
        if move.piece_to_drop:
            return Rule._is_legal_drop(board, move, player)
        else:
            return Rule._is_legal_move(board, move, player, players)

    @staticmethod
    def _is_legal_move(board, move, player, players):
        piece = board.get(move.from_x, move.from_y)
        if piece is None or piece.owner != player.owner:
            return False

        legal_destinations = piece.legal_moves(move.from_x, move.from_y, board)
        if (move.to_x, move.to_y) not in legal_destinations:
            return False

        if move.promote:
            promotion_zone_start = 0 if player.owner == 0 else 4
            can_promote = (move.from_y == promotion_zone_start) or (
                move.to_y == promotion_zone_start
            )
            if (
                not isinstance(piece, (Ohsho, Kinsho))
                and not piece.promoted
                and can_promote
            ):
                pass
            else:
                return False

        temp_board = board.clone()
        temp_players = copy.deepcopy(players)
        temp_board.apply(move, temp_players)
        if Rule.is_in_check(temp_board, player.owner):
            return False

        return True

    @staticmethod
    def _is_legal_drop(board, move, player):
        piece_to_drop = move.piece_to_drop
        x, y = move.to_x, move.to_y

        if board.get(x, y) is not None:
            return False

        if isinstance(piece_to_drop, Hohei):
            for row in range(5):
                p = board.get(x, row)
                if (
                    p
                    and isinstance(p, Hohei)
                    and not p.promoted
                    and p.owner == player.owner
                ):
                    return False

        temp_board = board.clone()
        temp_players = [copy.deepcopy(player)]
        temp_board.apply(move, temp_players)
        if Rule.is_in_check(temp_board, player.owner):
            return False

        return True

    @staticmethod
    def is_in_check(board, owner):
        king_pos = None
        for y in range(5):
            for x in range(5):
                piece = board.get(x, y)
                if piece and isinstance(piece, Ohsho) and piece.owner == owner:
                    king_pos = (x, y)
                    break
            if king_pos:
                break

        if not king_pos:
            return True

        opponent_owner = 1 - owner
        for y in range(5):
            for x in range(5):
                piece = board.get(x, y)
                if piece and piece.owner == opponent_owner:
                    if king_pos in piece.legal_moves(x, y, board):
                        return True
        return False

    @staticmethod
    def is_checkmate(board, player, players):
        if not Rule.is_in_check(board, player.owner):
            return False

        for y in range(5):
            for x in range(5):
                piece = board.get(x, y)
                if piece and piece.owner == player.owner:
                    for move_to in piece.legal_moves(x, y, board):
                        move = Move(x, y, move_to[0], move_to[1])
                        if Rule.is_legal(board, move, player, players):
                            return False
                        move_promote = Move(x, y, move_to[0], move_to[1], promote=True)
                        if Rule.is_legal(board, move_promote, player, players):
                            return False

        for i, p_drop in enumerate(player.captured_pieces):
            for y_drop in range(5):
                for x_drop in range(5):
                    move = Move(None, None, x_drop, y_drop, piece_to_drop=p_drop)
                    original_captured = player.captured_pieces
                    player.captured_pieces = (
                        original_captured[:i] + original_captured[i + 1 :]
                    )
                    if Rule.is_legal(board, move, player, players):
                        player.captured_pieces = original_captured
                        return False
                    player.captured_pieces = original_captured

        return True


class Game:
    def __init__(self):
        self.board = Board()
        self.players = [Player("先手", 0), Player("後手", 1)]
        self.turn = 0
        self.history = []

    def play_move(self, move: "Move"):
        if Rule.is_legal(self.board, move, self.current_player(), self.players):
            self.board.apply(move, self.players)
            self.history.append(move)
            self.turn = 1 - self.turn
            return True
        else:
            print("--- 不正な手です ---")
            return False

    def current_player(self) -> "Player":
        return self.players[self.turn]

    def run(self):
        while True:
            self.board.display(self.players)
            player = self.current_player()
            print(f"\n{player.name} (Player {player.owner})のターンです。")

            if Rule.is_checkmate(self.board, player, self.players):
                print(f"詰みです！ {self.players[1-self.turn].name}の勝ちです。")
                break

            if Rule.is_in_check(self.board, player.owner):
                print("*** 王手です！ ***")

            try:
                raw_input = input(
                    "指し手を入力してください (例: 'm 2 3 2 2' or 'd 歩 3 3'): "
                )
                parts = raw_input.split()

                move = None
                if parts[0] == "m" and len(parts) >= 5:
                    fx, fy, tx, ty = map(int, parts[1:5])
                    promote = len(parts) > 5 and parts[5] == "p"
                    move = Move(fx, fy, tx, ty, promote)
                elif parts[0] == "d" and len(parts) == 4:
                    piece_name = parts[1]
                    tx, ty = map(int, parts[2:])
                    piece_to_drop = None
                    for i, p in enumerate(player.captured_pieces):
                        if p.short_name() == piece_name:
                            piece_to_drop = player.captured_pieces.pop(i)
                            break
                    if piece_to_drop:
                        move = Move(None, None, tx, ty, piece_to_drop=piece_to_drop)
                    else:
                        print("その持ち駒はありません。")
                        continue
                else:
                    print("入力形式が正しくありません。")
                    print("盤上の駒を動かす: m [元x] [元y] [先x] [先y] (pで成り)")
                    print("持ち駒を打つ: d [駒名] [先x] [先y]")
                    continue

                if not self.play_move(move):
                    if move.piece_to_drop:
                        player.captured_pieces.append(move.piece_to_drop)

            except (ValueError, IndexError):
                print("入力エラーです。数値を正しく入力してください。")
                continue

def get_legal_moves(board, player, players):
    legal_moves = []

    for y in range(5):
        for x in range(5):
            piece = board.get(x, y)
            if piece and piece.owner == player.owner:
                for to_x, to_y in piece.legal_moves(x, y, board):
                    move_p = Move(x, y, to_x, to_y, promote=True)
                    if Rule.is_legal(board, move_p, player, players):
                        legal_moves.append(move_p)
                    move_np = Move(x, y, to_x, to_y, promote=False)
                    if Rule.is_legal(board, move_np, player, players):
                        legal_moves.append(move_np)

    for i, p in enumerate(player.captured_pieces):
        for y in range(5):
            for x in range(5):
                move = Move(None, None, x, y, piece_to_drop=p)
                if Rule.is_legal(board, move, player, players):
                    legal_moves.append(move)

    return legal_moves


def apply_move(board, move, players):
    board.apply(move, players)
    return board

def is_game_over(board, players):
    for i, player in enumerate(players):
        if Rule.is_checkmate(board, player, players):
            return True, 1 - i
    return False, None


def self_play_game(model):
    board = Board()
    players = [Player("先手", 0), Player("後手", 1)]
    history = []
    turn = 0

    while True:
        player = players[turn]
        legal_moves = get_legal_moves(board, player, players)

        if not legal_moves:
            break

        board_tensor = np.zeros((5, 5, 23), dtype=np.float32)

        policy_pred, value_pred = model.predict(np.expand_dims(board_tensor, axis=0), verbose=0)
        move = legal_moves[np.random.randint(len(legal_moves))]

        board = apply_move(board, move, players)
        history.append((board_tensor, move))
        is_over, winner = is_game_over(board, players)
        if is_over:
            return [(b, move, 1 if winner == turn else -1) for b, move in history]

        turn = 1 - turn

def train_model(model, games):

    X = []
    y_policy = []
    y_value = []

    for board_tensor, move, result in games:
        X.append(board_tensor)
        policy_label = np.zeros(4672)
        policy_label[0] = 1
        y_policy.append(policy_label)

        y_value.append(result)

    X = np.array(X)
    y_policy = np.array(y_policy)
    y_value = np.array(y_value)

    model.compile(optimizer="adam",
                  loss={"policy_head": "categorical_crossentropy", "value_head": "mse"},
                  loss_weights={"policy_head": 1.0, "value_head": 1.0})
    
    model.fit(X, {"policy_head": y_policy, "value_head": y_value}, batch_size=32, epochs=1)


if __name__ == "__main__":
    game = Game()
    game.run()
