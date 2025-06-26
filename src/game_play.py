import numpy as np
from shogi_game import get_legal_moves, Move, Rule, Game
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from keras.models import load_model

def board_to_input_tensor(board, current_player):
    input_tensor = np.zeros((5, 5, 23), dtype=np.float32)

    for y in range(5):
        for x in range(5):
            piece = board.get(x, y)
            if piece:
                index = piece.encode_index(current_player)
                input_tensor[y, x, index] = 1.0

    return np.expand_dims(input_tensor, axis=0)

def move_to_index(move):
    if move.piece_to_drop:
        p = move.piece_to_drop.short_name()
        i = piece_to_index[p]
        return 4000 + i * 25 + move.to_y * 5 + move.to_x
    else:
        index = ((move.from_y * 5 + move.from_x) * 25 + move.to_y * 5 + move.to_x) * 2
        return index + (1 if move.promote else 0)

def index_to_move(index, board, player, players):
    legal_moves = get_legal_moves(board, player, players)
    for move in legal_moves:
        if move_to_index(move) == index:
            return move
    return None

def choose_move_from_model(model, board, player, players):
    input_tensor = board_to_input_tensor(board, player.owner)

    policy_logits, _ = model.predict(input_tensor)
    policy = policy_logits[0]

    legal_moves = get_legal_moves(board, player, players)
    legal_indices = [move_to_index(m) for m in legal_moves]
    legal_probs = [(i, policy[i]) for i in legal_indices]
    legal_probs.sort(key=lambda x: x[1], reverse=True)

    for i, _ in legal_probs:
        move = index_to_move(i, board, player, players)
        if move:
            return move
    return None


# === ユーザー入力のパース ===
def parse_move_input(raw_input, player):
    parts = raw_input.strip().split()
    if parts[0] == 'm' and len(parts) >= 5:
        fx, fy, tx, ty = map(int, parts[1:5])
        promote = len(parts) > 5 and parts[5] == 'p'
        return Move(fx, fy, tx, ty, promote)
    elif parts[0] == 'd' and len(parts) == 4:
        piece_name = parts[1]
        tx, ty = map(int, parts[2:])
        for i, p in enumerate(player.captured_pieces):
            if p.short_name() == piece_name:
                return Move(None, None, tx, ty, piece_to_drop=player.captured_pieces.pop(i))
    return None

def run_vs_model(model):
    game = Game()
    human_player_id = int(input("先手(0) or 後手(1) どちらで対戦しますか？ > "))

    while True:
        game.board.display(game.players)
        player = game.current_player()

        if Rule.is_checkmate(game.board, player, game.players):
            print(f"詰みです！ {game.players[1 - game.turn].name}の勝ちです。")
            break

        if Rule.is_in_check(game.board, player.owner):
            print("*** 王手です！ ***")

        if player.owner == human_player_id:
            raw_input = input("指し手を入力してください (例: 'm 2 3 2 2' or 'd 歩 3 3'): ")
            move = parse_move_input(raw_input, player)
        else:
            print("AIが思考中...")
            move = choose_move_from_model(model, game.board, player, game.players)

        if not move:
            print("不正な入力または合法手がありません")
            continue

        if not game.play_move(move):
            if move.piece_to_drop:
                player.captured_pieces.append(move.piece_to_drop)
            print("不正な手でした。やり直してください。")

piece_to_index = {
    '歩': 0, '香': 1, '桂': 2, '銀': 3, '金': 4,
    '角': 5, '飛': 6, 'と': 7, '成香': 8, '成桂': 9, '成銀': 10,
    '馬': 11, '竜': 12
}

model = load_model("model_best.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError())
run_vs_model(model)
