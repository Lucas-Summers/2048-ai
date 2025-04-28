from flask import Flask, render_template, jsonify, request
from src.game.board import Board
from src.game.game import Game2048
from src.ai.random_agent import RandomAgent
import os
# Import MinimaxAgent when you implement it
# from src.ai.minimax import MinimaxAgent

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
# app = Flask(__name__)
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Store game instances in a simple dictionary
games = {}

@app.route('/')
def index():
    """Render the main game page"""
    return render_template('index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game"""
    game_id = request.json.get('game_id', 'default')
    games[game_id] = Game2048()
    return jsonify(board=games[game_id].board.grid.tolist(), 
                   score=games[game_id].score)

@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Make a player move"""
    game_id = request.json.get('game_id', 'default')
    direction = int(request.json.get('direction'))
    
    game = games.get(game_id)
    if not game:
        return jsonify(error="Game not found"), 404
    
    move_result = game.step(direction)
    
    # Convert any NumPy types to native Python types
    movements = []
    for move in move_result['movements']:
        movements.append({
            'from': [int(move['from'][0]), int(move['from'][1])],
            'to': [int(move['to'][0]), int(move['to'][1])],
            'value': int(move['value'])
        })
        
    merges = []
    for merge in move_result['merges']:
        merges.append({
            'position': [int(merge['position'][0]), int(merge['position'][1])],
            'value': int(merge['value'])
        })
    
    new_tile = None
    if move_result['new_tile']:
        new_tile = {
            'position': [int(move_result['new_tile']['position'][0]), 
                         int(move_result['new_tile']['position'][1])],
            'value': int(move_result['new_tile']['value'])
        }
    
    return jsonify(
        board=game.board.grid.tolist(),
        score=int(game.score),
        moved=move_result['moved'],
        movements=movements,
        merges=merges,
        new_tile=new_tile,
        game_over=game.is_game_over()
    )

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """Get and make an AI move"""
    game_id = request.json.get('game_id', 'default')
    ai_type = request.json.get('ai_type', 'random')  # Default to random for now
    
    game = games.get(game_id)
    if not game:
        return jsonify(error="Game not found"), 404
    
    # For now, only implement the random agent
    agent = RandomAgent()
    
    # Get AI's move
    direction = agent.get_move(game)
    
    # If no valid move is available
    if direction == -1:
        return jsonify(
            moved=False,
            game_over=True,
            board=game.board.grid.tolist(),  # Convert NumPy array to list
            score=int(game.score),  # Convert to native Python int
            highest_tile=int(game.board.get_max_tile())  # Convert to native Python int
        )
    
    # Make the move
    move_result = game.step(direction)
    
    # Convert any NumPy types to native Python types
    movements = []
    for move in move_result['movements']:
        movements.append({
            'from': [int(move['from'][0]), int(move['from'][1])],
            'to': [int(move['to'][0]), int(move['to'][1])],
            'value': int(move['value'])
        })
        
    merges = []
    for merge in move_result['merges']:
        merges.append({
            'position': [int(merge['position'][0]), int(merge['position'][1])],
            'value': int(merge['value'])
        })
    
    new_tile = None
    if move_result['new_tile']:
        new_tile = {
            'position': [int(move_result['new_tile']['position'][0]), 
                         int(move_result['new_tile']['position'][1])],
            'value': int(move_result['new_tile']['value'])
        }
    
    return jsonify(
        board=game.board.grid.tolist(),
        score=int(game.score),
        moved=move_result['moved'],
        direction=int(direction),
        movements=movements,
        merges=merges,
        new_tile=new_tile,
        game_over=game.is_game_over(),
        highest_tile=int(game.board.get_max_tile())
    )

@app.route('/api/update_generator', methods=['POST'])
def update_generator():
    """Update tile generator settings"""
    game_id = request.json.get('game_id', 'default')
    generator = request.json.get('generator', 'random')
    
    game = games.get(game_id)
    if not game:
        return jsonify(error="Game not found"), 404
    
    # You would implement this feature in your game class
    # For now, just acknowledge the request
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
