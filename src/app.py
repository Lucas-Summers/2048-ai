from flask import Flask, render_template, jsonify, request
from src.game.board import Board
from src.game.game import Game2048
from src.ai.base import Agent
from src.ai.random import RandomAgent
import os
import importlib
import numpy as np

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Store game instances in a simple dictionary
games = {}

# Available AI agents
agents = {
    'random': RandomAgent(),
    # Add other agents as they're implemented
    # 'expectimax': ExpectimaxAgent(),
    # 'mcts': MCTSAgent(),
    # 'heuristic': HeuristicAgent(),
    # 'rl': RLAgent(),
}

def load_agent(agent_type):
    """Dynamically load an agent by type."""
    if agent_type in agents:
        return agents[agent_type]
        
    try:
        # Try to dynamically import the requested agent
        module_name = f"src.ai.{agent_type}_agent"
        class_name = f"{agent_type.capitalize()}Agent"
        
        # Import the module
        module = importlib.import_module(module_name)
        
        # Get the agent class and create an instance
        agent_class = getattr(module, class_name)
        agent = agent_class()
        
        # Cache the agent for future use
        agents[agent_type] = agent
        
        return agent
    except (ImportError, AttributeError) as e:
        # If agent can't be loaded, fall back to random
        print(f"Agent '{agent_type}' could not be loaded: {e}")
        return agents['random']

@app.route('/')
def index():
    """Render the main game page"""
    agent_types = list(agents.keys())
    return render_template('index.html', agents=agent_types)

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game"""
    game_id = request.json.get('game_id', 'default')
    games[game_id] = Game2048()
    
    # Get initial board state
    game = games[game_id]
    
    return jsonify(
        board=game.board.grid.tolist(), 
        score=game.score,
        game_over=game.is_game_over(),
        highest_tile=int(game.board.get_max_tile())
    )

@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Make a player move"""
    game_id = request.json.get('game_id', 'default')
    direction = int(request.json.get('direction'))
    
    game = games.get(game_id)
    if not game:
        return jsonify(error="Game not found"), 404
    
    # Try to make the move - the result now has all the movement info directly
    move_result = game.step(direction)
    
    # Convert any NumPy types to native Python types for JSON serialization
    result = sanitize_response(move_result, game)
    
    return jsonify(result)

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    """Get and make an AI move"""
    game_id = request.json.get('game_id', 'default')
    ai_type = request.json.get('ai_type', 'random')  # Default to random
    
    game = games.get(game_id)
    if not game:
        return jsonify(error="Game not found"), 404
    
    # Load the requested agent
    agent = load_agent(ai_type)
    
    # Get AI's move
    direction = agent.get_move(game)
    
    # If no valid move is available
    if direction == -1:
        return jsonify(
            moved=False,
            game_over=True,
            board=game.board.grid.tolist(),
            score=int(game.score),
            highest_tile=int(game.board.get_max_tile()),
            agent_type=ai_type,
            agent_stats=agent.get_stats() if hasattr(agent, 'get_stats') else {}
        )
    
    # Make the move
    move_result = game.step(direction)
    
    # Convert any NumPy types to native Python types for JSON serialization
    result = sanitize_response(move_result, game)
    
    # Add AI-specific information
    result.update({
        'direction': int(direction),
        'agent_type': ai_type,
        'agent_stats': agent.get_stats() if hasattr(agent, 'get_stats') else {}
    })
    
    return jsonify(result)

@app.route('/api/get_agent_types', methods=['GET'])
def get_agent_types():
    """Get a list of available AI agent types"""
    return jsonify(list(agents.keys()))

@app.route('/api/reset_game', methods=['POST'])
def reset_game():
    """Reset an existing game or create a new one"""
    game_id = request.json.get('game_id', 'default')
    
    # If game exists, reset it, otherwise create a new one
    if game_id in games:
        games[game_id].reset()
    else:
        games[game_id] = Game2048()
    
    game = games[game_id]
    
    return jsonify(
        board=game.board.grid.tolist(),
        score=game.score,
        game_over=game.is_game_over(),
        highest_tile=int(game.board.get_max_tile())
    )

def sanitize_response(move_result, game):
    """
    Convert any NumPy types to native Python types for JSON serialization.
    
    Args:
        move_result: Result from game.step()
        game: The Game2048 instance
        
    Returns:
        dict: JSON-serializable response
    """
    # Convert basic information
    result = {
        'board': game.board.grid.tolist(),
        'score': int(game.score),
        'moved': move_result['moved'],
        'game_over': game.is_game_over(),
        'highest_tile': int(game.board.get_max_tile())
    }
    
    # Convert movements
    movements = []
    for move in move_result.get('movements', []):
        movements.append({
            'from': [int(move['from'][0]), int(move['from'][1])],
            'to': [int(move['to'][0]), int(move['to'][1])],
            'value': int(move['value'])
        })
    result['movements'] = movements
    
    # Convert merges
    merges = []
    for merge in move_result.get('merges', []):
        # Handle different merge formats (with or without merged_from)
        merged_from = []
        if 'merged_from' in merge:
            for pos in merge['merged_from']:
                merged_from.append([int(pos[0]), int(pos[1])])
        
        merges.append({
            'position': [int(merge['position'][0]), int(merge['position'][1])],
            'value': int(merge['value']),
            'merged_from': merged_from if merged_from else None
        })
    result['merges'] = merges
    
    # Convert new tile info
    new_tile = move_result.get('new_tile')
    if new_tile:
        result['new_tile'] = {
            'position': [int(new_tile['position'][0]), int(new_tile['position'][1])],
            'value': int(new_tile['value'])
        }
    else:
        result['new_tile'] = None
    
    # Add any additional information
    if 'score_gained' in move_result:
        result['score_gained'] = int(move_result['score_gained'])
    if 'total_score' in move_result:
        result['total_score'] = int(move_result['total_score'])
    
    return result

if __name__ == '__main__':
    app.run(debug=True)
