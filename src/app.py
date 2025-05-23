from flask import Flask, render_template, jsonify, request
from src.game.game import Game2048
from src.ai.random import RandomAgent
from src.ai.mcts import MctsAgent
import os
import importlib

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Store game instances in a simple dictionary
games = {}

# Available AI agents
agents = {
    'random': RandomAgent(),
    'mcts': MctsAgent(thinking_time=0.2, max_sim_depth=20, rollout_type="greedy", max_branching_factor=2),
    # Add other agents as they're implemented
    # 'expectimax': ExpectimaxAgent(),
    # 'heuristic': HeuristicAgent(),
    # 'rl': RLAgent(),
}

def load_agent(agent_type):
    if agent_type in agents:
        return agents[agent_type]
        
    try:
        # Try to dynamically import the requested agent
        module_name = f"src.ai.{agent_type}_agent"
        class_name = f"{agent_type.capitalize()}Agent"
        module = importlib.import_module(module_name)
        agent_class = getattr(module, class_name)
        agent = agent_class()
        agents[agent_type] = agent
        
        return agent
    except (ImportError, AttributeError) as e:
        print(f"Agent '{agent_type}' could not be loaded: {e}")
        return agents['random']

@app.route('/')
def index():
    agent_types = list(agents.keys())
    return render_template('index.html', agents=agent_types)

@app.route('/api/new_game', methods=['POST'])
def new_game():
    game_id = request.json.get('game_id', 'default')
    games[game_id] = Game2048()
    
    game = games[game_id]
    
    return jsonify(
        board=game.board.grid.tolist(), 
        score=game.score,
        game_over=game.is_game_over(),
        highest_tile=int(game.board.get_max_tile())
    )

@app.route('/api/make_move', methods=['POST'])
def make_move():
    game_id = request.json.get('game_id', 'default')
    direction = int(request.json.get('direction'))
    
    game = games.get(game_id)
    if not game:
        return jsonify(error="Game not found"), 404
    
    move_result = game.step(direction)
    result = sanitize_response(move_result, game)
    
    return jsonify(result)

@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    game_id = request.json.get('game_id', 'default')
    ai_type = request.json.get('ai_type', 'random')  # Default to random
    
    game = games.get(game_id)
    if not game:
        return jsonify(error="Game not found"), 404
    
    agent = load_agent(ai_type)
    direction = agent.get_move(game)
    
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
    
    move_result = game.step(direction)
    result = sanitize_response(move_result, game)
    result.update({
        'direction': int(direction),
        'agent_type': ai_type,
        'agent_stats': agent.get_stats() if hasattr(agent, 'get_stats') else {}
    })
    
    return jsonify(result)

@app.route('/api/get_agent_types', methods=['GET'])
def get_agent_types():
    return jsonify(list(agents.keys()))

@app.route('/api/reset_game', methods=['POST'])
def reset_game():
    game_id = request.json.get('game_id', 'default')
    
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
    """
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
    
    # additional info
    if 'score_gained' in move_result:
        result['score_gained'] = int(move_result['score_gained'])
    if 'total_score' in move_result:
        result['total_score'] = int(move_result['total_score'])
    
    return result

if __name__ == '__main__':
    app.run(debug=True)
