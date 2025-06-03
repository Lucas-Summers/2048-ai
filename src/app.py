from flask import Flask, render_template, jsonify, request
from src.game.game import Game2048
from src.ai.random import RandomAgent
from src.ai.mcts import MctsAgent
from src.ai.expectimax import ExpectimaxAgent
from src.ai.rl import RLAgent
from src.ai.greedy import GreedyAgent
import os
import importlib
import time
import numpy as np

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

games = {}

SPEED_THINKING_TIMES = {
    'full': 0.2,
    'fast': 0.5,
    'medium': 1.0,
    'slow': 2.0
}
DEFAULT_THINKING_TIME = SPEED_THINKING_TIMES['fast']

RL_MODEL_PATH = "src/utils/runs/2025-05-31_15-39-49/best_2842.pt"
rl_rollout_agent = RLAgent.load_model(RL_MODEL_PATH, training=False, name="RL_Rollout")

agents = {
    'random': RandomAgent(),
    'greedy': GreedyAgent(tile_weight=1.0, score_weight=0.1),
    'mcts': MctsAgent(thinking_time=DEFAULT_THINKING_TIME),
    'mcts_exp': MctsAgent(thinking_time=DEFAULT_THINKING_TIME, rollout_type="expectimax"),
    'expect': ExpectimaxAgent(thinking_time=DEFAULT_THINKING_TIME),
    'rl': RLAgent(training=False),
    'mcts_rl': MctsAgent(thinking_time=DEFAULT_THINKING_TIME, rollout_type="rl", rl_model_path=RL_MODEL_PATH),
}

def load_agent(agent_type, thinking_time=None):
    if thinking_time is None:
        thinking_time = DEFAULT_THINKING_TIME
        
    if agent_type == 'mcts':
        return MctsAgent(thinking_time=thinking_time)
    elif agent_type == 'mcts_exp':
        return MctsAgent(thinking_time=thinking_time, rollout_type="expectimax")
    elif agent_type == 'expect':
        return ExpectimaxAgent(thinking_time=thinking_time)
    elif agent_type == 'mcts_rl':
        return MctsAgent(thinking_time=thinking_time, rollout_type="rl", rl_model_path=RL_MODEL_PATH)
    
    if agent_type in agents:
        return agents[agent_type]
        
    try:
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
    speed = request.json.get('speed', 'fast')  # Default to fast speed
    
    game = games.get(game_id)
    if not game:
        return jsonify(error="Game not found"), 404
    
    thinking_time = SPEED_THINKING_TIMES.get(speed, DEFAULT_THINKING_TIME)
    
    # Load agent with appropriate thinking time
    agent = load_agent(ai_type, thinking_time)
    
    start_time = time.time()
    try:
        direction = agent.get_move(game)
    except Exception as e:
        print(f"Error in {ai_type} agent.get_move(): {e}")
        import traceback
        traceback.print_exc()
        return jsonify(
            error=f"Agent error: {str(e)}",
            moved=False,
            game_over=True,
            board=game.board.grid.tolist(),
            score=int(game.score),
            agent_type=ai_type
        ), 500
    
    # Standardize thinking time across all agents
    elapsed_time = time.time() - start_time
    if elapsed_time < thinking_time:
        time.sleep(thinking_time - elapsed_time)
    total_thinking_time = time.time() - start_time
    
    if direction == -1:
        agent_stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}
        agent_stats = to_native_type(agent_stats)
        return jsonify(
            moved=False,
            game_over=True,
            board=game.board.grid.tolist(),
            score=int(game.score),
            highest_tile=int(game.board.get_max_tile()),
            agent_type=ai_type,
            agent_stats=agent_stats,
            thinking_time=total_thinking_time,
            speed=speed
        )
    
    move_result = game.step(direction)
    result = sanitize_response(move_result, game)
    
    agent_stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}
    agent_stats['thinking_time'] = total_thinking_time
    agent_stats = to_native_type(agent_stats)
    
    result.update({
        'direction': int(direction),
        'agent_type': ai_type,
        'agent_stats': agent_stats,
        'speed': speed
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

def safe_int(value):
    """Safely convert NumPy types to Python integers."""
    if hasattr(value, 'item'):
        return int(value.item())
    else:
        return int(value)

def sanitize_response(move_result, game):
    """Convert any NumPy types to native Python types for JSON serialization."""
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
            'from': [safe_int(move['from'][0]), safe_int(move['from'][1])],
            'to': [safe_int(move['to'][0]), safe_int(move['to'][1])],
            'value': safe_int(move['value'])
        })
    result['movements'] = movements
    
    # Convert merges
    merges = []
    for merge in move_result.get('merges', []):
        merged_from = []
        for pos in merge['merged_from']:
            merged_from.append([safe_int(pos[0]), safe_int(pos[1])])
        
        merges.append({
            'position': [safe_int(merge['position'][0]), safe_int(merge['position'][1])],
            'value': safe_int(merge['value']),
            'merged_from': merged_from if merged_from else None
        })
    result['merges'] = merges
    
    # Convert new tile info
    new_tile = move_result.get('new_tile')
    if new_tile:
        result['new_tile'] = {
            'position': [safe_int(new_tile['position'][0]), safe_int(new_tile['position'][1])],
            'value': safe_int(new_tile['value'])
        }
    else:
        result['new_tile'] = None
    
    # Additional info
    if 'score_gained' in move_result:
        result['score_gained'] = safe_int(move_result['score_gained'])
    if 'total_score' in move_result:
        result['total_score'] = safe_int(move_result['total_score'])
    
    return result

def to_native_type(obj):
    """Convert numpy types to native python dicts/lists"""
    if isinstance(obj, dict):
        return {k: to_native_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native_type(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_native_type(v) for v in obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
