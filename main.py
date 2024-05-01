import json
from torch import optim

from Anet.ANET import ANET
from MCTSSystem import MCTSSystem
from Topp.PlayGame import PlayGame
from Topp.Topp import Topp
from games.Hex import Hex
from games.TicTacToe import TicTacToe

# Funciton for loading the config file
def load_config(config_file='config/parameters.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    # Load the config
    config = load_config()

    # Select game
    board_size = config['game']['board_size']
    if(config['game']['type_of_game'] == 'TicTacToe'):
        game = TicTacToe()
        total_actions = board_size**2
    elif(config['game']['type_of_game'] == 'Hex'):
        game = Hex(board_size)
        total_actions = board_size**2
    else:
        raise ValueError(f"Unsupported game!")

    # Initialize the Actor Network
    anet = ANET(config_path="config/anet.json")

    # Initialize learning rate
    learning_rate = config['training']['learning_rate']

    # Initialize optimizer based on configuration
    optimizer_name = config['training']['optimizer']  
    if optimizer_name == "Adam":
        optimizer = optim.Adam(anet.parameters(), lr=learning_rate)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(anet.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(anet.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSProp":
        optimizer = optim.RMSprop(anet.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Select training or TOPP
    if(config['mode'] == 'Training'):

        # Initalize the system class
        mcts = MCTSSystem(
            anet,
            state_manager=game,
            board_size=board_size,
            total_actions=total_actions,
            optimizer=optimizer,
            num_games=config['training']['num_episodes'],
            batch_size=config['training']['batch_size'],
            c=config['training']['c'],
            mcts_searches = config['training']['mcts_searches'],
            e_greedy_mcts = config['training']['e_greedy_mcts'],
            num_anet_cached = config['topp']['num_anet_cached'],
            model_path=config['training']['model_path'],
            optimizer_path=config['training']['optimizer_path']
        )

        # Run the system
        mcts.run_system()

        # Plot the learning progress
        if config['plot_learning']:
            mcts.plot_learning_progress()

    # Select training or TOPP
    elif(config['mode'] == 'Topp'):

        # Initialize Topp class
        topp = Topp(
            model_paths=config['topp']['model_paths'],
            num_games=config['topp']['num_games'],
            anet=anet,
            state_manager=game
        )

        # Run Topp
        topp.run_tournament()

    # Select training or TOPP
    elif(config['mode'] == 'Play'):

        # Initialize Play game class
        match = PlayGame(
            state_manager=game,
            anet=anet,
            model_path=config['playgame']['model_path']
        )

        # Play match
        match.play_match(config['playgame']['start'])

if __name__ == "__main__":
    main()
