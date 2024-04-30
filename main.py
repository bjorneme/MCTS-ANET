from torch import optim

from ANET import ANET
from MCTSSystem import MCTSSystem
from Topp import Topp
from games.Hex import Hex

game = Hex(4)
anet = ANET(config_path="config/anet.json")

system = MCTSSystem(
    anet,
    state_manager=game,
    board_size=4,
    total_actions=16,
    optimizer=optim.Adam(anet.parameters(), lr=0.001),
    num_games=100,
    batch_size=256,
    c=1.41,
    mcts_searches=2500,
    model_path='models/model_Hex_4_3.pth',
    optimizer_path='optimizer_0.pth'
)
# system.run_system()

topp = Topp(
    model_paths = ['models/model_untrained.pth','model_50.pth', 'model_60.pth','model_70.pth', 'models/model_Hex_4_3.pth'],
    num_games= 20,
    anet = ANET(config_path="config/anet.json"),
    state_manager=Hex(4)
)
topp.run_tournament()