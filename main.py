# Simulate the four Topp games
from ANET import ANET
from Topp import Topp
from games.Hex import Hex


topp = Topp(
    model_paths = ['models/model_Hex_4_1.pth','models/model_Hex_4_2.pth', 'models/model_Hex_4_3.pth','models/model_untrained.pth'],
    num_games= 20,
    anet = ANET(),
    state_manager=Hex(4)
)
topp.run_tournament()