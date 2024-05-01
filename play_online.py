# Import and initialize your own actor
import numpy as np
import torch.nn.functional as F
import torch
from ANET.ANET import ANET
actor = ANET(config_path="config/anet.json")

# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient

# Import and override the `handle_get_action` hook in ActorClient
class MyClient(ActorClient):
    def handle_get_action(self, state):

        actor.load_state_dict(torch.load("models/model_Hex_7.pth"))

        flat_board = np.array(state[1:])
        matrix_2d = np.zeros((7,7))

        # Convert 0 to 1 and non-zero to 0
        valid_moves = np.where(flat_board == 0, 1 or 2, 0)

        # Fill the matrix with elements from the original array
        matrix_2d.ravel()[:49] = flat_board

        # Replace 2 with -1
        matrix_2d[matrix_2d == 2] = -1


        input = actor.prepare_input([matrix_2d], [state[0]])
        predicted_probs = actor(input)
        predicted_probs = F.softmax(predicted_probs, dim=1) * torch.Tensor(valid_moves)

        move = int(torch.argmax(predicted_probs[0]))

        row = move // 7
        col = move % 7


        return row, col
    


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(auth="202205104a9b4384b4e24c90557b7e1d")
    client.run()