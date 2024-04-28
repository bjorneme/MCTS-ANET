

# Replaybuffer: store samples from self play
import random

class ReplayBuffer:
    def __init__(self, max_size = 10000):
        self.max_size = max_size # Max size replay buffer
        self.buffer = [] # The replay buffer
        self.position = 0 # Position in the buffer

    def push_to_buffer(self, board_state, action_probs, value, player):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)

        # Save to the buffer
        self.buffer[self.position] = (board_state, action_probs, value, player)
        self.position = (self.position + 1) % self.max_size

    def get_sample(self, batch_size):
        # Retrieve a batch of elements randomly
        samples = random.sample(self.buffer, batch_size)

        # Unpack the samples
        board_states, action_probs, values, players = zip(*samples)

        # Return the samples
        return board_states, action_probs, values, players


