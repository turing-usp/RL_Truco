import random
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrucoPlayer:
    """
    Superclasse para todos os jogadores
    """
    def __init__(self, name):
        self.name = name
        self.type = None # LearningPlayer or NonLearningPlayer

class LearningPlayer(TrucoPlayer):
    def __init__(self, name):
        super().__init__(name)
        self.type = LearningPlayer

class NonLearningPlayer(TrucoPlayer):
    def __init__(self, name):
        super().__init__(name)
        self.type = NonLearningPlayer

    def choose_action(self, obs, valid_actions):
        raise NotImplementedError

class RandomBotPlayer(NonLearningPlayer):
    """
    Classe do jogador com ações aleatórias
    """

    def choose_action(self, obs, info):
        return random.choice(info["valid_actions"])

class NetworkBotPlayer(NonLearningPlayer):
    """
    Classe do jogador cuja estratégia é dada por uma rede neural
    """

    def __init__(self, name, network):
        super().__init__(name)
        self.network = network
    
    def convert_obs_to_state(self, obs):
        state = np.array([*obs["current_player_cards"], obs["other_card"], obs["first_hand_winner"], obs["current_player_score"], obs["other_player_score"], obs["current_bet"], int(obs["trucable"]), int(obs["respond"]), *obs["card_frequency"]], dtype=np.int64)
        state = torch.from_numpy(state).unsqueeze(dim=0).float().to(device)
        return state
    
    def choose_action(self, obs, info):
        state = self.convert_obs_to_state(obs)
        valid_actions = info["valid_actions"]
        av = self.network(state).detach()
        action = valid_actions[np.argmax([av[0, valid_action].item() for valid_action in valid_actions])]
        return action

class HumanPlayer(NonLearningPlayer):
    """
    Classe do jogador humano (recebe a ação inserida pelo usuário)
    """

    def choose_action(self, obs, valid_actions):
        print(f"obs: {obs}")
        print(f"valid_actions: {valid_actions}")
        return int(input("Chosen action: "))