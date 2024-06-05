import gym
from gym import spaces
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Dicionário do deck (apenas usado pra visualização se quiser e não internamente)
dict_deck = {
    14: '4p',  # 4 de paus (Zap)
    13: '7c',  # 7 de copas (Copeta)
    12: 'Ae',  # Ás de espadas (Espadilha)
    11: '7o',  # 7 de ouros (Ourito)
    10: '3',
    9: '2',
    8: 'A',
    7: 'K',
    6: 'J',
    5: 'Q',
    4: '7',
    3: '6',
    2: '5',
    1: '4',
    0: 'Carta já jogada'
}

class TrucoMineiroEnv(gym.Env):
    def __init__(self):
        # Cria o deck
        self.deck =  self._create_deck()
        # Contador de mãos jogadas
        self.turn = 1
        # Definindo o espaço de ação (0, 1, 2 representam as cartas na mão do agente)
        self.action_space = spaces.Discrete(3)
        # Definindo o espaço de observação (carta jogada pelo oponente, cartas na mão do agente, estado da primeira mão)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(15),  # Carta jogada pelo oponente, 0 se for a vez do agente
            spaces.MultiDiscrete([15]*3),  # Cartas na mão do agente (0 representa carta jogada)
            spaces.Discrete(3)  # Estado da primeira mão (0 - essa é a primeira mão, 1 - oponente ganhou, 2 - empate, 3 - agente ganhou)
        ))
        # Variáveis de estado
        self.opponent_card = 0  # Carta jogada pelo oponente
        self.agent_cards = np.array([self.draw(), self.draw(), self.draw()])  # Agente compra 3 cartas
        self.first_hand_winner = 0  # Estado da primeira mão

    def _create_deck(self):
        # Deck de cartas com 4p=14 > 7c=13 > Ae=12 > 7o=11 > 3=10 > 2=9 > A=8 > K=7 > J=6 > Q=5 > 7=4 > 6=3 > 5=2 > 4=1
        # 1 de cada manilha, 3 cartas A, 3 cartas 4, 2 cartas 7 e as 4 das demais
        return 1*[14] + 1*[13] + 1*[12] + 1*[11] + 4*[10] + 4*[9] + 3*[8] + 4*[7] + 4*[6] + 4*[5] + 2*[4] + 4*[3] + 4*[2] + 3*[1]

    def draw(self):
        # Compra uma carta do deck
        if self.deck:
            card_index = random.randint(0, len(self.deck) - 1)
            card = self.deck.pop(card_index)
            return card
        else:
            return None

    def step(self, action):
        # Executa a ação (joga uma carta se a ação for 1, 2 ou 3)
        if action != 0:
            played_card = self.agent_cards[action - 1]
            self.agent_cards[action - 1] = 0  # Marca a carta como jogada
        else:
            played_card = 0

        # Determina o vencedor da mão se ambos jogaram
        if self.opponent_card != 0 and played_card != 0:
            hand_winner = self._determine_hand_winner(self.opponent_card, played_card)

        # Determina o vencedor da rodada, se existir
        round_winner = self._determine_round_winner(hand_winner)

        if self.turn == 1:
          self.first_hand_winner = hand_winner

        # Avança o turno
        self.turn += 1

        # Determina a recompensa (0 para empates ou rodada inacabada, +1 vitória, -1 derrota)
        reward = 0 if round_winner == 0 else round_winner - 2

        # Retorna a observação, a recompensa (-1, 0 ou 1) se a rodada acabou ou 0 se a rodada não acabou e a flag de rodada acabada
        return (self.opponent_card, self.agent_cards, self.first_hand_winner), reward, round_winner != 0

    def reset(self):
        # Reseta o ambiente
        self.deck =  self._create_deck()
        self.turn = 1
        self.opponent_card = 0
        self.agent_cards = np.array([self.draw(), self.draw(), self.draw()])
        self.first_hand_winner = 0
        return self.opponent_card, self.agent_cards, self.first_hand_winner

    def _determine_hand_winner(self, opponent_card, agent_card):
        # Lógica para determinar o vencedor de uma mão (1=oponente ganha 2=empate 3=agente ganha)
        if agent_card > opponent_card:
          return 3
        if agent_card < opponent_card:
          return 1
        return 2

    def _determine_round_winner(self, hand_winner):
        # Lógica para determinar o vencedor de uma rodada (0=indeterminado 1=oponente ganha 2=empate 3=agente ganha)
        if self.turn == 3 or self.first_hand_winner == 2: # Turno 3 ou primeira mão empatou
          return hand_winner
        if self.first_hand_winner == 1 and hand_winner != 3: # Oponente ganha primeira mão
          return 1
        if self.first_hand_winner == 3 and hand_winner != 1: # Agente ganha primeira mão
          return 3
        return 0 # Default

def test():
   # Testes no ambiente
    truco = TrucoMineiroEnv()
    observation = truco.reset()
    done = False
    total_reward = 0

    while not done:
        print(f"Observação (opponent_card, agent_cards[], first_hand_winner): {observation}")
        print(f"Cartas do agente: {[dict_deck[card_value] for card_value in truco.agent_cards]}")
        if dict_deck[truco.opponent_card] == 'Carta indisponível':
            print(f"Carta do oponente: ele joga depois")
        else:
            print(f"Carta do oponente: {dict_deck[truco.opponent_card]}")
        while True:
            action = random.randint(0, 2)  # Escolhe uma ação aleatória (trocar pelo agente)
            if truco.agent_cards[action] != 0:
                break
        print(f"Carta jogada pelo agente: {dict_deck[truco.agent_cards[action]]}")
        result = truco.step(action)
        observation, reward, done = result['observation'], result['reward'], result['done']
        total_reward += reward

        print(f"Recompensa obtida neste passo: {reward}")
        print(f"Recompensa acumulada: {total_reward}")
        print(f"Placar: Agente {truco.score[1]} x {truco.score[0]} Oponente\n")

if __name__ == "__main__":
    test()