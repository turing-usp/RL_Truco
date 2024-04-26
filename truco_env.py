# Imports
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Dicionário do deck (apenas usado para visualização se quiser e não internamente)
dict_deck = {
    14: "4p",  # 4 de paus (Zap)
    13: "7c",  # 7 de copas (Copeta)
    12: "Ae",  # Ás de espadas (Espadilha)
    11: "7o",  # 7 de ouros (Ourito)
    10: "3",
    9: "2",
    8: "A",
    7: "K",
    6: "J",
    5: "Q",
    4: "7",
    3: "6",
    2: "5",
    1: "4",
    0: "Carta indisponível",
}


class TrucoAgent:
    """
    Classe do agente com ações default aleatórias
    """

    def __init__(self, name):
        self.name = name
        self.cards = []

    def draw_cards(self, deck):
        self.cards = np.sort(
            [deck.pop(random.randint(0, len(deck) - 1)) for _ in range(3)]
        )[::-1]

    def choose_action(self):
        # Escolhe uma ação aleatória válida (uma carta para jogar)
        n_cards = sum(1 for x in self.cards if x != 0)
        action = random.randint(0, n_cards - 1)
        return action

    def play_card(self, action):
        card_played = self.cards[action]
        self.cards[action] = 0  # Marca a carta como jogada
        return card_played


# Ambiente
"""
TODO: renderizar ambiente
TODO: implementar mecânica de truco
TODO: implementar 2v2
"""

class TrucoMineiroEnv(gym.Env):
    """
    Ambiente truco mineiro 1v1 multi agentes
    """

    def __init__(self):
        # Inicializa o espaço de ação e observação
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "current_player_card": spaces.Discrete(15),
                "other_player_card": spaces.Discrete(15),
                "first_hand_winner": spaces.Discrete(
                    4
                ),  # 0=essa é a primeira mão; 1=Player 1 ganhou; 2=Player 2 ganhou; 3=empate
            }
        )
        # Cria o deck
        self.deck = self._create_deck()
        # Contador de mãos jogadas
        self.turn = 0
        # Placar [jogador1, jogador2]
        self.game_score = [0, 0]
        self.round_score = [0, 0]
        # Cria agentes
        self.players = [TrucoAgent("Player 1"), TrucoAgent("Player 2")]
        # Aleatoriza quem começa
        self.round_starter = random.randint(0, 1)
        self.current_player = self.round_starter
        self.other_player = 1 - self.current_player
        self.current_card = 0
        self.other_card = 0
        self.first_hand_winner = 0
        self.hand_winner = 0
        # Inicializa cartas
        self.reset()

    def _create_deck(self):
        # Retorna uma lista embaralhada de cartas
        deck = (
            [14, 13, 12, 11]
            + [10] * 4
            + [9] * 4
            + [8] * 3
            + [7] * 4
            + [6] * 4
            + [5] * 4
            + [4] * 2
            + [3] * 4
            + [2] * 4
            + [1] * 3
        )
        random.shuffle(deck)
        return deck

    def reset(self, reset_score=True):
        self.deck = self._create_deck()
        for player in self.players:
            player.draw_cards(self.deck)
        self.round_starter = 1 - self.round_starter
        self.current_player = self.round_starter
        self.other_player = 1 - self.current_player
        self.current_card = 0
        self.other_card = 0
        self.turn = 0
        if reset_score:
            self.game_score = [0, 0]
        self.round_score = [0, 0]
        self.first_hand_winner = 0
        return self._get_obs()

    def step(self, action):
        # Verifica se a ação é válida
        n_cards = sum(1 for x in self.players[self.current_player].cards if x != 0)
        possible_actions = range(0, n_cards)
        if action not in possible_actions:
            raise ValueError(
                f"Invalid action. Action must be one of these: {list(possible_actions)}."
            )

        # Executa a ação do jogador atual
        self.current_card = self.players[self.current_player].play_card(action)

        # Sort na mão do player
        self.players[self.current_player].cards = np.sort(
            self.players[self.current_player].cards
        )[::-1]

        # Se o outro jogador ainda não jogou, encerra o step
        if self.other_card == 0:
            self.current_player, self.other_player = (
                self.other_player,
                self.current_player,
            )
            self.current_card, self.other_card = self.other_card, self.current_card
            observation, reward, done, info = (
                self._get_obs(),
                0,
                False,
                self._get_info(),
            )
            return observation, reward, done, info

        # Determina o vencedor da mão se houver
        self.hand_winner = self._determine_hand_winner(
            self.current_card, self.other_card
        )
        if self.hand_winner == 1 or self.hand_winner == 2:
            self.round_score[self.hand_winner - 1] += 1

        # Determina o vencedor da rodada, se existir e atualiza o placar
        round_winner = self._determine_round_winner()
        if round_winner == 1 or round_winner == 2:
            self.game_score[round_winner - 1] += 1

        # Determina quem ganhou a primeira mão se estiver nela
        if self.turn == 0:
            self.first_hand_winner = self.hand_winner

        # Reseta as cartas jogadas
        self.other_card = 0
        self.current_card = 0

        # Avança o turno
        self.turn += 1

        # Determina a recompensa (0 para empates ou rodada inacabada, +1 vitória, -1 derrota)
        if round_winner == self.current_player + 1:
            reward = 1
        elif round_winner == self.other_player + 1:
            reward = -1
        else:
            reward = 0

        # Troca os jogadores de lugar se precisar (mantém em caso de empate, senão quem ganhou começa a próxima)
        self.current_player, self.other_player = self.other_player, self.current_player
        if self.hand_winner == 1:
            self.current_player, self.other_player = 0, 1
        elif self.hand_winner == 2:
            self.current_player, self.other_player = 1, 0

        done = False if 12 not in self.game_score else True

        # Retorna a observação, a recompensa (-1, 0 ou 1) se a rodada acabou ou 0 se a rodada não acabou e a flag de rodada acabada
        if round_winner != 0:
            self.reset(reset_score=False)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def _determine_hand_winner(self, card1, card2):
        # Determina quem vence a mão (1=Player 1 ganha; 2=Player 2 ganha; 3=empate)
        if card1 > card2:
            return self.current_player + 1  # Current ganha
        elif card2 > card1:
            return self.other_player + 1  # Other jogador ganha
        return 3  # Empate

    def _get_obs(self):
        return {
            "current_player_cards": self.players[self.current_player].cards,
            "other_card": self.other_card,
            "first_hand_winner": self.first_hand_winner,
        }

    def _get_info(self):
        return {
            "current_player": self.current_player,
            "player1_cards": self.players[0].cards,
            "player2_cards": self.players[1].cards,
            "round_score": self.round_score,
            "game_score": self.game_score,
            "hand_winner": self.hand_winner - 1,
            "first_hand_winner": self.first_hand_winner,
        }

    def _determine_round_winner(self):
        # Lógica para determinar o vencedor de uma rodada
        # 0=indeterminado; 1=Player 1 ganha; 2=Player 2 ganha; 3=empate
        if self.turn == 2:  # Terceiro turno
            return self.hand_winner
        if self.first_hand_winner == 3:  # Primeira mão empatou
            if self.hand_winner == 3:
                return 0
            return self.hand_winner
        if (
            self.first_hand_winner == self.current_player + 1
            and self.hand_winner != self.other_player + 1
        ):  # Current ganha primeira mão
            return self.current_player + 1
        if (
            self.first_hand_winner == self.other_player + 1
            and self.hand_winner != self.current_player + 1
        ):  # Other ganha primeira mão
            return self.other_player + 1
        return 0  # Default


def test_game():
    # Testes no ambiente
    truco = TrucoMineiroEnv()
    observation = truco.reset()
    done = False
    rewards = [0, 0]
    switch = False

    while not done:
        if not switch:
            print(
                f"Mão {truco.turn + 1}/3 - Placar do round: {truco.players[0].name} ({truco.round_score[0]} x {truco.round_score[1]}) {truco.players[1].name}"
            )

        current_player = truco.players[truco.current_player]
        other_player = truco.players[truco.other_player]

        print(f"Vez do {current_player.name}")
        print(f"Observação: {observation}")
        print(
            f"Cartas do {current_player.name}: {[dict_deck[card] for card in current_player.cards]}"
        )

        # Escolhe uma ação aleatória para o jogador atual (trocar pelo agente RL)
        action = current_player.choose_action()
        print(
            f"Carta jogada pelo {current_player.name}: {dict_deck[current_player.cards[action]]}"
        )

        observation, reward, done, info = truco.step(action)

        # Distribui as recompensas
        player1_reward = reward if current_player.name == "Player 1" else -reward
        player2_reward = reward if current_player.name == "Player 2" else -reward
        rewards[0], rewards[1] = (
            rewards[0] + player1_reward,
            rewards[1] + player2_reward,
        )

        print(
            f"Recompensa obtida neste passo: {truco.players[0].name} = {player1_reward}; {truco.players[1].name} = {player2_reward}"
        )

        if switch:
            hand_winner = info["hand_winner"]
            if hand_winner == 2:
                print("Mão empatada.")
            else:
                print(f"{truco.players[hand_winner].name} ganhou.")
            print(
                f"Recompensas acumuladas: {truco.players[0].name} = {rewards[0]}; {truco.players[1].name} = {rewards[1]}"
            )
            print(
                f"Placar do jogo: {truco.players[0].name} ({truco.game_score[0]} x {truco.game_score[1]}) {truco.players[1].name}\n"
            )

        switch = not (switch)

if __name__ == "__main__":
    test_game()