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
        # Espaço de ação
        #   0: jogar carta 0, 1: jogar carta 1, 2: jogar carta 2
        #   3: pede aumento da aposta, que varia dependendo da aposta atual
        #       possibilidades: pedir truco > pedir 6 > pedir 10 > pedir 12
        #   4: aceitar aumento, 5: recusar aumento
        self.action_space = spaces.Discrete(6)
        # Espaço de observação:
        self.observation_space = spaces.Dict(
            {
                # Cartas na mão 3*(0 a 14)
                "current_player_cards": spaces.MultiDiscrete([15] * 3),
                # Cartas de 0 a 14
                "other_player_card": spaces.Discrete(15),
                # 0=essa é a primeira mão; 1=Player 1 ganhou; 2=Player 2 ganhou; 3=empate
                "first_hand_winner": spaces.Discrete(4),
                # Placares
                "current_player_score": spaces.Discrete(13),
                "other_player_score": spaces.Discrete(13),
                # Valor do round: 0=2, 1=4, 2=6, 3=10, 4=12
                "current_bet": spaces.Discrete(5),
                # Se dá para pedir/aumentar truco: 0=não, 1=sim
                "trucable": spaces.Discrete(2),
                # Se eu preciso responder ao truco/aumento
                "respond": spaces.Discrete(2),
                # Lista de quantas vezes cada carta foi jogada no round
                "card_frequency": spaces.MultiDiscrete([3, 4, 4, 2, 4, 4, 4, 3, 4, 4, 1, 1, 1, 1])
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
        # Frequência de cartas jogadas no round
        self.card_frequency = np.array(14 * [0])
        # Mecânica de truco
        self.current_bet = 2
        self.trucable = [True, True] # Se é trucável/aumentável
        self.respond = False
        # Inicializa cartas
        self.reset()

    def _create_deck(self):
        # Retorna uma lista embaralhada de cartas
        # 4 de cada carta exceto manilhas (1), A (3), 7 (2) e 4 (3)
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
        self.card_frequency *= 0
        self.current_bet = 2
        self.trucable = [True, True]
        self.respond = False
        return self._get_obs()

    def step(self, action):
        # Quando um truco/aumento precisa ser respondido
        if self.respond == True and action in [0, 1, 2]:
            raise ValueError(f"Invalid action. Player needs to respond to truco/raise call.")
        # Ações de jogar carta
        if (action >= 0) and (action <= 2):
            return self.handle_play_card(action)
        # Pede truco
        elif action == 3:
            return self.handle_truco_call()
        # Responde ao truco
        elif action == 4 or action == 5:
            return self.handle_response(action)
        # Ação inválida
        raise ValueError(
            f"Invalid action. Action must be an integer between 0 and 5 inclusive."
        )
    
    def handle_response(self, action):
        '''
        Lógica para responder a truco ou aumento de aposta
        '''
        # Player respondeu a truco sem ninguém ter pedido
        if self.respond == False:
            raise ValueError(f"Invalid action. Player responded to a truco or raise that doesn't exist.")
        # Retira a necessidade de responder
        self.respond = False
        # Se aceita aumenta a aposta
        if action == 4:
            reward = 0
            if self.current_bet == 6:
                self.current_bet == 10
            else:
                self.current_bet += 2
            # Reforça que current pode pedir truco e other não pode
            self.trucable[self.current_player] = True
            self.trucable[self.other_player] = False
            self._switch_players()
            done = False
        # Se recusa, atualiza o placar e distribui as recompensas
        else:
            reward = -self.current_bet
            self.game_score[self.current_player] -= self.current_bet
            self.game_score[self.other_player] += self.current_bet
            done = any(x >= 12 for x in self.game_score)
            self.reset(reset_score=False)
        return self._get_obs(), reward, done, self._get_info()

    def handle_truco_call(self):
        '''
        Lógica para pedir truco ou aumentar aposta
        '''
        # Quando não pode pedir truco
        if self.trucable[self.current_player] == False:
            raise ValueError(f"Invalid action. Player is not allowed to truco/raise.")
        # Current não pode mais pedir/aumentar truco
        self.trucable[self.current_player] = False
        self.trucable[self.other_player] = True
        # Solicita resposta do other
        self.respond = True
        # Passa a vez
        self._switch_players()
        reward, done = 0, False
        return self._get_obs(), reward, done, self._get_info()

    def handle_play_card(self, action):
        '''
        Lógica para jogar carta
        '''
        # Verifica se a ação é válida
        n_cards = sum(1 for x in self.players[self.current_player].cards if x != 0)
        possible_actions = range(0, n_cards)
        if action not in possible_actions:
            raise ValueError(f"Invalid action. Player tried to play an unavailable card.")

        # Executa a ação do jogador atual
        card_played = self.players[self.current_player].play_card(action)
        self.current_card = card_played
        self.card_frequency[card_played - 1] += 1

        # Sort na mão do player
        self.players[self.current_player].cards = np.sort(
            self.players[self.current_player].cards
        )[::-1]

        # Se o outro jogador ainda não jogou, encerra o step
        if self.other_card == 0:
            self._switch_players()
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
            self.game_score[round_winner - 1] += self.current_bet

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
            reward = self.current_bet
        elif round_winner == self.other_player + 1:
            reward = -self.current_bet
        else:
            reward = 0

        # Troca os jogadores de lugar se precisar (mantém em caso de empate, senão quem ganhou começa a próxima)
        self._switch_players()
        if self.hand_winner == 1:
            self.current_player, self.other_player = 0, 1
        elif self.hand_winner == 2:
            self.current_player, self.other_player = 1, 0

        done = any(x >= 12 for x in self.game_score)

        # Retorna a observação, a recompensa (-1, 0 ou 1) se a rodada acabou ou 0 se a rodada não acabou e a flag de rodada acabada
        if round_winner != 0:
            self.reset(reset_score=False)

        return self._get_obs(), reward, done, self._get_info()

    def _switch_players(self):
        self.current_player, self.other_player = self.other_player, self.current_player
        self.current_card, self.other_card = self.other_card, self.current_card

    def _determine_hand_winner(self, card1, card2):
        # Determina quem vence a mão (1=Player 1 ganha; 2=Player 2 ganha; 3=empate)
        if card1 > card2:
            return self.current_player + 1  # Current ganha
        elif card2 > card1:
            return self.other_player + 1  # Other jogador ganha
        return 3  # Empate

    def _get_obs(self):
        bet_dict = {2:0, 4:1, 6:2, 10:3, 12:4}
        return {
            "current_player_cards": self.players[self.current_player].cards,
            "other_card": self.other_card,
            "first_hand_winner": self.first_hand_winner,
            "current_player_score": self.game_score[self.current_player],
            "other_player_score": self.game_score[self.other_player],
            "current_bet": bet_dict[self.current_bet],
            "trucable": self.trucable[self.current_player],
            "respond": self.respond,
            "card_frequency": self.card_frequency,
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
            "current_bet_value": self.current_bet,
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