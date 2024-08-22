# Imports
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
import random

from truco_players import LearningPlayer, NonLearningPlayer

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

    def __init__(self, num_players, teams):
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
        # Cria a tabela de pontuações das cartas
        self.map = self._map_cards()
        # Contador de mãos jogadas
        self.turn = 0
        # Placar [jogador1, jogador2]
        self.game_score = [0, 0]
        self.round_score = [0, 0]
        # Cria agentes
        self.num_players = num_players
        self.teams = None
        self.players = [None for _ in range(num_players)]
        self.has_learning_player = None
        self.set_players(teams)
        self.cards = [[] for _ in range(num_players)]
        # Aleatoriza quem começa
        self.round_starter = random.randint(0, num_players - 1)
        self.current_player_index = self.round_starter
        self.other_player_index = 1 - self.current_player_index # conferir isso depois pro n v n
        self.current_card = 'x'
        self.other_card = 'x'
        self.first_hand_winner = 0
        self.hand_winner = 0
        # Frequência de cartas jogadas no round
        self.card_frequency = np.array(14 * [0])
        # Mecânica de truco
        self.current_bet = 2
        self.trucable = [True, True] # Se é trucável/aumentável
        self.respond = False
        self.round_ended = False
        # Inicializa cartas
        self.reset()

    def set_players(self, teams):
        self.teams = teams
        num_learning_players = 0
        for i in range(self.num_players // 2):
            if teams[0][i].type == LearningPlayer: num_learning_players += 1
            self.players[2 * i] = teams[0][i]
            if teams[1][i].type == LearningPlayer: num_learning_players += 1
            self.players[2 * i + 1] = teams[1][i]

        if num_learning_players == 0:
            self.has_learning_player = False
        elif num_learning_players == 1:
            self.has_learning_player = True
        else:
            raise Exception("There cannot be more than 1 learning player!")

    def _create_deck(self):
       # Retorna uma lista embaralhada de cartas
        suits = ['spades', 'hearts', 'diamonds', 'clubs']
        ranks = ['ace', '2', '3', '4', '5', '6', '7', 'jack', 'queen', 'king']
        deck = []
        for suit in suits:
            for rank in ranks:
                deck.append(f'{suit}_{rank}')
        random.shuffle(deck)
        return deck

    def _map_cards(self):
        default_mapping = {
            '3': 10,
            '2': 9,
            'ace': 8,
            'king': 7,
            'jack': 6,
            'queen': 5,
            '7': 4,
            '6': 3,
            '5': 2,
            '4': 1
        }

        card_points_map = {
            'x': 0, # carta já jogada
            'clubs_4': 14,
            'hearts_7': 13,
            'spades_ace': 12,
            'diamonds_7': 11
        }
        for card in self.deck:
            if card not in card_points_map:
                card_points_map[card] = default_mapping[card.split('_')[-1]]

        return card_points_map

    def _draw_cards(self):
        for i in range(self.num_players):
            self.cards[i] = np.sort(
                [self.deck.pop(random.randint(0, len(self.deck) - 1)) for _ in range(3)]
            )

    def reset(self, reset_score=True):
        if self.players[0] == None: raise Exception("Players must be set before calling reset!")
        self.deck = self._create_deck()
        self._draw_cards()
        self.round_starter = 1 - self.round_starter
        self.current_player_index = self.round_starter
        self.other_player_index = 1 - self.current_player_index
        self.current_card = 'x'
        self.other_card = 'x'
        self.turn = 0
        if reset_score:
            self.game_score = [0, 0]
        self.round_score = [0, 0]
        self.first_hand_winner = 0
        self.card_frequency *= 0
        self.current_bet = 2
        self.trucable = [True, True]
        self.respond = False
        self.round_ended = False
        if self.has_learning_player and self.players[self.current_player_index].type == NonLearningPlayer:
            self.handle_action(self.players[self.current_player_index].choose_action(self._get_obs(), self._get_info()))
        return self._get_obs(), self._get_info()

    def step(self, action):
        if not self.has_learning_player: raise Exception("step method cannot be used without a learning player!")
        # Processa a ação do agente
        obs, reward, done, info = self.handle_action(action)
        # Estimula e processa as ações dos demais jogadores (SUPORTE PARA APENAS 1v1 POR ENQUANTO)
        while not info["round_ended"] and self.players[self.current_player_index].type == NonLearningPlayer:
            obs, reward, done, info = self.handle_action(self.players[self.current_player_index].choose_action(obs, info))
        if info["round_ended"]:
            if self.players[self.current_player_index] == LearningPlayer:
                reward = -reward
            else:
                info["victory"] = not info["victory"]
        return obs, reward, done, info

    def handle_action(self, action):
        # por ora está:
        # obs e info relativos ao jogador depois do que executou a ação
        # reward relativo a quem executou a ação

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
        # Se aceita continua o jogo
        if action == 4:
            reward = 0
            done = False
        # Se recusa, atualiza o aposta, placar e distribui as recompensas
        else:
            if self.current_bet == 10:
                self.current_bet = 6
            else:
                self.current_bet -= 2
            reward = -self.current_bet
            self.game_score[self.other_player_index] += self.current_bet # atualiza o placar primeiro
            done = any(x >= 12 for x in self.game_score)
            self.round_ended = True
        self._switch_players()
        return self._get_obs(), reward, done, self._get_info()

    def handle_truco_call(self):
        '''
        Lógica para pedir truco ou aumentar aposta
        '''
        # Quando não pode pedir truco
        if self.trucable[self.current_player_index] == False:
            raise ValueError(f"Invalid action. Player is not allowed to truco/raise.")
        if self.current_bet >= 12:
            raise ValueError(f"Invalid action. Current bet is maxed at {self.current_bet}.")
        # Aumenta a aposta
        if self.current_bet == 6:
            self.current_bet = 10
        else:
            self.current_bet += 2
        # Current não pode mais pedir/aumentar truco
        self.trucable[self.current_player_index] = False
        # Se os dois já forem ganhar o jogo com a aposta atual, other não pode mais aumentar
        min_sum_score_bet = min([(self.current_bet + score) for score in self.game_score])
        if min_sum_score_bet >= 12:
            self.trucable[self.other_player_index] = False
        else:
            self.trucable[self.other_player_index] = True
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
        n_cards = sum(1 for x in self.cards[self.current_player_index] if x != 'x')
        possible_actions = range(0, n_cards)
        if action not in possible_actions:
            raise ValueError(f"Invalid action. Player tried to play an unavailable card.")

        # Executa a ação do jogador atual
        card_played = self.cards[self.current_player_index][action]
        self.cards[self.current_player_index][action] = 'x'  # Marca a carta como jogada
        self.current_card = card_played
        self.card_frequency[self.map[card_played] - 1] += 1

        # Sort na mão do player
        self.cards[self.current_player_index] = np.sort(
            self.cards[self.current_player_index]
        )

        # Se o outro jogador ainda não jogou, encerra a chamada
        if self.other_card == 'x':
            self._switch_players()
            return self._get_obs(), 0, False, self._get_info()

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
        self.other_card = 'x'
        self.current_card = 'x'

        # Avança o turno
        self.turn += 1

        # Determina a recompensa (0 para empates ou rodada inacabada, +1 vitória, -1 derrota)
        if round_winner == self.current_player_index + 1:
            reward = self.current_bet
        elif round_winner == self.other_player_index + 1:
            reward = -self.current_bet
        else:
            reward = 0

        # Troca os jogadores de lugar se precisar (mantém em caso de empate, senão quem ganhou começa a próxima)
        self._switch_players()
        if self.hand_winner == 1:
            self.current_player_index, self.other_player_index = 0, 1
        elif self.hand_winner == 2:
            self.current_player_index, self.other_player_index = 1, 0

        done = any(x >= 12 for x in self.game_score)

        # Retorna a observação, a recompensa (-1, 0 ou 1) se a rodada acabou ou 0 se a rodada não acabou e a flag de rodada acabada
        if round_winner != 0:
            self.round_ended = True

        return self._get_obs(), reward, done, self._get_info()

    def play(self):
        if self.has_learning_player: raise Exception("play method cannot be used with a learning player")
        while True:
            pass

    def _switch_players(self):
        self.current_player_index, self.other_player_index = self.other_player_index, self.current_player_index
        self.current_card, self.other_card = self.other_card, self.current_card

    def _determine_hand_winner(self, card1, card2):
        # Determina quem vence a mão (1=Player 1 ganha; 2=Player 2 ganha; 3=empate)
        if self.map[card1] > self.map[card2]:
            return self.current_player_index + 1  # Current ganha
        elif self.map[card2] > self.map[card1]:
            return self.other_player_index + 1  # Other jogador ganha
        return 3  # Empate

    def _get_obs(self):
        bet_dict = {2:0, 4:1, 6:2, 10:3, 12:4}
        return {
            "current_player_cards": [self.map[card] for card in self.cards[self.current_player_index]],
            "other_card": self.map[self.other_card],
            "first_hand_winner": self.first_hand_winner,
            "current_player_score": self.game_score[self.current_player_index],
            "other_player_score": self.game_score[self.other_player_index],
            "current_bet": bet_dict[self.current_bet],
            "trucable": self.trucable[self.current_player_index],
            "respond": self.respond,
            "card_frequency": self.card_frequency,
        }

    def _get_info(self):
        return {
            "current_player_cards": self.cards[self.current_player_index],
            "round_score": self.round_score,
            "game_score": self.game_score,
            "hand_winner": self.hand_winner - 1,
            "first_hand_winner": self.first_hand_winner,
            "current_bet_value": self.current_bet,
            "round_ended": self.round_ended,
            "valid_actions": self._determine_valid_actions(),
            "victory": self.game_score[self.current_player_index] >= 12,
        }

    def _determine_valid_actions(self):
        valid_actions = []
        if self.respond: valid_actions += [4, 5]
        else:
            if self.cards[self.current_player_index][0] != 'x': valid_actions += [0]
            if self.cards[self.current_player_index][1] != 'x': valid_actions += [1]
            if self.cards[self.current_player_index][2] != 'x': valid_actions += [2]
            if self.trucable[self.current_player_index]: valid_actions += [3]
        return valid_actions

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
            self.first_hand_winner == self.current_player_index + 1
            and self.hand_winner != self.other_player_index + 1
        ):  # Current ganha primeira mão
            return self.current_player_index + 1
        if (
            self.first_hand_winner == self.other_player_index + 1
            and self.hand_winner != self.current_player_index + 1
        ):  # Other ganha primeira mão
            return self.other_player_index + 1
        return 0  # Default

    def render(self, render_mode="rgb_array"):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        observation = self._get_obs()
        current_player_cards = self.cards[self.current_player_index]
        other_card = self.other_card
        respond = self.respond
        first_hand_winner = observation["first_hand_winner"]
        score = f"{observation['current_player_score']} x {observation['other_player_score']}"
        current_bet = observation["current_bet"]


        num_players = len(self.players)
        screen_width, screen_height = 900, 750
        card_img_height = 141
        card_img_width = 101
        logo_width = 54
        logo_height = 64
        spacing = 50

        bg_color = (7, 99, 36)
        white = (255, 255, 255)
        yellow = (255, 255, 51)

        if not hasattr(self, "screen"):
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        if not hasattr(self, "clock"):
            self.clock = pygame.time.Clock()

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(
            os.path.join("font", "Roboto-Black.ttf"), 35
        )

        score_text = small_font.render(
            f"Player's team {score} Opponent's team", True, white
        )
        score_text_rect = self.screen.blit(score_text, (screen_width // 2 - score_text.get_width() // 2, spacing // 4))

        def scale_card_img(card_img, shape=(card_img_width, card_img_height)):
            return pygame.transform.scale(card_img, shape)

        table_logo = scale_card_img(
            get_image(
                os.path.join(
                    "img",
                    f"turing_logo.png",
                )
            ),
            (logo_width, logo_height)
        )
        for idx in range(4):
            self.screen.blit(
                table_logo,
                (
                    (idx // 2) * screen_width + (1 - 2 * (idx // 2)) * spacing - logo_width // 2,
                    (idx % 2) * screen_height + (1 - 2 * (idx % 2)) * spacing - logo_height // 2,
                )
            )

        def calc_coord_x(num_cards, idx):
            if num_cards == 3:
                return screen_width // 2 - (3 - 2 * idx) * (card_img_width // 2) - (1 - idx) * spacing // 4
            elif num_cards == 2:
                return screen_width // 2 - (1 - idx) * (card_img_width) - (1 - 2 * idx) * spacing // 4
            else:
                return screen_width // 2 - card_img_width // 2

        other_team_text = small_font.render(
            "Other team:", True, white
        )
        other_team_text_rect = self.screen.blit(
            other_team_text, (spacing, score_text_rect.bottom + card_img_height // 2 + spacing - other_team_text.get_height() // 2)
            )

        for idx in range(num_players):
            # TODO: mudar para as cartas de todos do time adversário
            if not self.other_card == 'x':
                card_img = scale_card_img(
                    get_image(
                        os.path.join(
                            "img",
                            f"{self.other_card}.png",
                        )
                    )
                )
                self.screen.blit(
                    card_img,
                    (
                        calc_coord_x(num_cards=num_players/2, idx=idx),
                        score_text_rect.bottom + spacing,
                    ),
                )

        first_round_title = small_font.render(
            "First round:", True, white
        )
        first_round_title_rect = self.screen.blit(
            first_round_title,
            (
                screen_width - spacing - first_round_title.get_width(),
                score_text_rect.bottom + card_img_height // 2 + spacing - first_round_title.get_height()
            )
        )

        first_round_str = "Win" if first_hand_winner == self.current_player_index + 1 else "Loss" if first_hand_winner == self.other_player_index + 1 else "Draw" if self.turn == 2 else " "
        first_round_status = small_font.render(
            first_round_str, True, white
        )
        self.screen.blit(
            first_round_status,
            (
                screen_width - spacing - (first_round_title.get_width() // 2 + first_round_status.get_width() // 2),
                first_round_title_rect.bottom
            )
        )


        team2_text = small_font.render(
            "Player's team:", True, white
        )
        team2_text_rect = self.screen.blit(
            team2_text, (spacing, other_team_text_rect.bottom + card_img_height + spacing - team2_text.get_height() // 2)
            )

        #for idx in range(num_players):
            # TODO: mudar para as cartas do time


        current_bet_title = small_font.render(
            "Current bet:", True, white
        )
        current_bet_title_rect = self.screen.blit(
            current_bet_title,
            (
                screen_width - spacing - current_bet_title.get_width(),
                other_team_text_rect.bottom + card_img_height + spacing - current_bet_title.get_height()
            )
        )

        current_bet_list = ["None", "Truco", "6", "9", "12"]
        current_bet_status = small_font.render(
            f"{current_bet_list[current_bet]}", True, white
        )
        self.screen.blit(
            current_bet_status,
            (
                screen_width - spacing - (current_bet_title.get_width() // 2 + current_bet_status.get_width() // 2),
                current_bet_title_rect.bottom
            )
        )

        log_str = "Truco or raise called" if respond else ""
        log_text = small_font.render(log_str, True, yellow)
        log_text_rect = self.screen.blit(
            log_text, (screen_width - spacing - log_text.get_width(), team2_text_rect.bottom + 2.0 * spacing)
        )

        player_text = small_font.render("Player's hand", True, white)
        self.screen.blit(
            player_text, (spacing, log_text_rect.bottom + spacing)
        )



        num_cards = sum([1 for card in self.cards[self.current_player_index] if card != 'x'])
        for idx in range(num_cards):
            card_img = scale_card_img(
                get_image(
                    os.path.join(
                        "img",
                        f"{self.cards[self.current_player_index][idx]}.png",
                    )
                )
            )
            self.screen.blit(
                card_img,
                (
                    calc_coord_x(num_cards=num_cards, idx=idx),
                    team2_text_rect.bottom + 3.75 * spacing,
                ),
            )

        if render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()


def test_game():
    # Testes no ambiente
    truco = TrucoMineiroEnv()
    observation = truco.reset()
    done = False
    rewards = [0, 0]
    player1_reward = -1

    while not done:
        print(
                f"Mão {truco.turn + 1}/3 - Placar do round: {truco.players[0].name} ({truco.round_score[0]} x {truco.round_score[1]}) {truco.players[1].name}"
            )

        current_player = truco.players[truco.current_player_index]
        other_player = truco.players[truco.other_player_index]

        print(f"Vez do {current_player.name}")
        print(f"Observação: {observation}")
        print(
            f"Cartas do {current_player.name}: {[dict_deck[card] for card in current_player.cards]}"
        )

        # Escolhe uma ação aleatória para o jogador atual (trocar pelo agente RL)
        action = int(input())
        if action not in [0, 1, 2]:
            observation, reward, done, info = truco.step(action)
            if action == 3:
                if truco.current_bet == 4:
                    print(f"{current_player.name} pediu truco")
                else:
                    print(f"{current_player.name} pediu {truco.current_bet}")
            elif action == 4:
                print(f"{current_player.name} aceitou")
            else:
                print(f"{current_player.name} recusou")
            print(f"observation = {observation}")
            print(f"info = {info}")

        #action = current_player.choose_action()

        else:
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

        if player1_reward != 0:
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
