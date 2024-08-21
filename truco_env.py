import gym
from gym import spaces
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from gymnasium.error import DependencyNotInstalled

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
        # Cria a tabela de pontuações das cartas
        self.map = self._map_cards()
        # Contador de mãos jogadas
        self.turn = 0
        # Placar 0=oponente, 1=agente
        self.score = [0,0]
        # Aleatoriza quem começa (0=oponente, 1=agente)
        self.first_player = random.randint(0, 1)
        # Definindo o espaço de ação (0, 1, 2 representam as cartas na mão do agente)
        self.action_space = spaces.Discrete(3)
        # Definindo o espaço de observação (carta jogada pelo oponente, cartas na mão do agente, estado da primeira mão)
        self.observation_space = spaces.Tuple((
        spaces.Discrete(15),  # Carta jogada pelo oponente, 0 se for a vez do agente
        spaces.MultiDiscrete([15]*3),  # Cartas na mão do agente (0 representa carta jogada)
        spaces.Discrete(4)  # Estado da primeira mão (0 - essa é a primeira mão, 1 - oponente ganhou, 2 - empate, 3 - agente ganhou)
        ))
        # Variáveis de estado
        self.opponent_card = 0 if self.first_player == 1 else self.draw() # Carta jogada pelo oponente (nenhuma ou aleatório)
        self.agent_cards = np.sort([self.draw(), self.draw(), self.draw()])
        # Agente compra 3 cartas
        self.first_hand_winner = 0  # Estado da primeira mão

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

    def draw(self):
        # Compra uma carta do deck
        if self.deck:
            card_index = random.randint(0, len(self.deck) - 1)
            card = self.deck.pop(card_index)
            return card
        else:
            return None

    def step(self, action):
        # Verifica se a ação é válida (0, 1 ou 2)
        if action not in [0, 1, 2]:
            raise ValueError("Invalid action. Action must be 0, 1, or 2.")

        # Executa a ação (joga uma carta)
        player_card = self.agent_cards[action]
        self.agent_cards[action] = 'x'  # Marca a carta como jogada

        # Se não tiver ação do oponente, joga uma carta aleatória para ele
        if self.opponent_card == 'x':
            self.opponent_card = self.draw()

        # Se alguém não jogou, levanta erros (não é para acontecer)
        if self.opponent_card == 'x':
            raise ValueError("Opponent has no card set.")
        if player_card == 'x':
            raise ValueError("Player has no card set.")

        # Determina o vencedor da mão
        hand_winner = self._determine_hand_winner(self.opponent_card, player_card)

        # Determina o vencedor da rodada, se existir e atualiza o placar
        round_winner = self._determine_round_winner(hand_winner)
        if round_winner == 1 or round_winner == 3:
            self.score[0 if round_winner == 1 else 1] += 1

        # Determina quem ganhou a primeira mão se estiver nela
        if self.turn == 0:
            self.first_hand_winner = hand_winner

        # Define quem começa jogando a próxima mão (mantém em caso de empate, senão quem ganhou começa a próxima)
        if hand_winner != 2:
            self.first_player = min(hand_winner - 1, 1)

        # Se o oponente começar, ele joga uma carta aleatória, senão ele começa sem carta
        if self.first_player == 0:
            self.opponent_card = self.draw()
        else:
            self.opponent_card = 'x'

        # Avança o turno
        self.turn += 1

        # Determina a recompensa (0 para empates ou rodada inacabada, +1 vitória, -1 derrota)
        reward = 0 if round_winner == 0 else round_winner - 2

        # Sort na mão do agente
        self.agent_cards = np.sort(self.agent_cards)

        # Retorna a observação, a recompensa (-1, 0 ou 1) se a rodada acabou ou 0 se a rodada não acabou e a flag de rodada acabada
        observation = (self.map[self.opponent_card], np.array([self.map[card] for card in self.agent_cards]), self.first_hand_winner)
        if round_winner != 0:
            self.reset(reset_score=False)
        done = False if 12 not in self.score else True
        return {'observation' : observation, 'reward' : reward, 'done' : done}

    def reset(self, seed = None, reset_score = True):
        # Reseta o ambiente
        super().reset(seed=seed)
        self.deck =  self._create_deck()
        self.turn = 0
        if reset_score:
            self.score = [0,0]
        self.first_player = random.randint(0, 1)
        self.opponent_card = 'x' if self.first_player == 1 else self.draw()
        self.agent_cards = np.sort([self.draw(), self.draw(), self.draw()])
        self.first_hand_winner = 0
        return {'observation' : (self.map[self.opponent_card], np.array([self.map[card] for card in self.agent_cards]), self.first_hand_winner)}

    def _determine_hand_winner(self, opponent_card, agent_card):
        # Lógica para determinar o vencedor de uma mão (1=oponente ganha 2=empate 3=agente ganha)
        if self.map[agent_card] > self.map[opponent_card]:
          return 3
        if self.map[agent_card] < self.map[opponent_card]:
          return 1
        return 2

    def _determine_round_winner(self, hand_winner):
        # Lógica para determinar o vencedor de uma rodada (0=indeterminado 1=oponente ganha 2=empate 3=agente ganha)
        if self.turn == 2 or self.first_hand_winner == 2: # Terceiro turno ou primeira mão empatou
            return hand_winner
        if self.first_hand_winner == 1 and hand_winner != 3: # Oponente ganha primeira mão
            return 1
        if self.first_hand_winner == 3 and hand_winner != 1: # Agente ganha primeira mão
            return 3
        return 0 # Default

    def render(self, render_mode="rgb_array"):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pygame`"
            )

        agent_cards = self.agent_cards
        other_card = self.opponent_card
        score_str = f"{self.score[1]} x {self.score[0]}"
        first_hand_winner = self.first_hand_winner


        num_players = 2
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
            f"Player's team {score_str} Opponent's team", True, white
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
            if not other_card == 'x':
                card_img = scale_card_img(
                    get_image(
                        os.path.join(
                            "img",
                            f"{other_card}.png",
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

        first_round_str = "Loss" if first_hand_winner == 1 else "Win" if first_hand_winner == 3 else "Draw" if first_hand_winner == 2 else " "
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


        player_text = small_font.render("Player's hand", True, white)
        self.screen.blit(
            player_text, (spacing, team2_text_rect.bottom + 3.0 * spacing)
        )

        num_cards = sum([1 for card in agent_cards if card != 'x'])
        for idx in range(num_cards):
            card_img = scale_card_img(
                get_image(
                    os.path.join(
                        "img",
                        f"{agent_cards[idx]}.png",
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

# def test():
#    # Testes no ambiente
#     truco = TrucoMineiroEnv()
#     observation = truco.reset()
#     done = False
#     total_reward = 0

#     while not done:
#         print(f"Observação (opponent_card, agent_cards[], first_hand_winner): {observation}")
#         print(f"Cartas do agente: {[dict_deck[card_value] for card_value in truco.agent_cards]}")
#         if dict_deck[truco.opponent_card] == 'Carta indisponível':
#             print(f"Carta do oponente: ele joga depois")
#         else:
#             print(f"Carta do oponente: {dict_deck[truco.opponent_card]}")
#         while True:
#             action = random.randint(0, 2)  # Escolhe uma ação aleatória (trocar pelo agente)
#             if truco.agent_cards[action] != 0:
#                 break
#         print(f"Carta jogada pelo agente: {dict_deck[truco.agent_cards[action]]}")
#         result = truco.step(action)
#         observation, reward, done = result['observation'], result['reward'], result['done']
#         total_reward += reward

#         print(f"Recompensa obtida neste passo: {reward}")
#         print(f"Recompensa acumulada: {total_reward}")
#         print(f"Placar: Agente {truco.score[1]} x {truco.score[0]} Oponente\n")

# if __name__ == "__main__":
#     test()