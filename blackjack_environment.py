from enum import Enum, unique
from gym import Env, spaces
from typing import Union
import numpy as np


@unique
class Action(Enum):
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3


@unique
class Card(Enum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10


@unique
class HandState(Enum):
    INIT = 0
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6
    H7 = 7
    H8 = 8
    H9 = 9
    H10 = 10
    H11 = 11
    H12 = 12
    H13 = 13
    H14 = 14
    H15 = 15
    H16 = 16
    H17 = 17
    H18 = 18
    H19 = 19
    H20 = 20
    H21 = 21
    BUST = 22
    S11 = 33
    S12 = 34
    S13 = 35
    S14 = 36
    S15 = 37
    S16 = 38
    S17 = 39
    S18 = 40
    S19 = 41
    S20 = 42
    S21 = 43


class Deck:
    def __init__(self, num_deck: int) -> None:
        self.num_deck: int = num_deck
        self.cards: dict[Card, int] = None
        self.hole_card: Card = None

    def draw(self, is_hole_card: bool = False) -> Card:
        card: Card = np.random.choice(
            list(self.cards.keys()),
            p=self.probability,
        )
        if is_hole_card:
            self.hole_card: Card = card
        else:
            self.cards[card] -= 1
        return card

    def open_hole_card(self) -> None:
        if self.hole_card is not None:
            self.cards[self.hole_card] -= 1
            self.hole_card = None

    def reset(self) -> None:
        num_card: int = self.num_deck * 4
        self.cards = {}
        for card in Card:
            if card != Card.TEN:
                self.cards[card] = num_card
            else:
                self.cards[card] = num_card * 4
        self.hole_card = None

    @property
    def info(self) -> dict:
        info: dict = {
            'cards': self.cards,
            'probability': self.probability,
            'hole_card': self.hole_card,
        }
        return info

    @property
    def num_all(self) -> int:
        if self.cards is None:
            return 0
        return sum(self.cards.values())

    @property
    def probability(self) -> np.ndarray:
        num_cards: list[int] = list(self.cards.values())
        return np.array(num_cards) / sum(num_cards)


class Player:
    def __init__(self) -> None:
        self.hand_state: HandState = None

    def add(self, card: Card) -> None:
        point: int = self.point
        is_soft: bool = self.is_soft

        point += card.value
        if card == Card.ACE and point <= 11:
            point += 10
            is_soft = True
        elif is_soft and point > 21:
            point -= 10
            is_soft = False
        point = min(point, 22)

        self.hand_state = HandState(point + 22 * int(is_soft))

    def reset(self) -> None:
        self.hand_state = HandState.INIT

    @property
    def info(self) -> dict:
        info: dict = {
            'hand_state': self.hand_state,
            'point': self.point,
            'is_soft': self.is_soft,
        }
        return info

    @property
    def is_bust(self) -> bool:
        return self.hand_state == HandState.BUST

    @property
    def is_soft(self) -> bool:
        return self.hand_state.value > HandState.BUST.value

    @property
    def point(self) -> int:
        if self.hand_state == HandState.INIT:
            return 0
        return (self.hand_state.value - 1) % 22 + 1


class Dealer(Player):
    def __init__(self) -> None:
        super(Dealer, self).__init__()
        self.up_card: Card = None
        self.hole_card: Card = None

    def add(self, card: Card) -> None:
        super(Dealer, self).add(card)

        if self.up_card is None:
            self.up_card = card
        elif self.hole_card is None:
            self.hole_card = card

    def reset(self) -> None:
        super(Dealer, self).reset()
        self.up_card = None
        self.hole_card = None

    @property
    def info(self) -> dict:
        info: dict = {
            'hand_state': self.hand_state,
            'point': self.point,
            'is_soft': self.is_soft,
            'up_card': self.up_card,
            'hole_card': self.hole_card,
        }
        return info


class BlackjackEnvironment(Env):
    def __init__(self, num_deck: Deck = 8) -> None:
        self.deck: Deck = Deck(num_deck)
        self.player: Player = Player()
        self.dealer: Dealer = Dealer()
        self.action_space: spaces.Discrete = spaces.Discrete(3)
        self.observation_space: spaces.Dict = spaces.Dict({
            'player_point': spaces.Discrete(19, start=2),
            'player_is_soft': spaces.Discrete(2),
            'dealer_open_card': spaces.Discrete(10, start=1),
        })

    def judge(self, is_doubled: bool) -> float:
        if self.player.is_bust:
            return - (1.0 + float(is_doubled))
        if self.dealer.is_bust or self.player.point > self.dealer.point:
            return 1.0 + float(is_doubled)
        if self.player.point < self.dealer.point:
            return - (1.0 + float(is_doubled))
        return 0.0

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        assert self.action_space.contains(action)

        if action == Action.HIT.value or action == Action.DOUBLE.value:
            card: Card = self.deck.draw()
            self.player.add(card)

        if action == Action.STAND.value or action == Action.DOUBLE.value \
                or self.player.is_bust:
            self.deck.open_hole_card()
            while self.dealer.point < 17:
                card: Card = self.deck.draw()
                self.dealer.add(card)
            reward: float = self.judge(action == Action.DOUBLE.value)
            is_done: bool = True
        else:
            reward: float = 0.0
            is_done: bool = False

        return self.observation, reward, is_done, self.info

    def reset(
        self,
        seed: int = None,
        return_info: bool = False,
        options: dict = None,
    ) -> Union[dict, tuple[dict, dict]]:
        super().reset(seed=seed)
        if self.deck.num_all < self.deck.num_deck * 26:
            self.deck.reset()
        self.player.reset()
        self.dealer.reset()

        card: Card = self.deck.draw()
        self.player.add(card)
        card: Card = self.deck.draw()
        self.dealer.add(card)
        card: Card = self.deck.draw()
        self.player.add(card)
        card: Card = self.deck.draw(is_hole_card=True)
        self.dealer.add(card)

        if return_info:
            return self.observation, self.info
        return self.observation

    @property
    def info(self) -> dict:
        info: dict = {
            'deck': self.deck.info,
            'player': self.player.info,
            'dealer': self.dealer.info,
        }
        return info

    @property
    def observation(self) -> dict:
        observation: dict = {
            'player_point': self.player.point,
            'player_is_soft': self.player.is_soft,
            'dealer_open_card': self.dealer.up_card.value,
        }
        return observation
