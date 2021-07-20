import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np
import gym
from gym.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from gym.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn
import cv2
import manual_control

# TODO: rewrite step and reward functions, integrated with the GS output
# TODO: write policy function!!

_image_library = {}


def get_image(path):
    from os import path as os_path
    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + '/' + path)
    return image


def env(**kwargs):
    env = raw_env(**kwargs)
    if env.continuous:
        env = wrappers.ClipOutOfBoundsWrapper(env)
    else:
        env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):

    metadata = {'render.modes': ['human', "rgb_array"], 'name': "gymEnvSSM"}

    def __init__(self, n_elements=256, local_ratio=0, time_penalty=-0.1, continuous=True, max_cycles=125):
        EzPickle.__init__(self, n_elements, local_ratio, time_penalty, continuous, max_cycles)
        self.n_elements = n_elements
        im = cv2.imread('body.png')
        h,w,c = im.shape
        self.element_body_height = h
        im = cv2.imread('element.png')
        h,w,c = im.shape
        self.element_head_height = 0
        self.element_height = h
        self.element_width = w
        self.element_radius = 0
        self.wall_width = w
        # self.screen_width = (2 * self.wall_width) + (self.element_width * self.n_elements)
        im = cv2.imread('background.png')
        h,w,c = im.shape
        self.screen_width = w
        self.screen_height = h
        obs_height = 855 - 472
        obs_width = 625 - 37


        assert self.element_width == self.wall_width, "Wall width and element width must be equal for observation calculation"
        assert self.n_elements >= 1, "n_elements must be greater than 1"

        self.agents = ["element_" + str(r) for r in range(self.n_elements)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_elements))))
        self._agent_selector = agent_selector(self.agents)

        self.observation_spaces = dict(
            zip(self.agents, [gym.spaces.Box(low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8)] * self.n_elements))
        self.continuous = continuous
        if self.continuous:
            self.action_spaces = dict(zip(self.agents, [gym.spaces.Box(low=-1, high=1, shape=(1,))] * self.n_elements))
        else:
            self.action_spaces = dict(zip(self.agents, [gym.spaces.Discrete(3)] * self.n_elements))
        self.state_space = gym.spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)


        pygame.init()
        pymunk.pygame_util.positive_y_is_up = False

        self.renderOn = False
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.max_cycles = max_cycles

        self.element_sprite = get_image('element.png')
        self.element_body_sprite = get_image('body.png')
        self.background = get_image('background.png')

        self.elementList = []
        self.elementPosVert = []  #Keeps track of vertical positions of elements
        self.elementRewards = []     # Keeps track of individual rewards
        self.elementGroupings = range(self.n_elements) # Keeps track of which grouping each element belongs to
        self.recentFrameLimit = 20  # Defines what "recent" means in terms of number of frames.
        self.recentelements = set()  # Set of elements that have touched the ball recently
        self.time_penalty = time_penalty
        self.local_ratio = local_ratio

        self.done = False

        self.pixels_per_position = 1
        self.n_element_positions = self.element_height

        self.screen.fill((0, 0, 0))
        self.draw_background()
        self.screen.blit(self.background, (0, 0))

        self.render_rect = pygame.Rect(
            self.wall_width,   # Left
            self.wall_width,   # Top
            self.screen_width - (2 * self.wall_width),                              # Width
            self.screen_height - (2 * self.wall_width) - self.element_body_height    # Height
        )

        self.frames = 0

        self.has_reset = False
        self.closed = False
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        observation = pygame.surfarray.pixels3d(self.screen)
        i = self.agent_name_mapping[agent]
        # values based on the background, the observation space includes all of the elements! it's a well-connected graph
        x_low = 37
        x_high = 625
        y_low = 472
        y_high = 855
        cropped = np.array(observation[x_low:x_high, y_low:y_high, :])
        observation = np.rot90(cropped, k=3)
        observation = np.fliplr(observation)
        return observation

    def state(self):
        '''
        Returns an observation of the global environment
        '''
        state = pygame.surfarray.pixels3d(self.screen).copy()
        state = np.rot90(state, k=3)
        state = np.fliplr(state)
        return state

    def enable_render(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.renderOn = True
        # self.screen.blit(self.background, (0, 0))
        self.draw_background()
        self.draw()

    def close(self):
        if not self.closed:
            self.closed = True
            if self.renderOn:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
                self.renderOn = False
                pygame.event.pump()
                pygame.display.quit()

    def add_walls(self):
        top_left = (self.wall_width, self.wall_width)
        top_right = (self.screen_width - self.wall_width, self.wall_width)
        bot_left = (self.wall_width, self.screen_height - self.wall_width)
        bot_right = (self.screen_width - self.wall_width, self.screen_height - self.wall_width)
        walls = [
            pymunk.Segment(self.space.static_body, top_left, top_right, 1),     # Top wall
            pymunk.Segment(self.space.static_body, top_left, bot_left, 1),      # Left wall
            pymunk.Segment(self.space.static_body, bot_left, bot_right, 1),     # Bottom wall
            pymunk.Segment(self.space.static_body, top_right, bot_right, 1)     # Right
        ]
        for wall in walls:
            wall.friction = .64
            self.space.add(wall)

    def add_element(self, space, x, y):
        element = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        element.position = x, y
        segment = pymunk.Segment(element, (0, 0), (self.element_width - (2 * self.element_radius), 0), self.element_radius)
        segment.friction = .64
        segment.color = pygame.color.THECOLORS["blue"]
        space.add(element, segment)
        return element

    def move_element(self, element, v):

        def cap(y):
            maximum_element_y = self.elementPosVert[self.elementList.index(element)] - 3*self.element_height # self.screen_height - self.wall_width - (self.element_height - self.element_head_height)
            if y > maximum_element_y:
                y = maximum_element_y
            elif y < maximum_element_y - (self.n_element_positions * self.pixels_per_position):
                y = maximum_element_y - (self.n_element_positions * self.pixels_per_position)
            return y

        element.position = (element.position[0], cap(element.position[1] - v * self.pixels_per_position))

    def reset(self):
        self.space = pymunk.Space(threaded=False)
        self.add_walls()
        # self.space.threads = 2
        self.space.gravity = (0.0, 750.0)
        self.space.collision_bias = .0001
        self.space.iterations = 10  # 10 is default in PyMunk

        self.elementList = []
        self.elementPosVert = []
        for i in range(int(math.sqrt(self.n_elements))):
            for j in range(int(math.sqrt(self.n_elements))):
                possible_y_displacements = np.arange(0, self.pixels_per_position * self.n_element_positions,
                                                         self.pixels_per_position)
                if (j == 0):
                    elemPos = self.screen_height - 434 + (i * 0.53 * self.element_height)
                    maximum_element_y = elemPos - 3*self.element_height
                    element = self.add_element(
                        self.space,
                        self.screen_width / 2 - self.element_width / 3.9 - (self.element_width * 0.67 * i), #x position
                        maximum_element_y - self.np_random.choice(possible_y_displacements) #y position
                    )
                else:
                    elemPos = self.elementPosVert[len(self.elementPosVert) -1] + (0.59 * self.element_height)
                    maximum_element_y = elemPos - 3*self.element_height
                    element = self.add_element(
                        self.space,
                        self.elementList[len(self.elementPosVert) - 1].position[0] + (self.element_width*0.65),#x position
                        maximum_element_y - self.np_random.choice(possible_y_displacements)#y position
                    )
                element.velociy = 0 #TODO: fix typo
                self.elementList.append(element)
                self.elementPosVert.append(elemPos)

        self.draw_background()
        self.draw()

        self.agents = self.possible_agents[:]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.has_reset = True
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        self.frames = 0

    def draw_background(self):

        outer_walls = pygame.Rect(
            0,   # Left
            0,   # Top
            self.screen_width,      # Width
            self.screen_height,     # Height
        )
        outer_wall_color = (58, 64, 65)
        pygame.draw.rect(self.screen, outer_wall_color, outer_walls)
        inner_walls = pygame.Rect(
            self.wall_width / 2,   # Left
            self.wall_width / 2,   # Top
            self.screen_width - self.wall_width,      # Width
            self.screen_height - self.wall_width,     # Height
        )
        inner_wall_color = (68, 76, 77)
        pygame.draw.rect(self.screen, inner_wall_color, inner_walls)
        self.screen.blit(self.background, (0, 0))
        self.draw_elements()

    def draw_elements(self):
        element_color = (56, 129, 197)
        black = (0,0,0)
        i = 0
        for element in self.elementList:
            # Height is the size of the blue part of the element.
            height = self.screen_height - self.wall_width - self.element_body_height - (element.position[1] + self.element_radius) + (self.element_body_height)
            self.screen.blit(self.element_body_sprite, (
                element.position[0] + self.element_radius, self.elementPosVert[i]))
            self.screen.blit(self.element_body_sprite, (
                element.position[0] + self.element_radius , element.position[1] + self.element_radius))
            body_rect = pygame.Rect(
                element.position[0] + self.element_radius+1,    # +1 to match up to element graphics
                element.position[1] + self.element_radius + self.element_body_height/2,
                self.element_width - 2,
                height + self.element_body_height*1.2 - (self.screen_height - self.elementPosVert[i])
            )
            line1 = pygame.Rect(
                element.position[0] + self.element_radius +1,    # +1 to match up to element graphics
                element.position[1] + self.element_radius + self.element_body_height/2,
                1,
                height + self.element_body_height*1.2 - (self.screen_height - self.elementPosVert[i])
            )
            line2 = pygame.Rect(
                element.position[0] + self.element_radius + 12,  # +1 to match up to element graphics
                element.position[1] + self.element_radius + self.element_body_height / 2,
                1,
                height + self.element_body_height * 1.2 - (self.screen_height - self.elementPosVert[i])
            )
            line3 = pygame.Rect(
                element.position[0] + self.element_radius + self.element_width - 2,  # +1 to match up to element graphics
                element.position[1] + self.element_radius + self.element_body_height / 2,
                1,
                height + self.element_body_height * 1.2 - (self.screen_height - self.elementPosVert[i])
            )
            i += 1
            pygame.draw.rect(self.screen, element_color, body_rect)
            pygame.draw.rect(self.screen, black, line1)
            pygame.draw.rect(self.screen, black, line2)
            pygame.draw.rect(self.screen, black, line3)
            self.screen.blit(self.element_sprite, (element.position[0] - self.element_radius, element.position[1] - self.element_radius - self.element_height / 2.5))

    def draw(self):
        self.draw_background()

        self.draw_elements()

        font = pygame.font.SysFont(None, 12)
        white = (255, 255, 255)
        for element in self.elementList:
            grouping = font.render(str(self.elementGroupings[self.elementList.index(element)]), True, white)
            self.screen.blit(grouping,
                             (element.position[0] + self.element_width/2 - grouping.get_width()/2, element.position[1] - grouping.get_height()/2))

    def get_local_reward(self, prev_position, curr_position):
        local_reward = .5 * (prev_position - curr_position)
        return local_reward

    def render(self, mode="human"):
        if not self.renderOn:
            # sets self.renderOn to true and initializes display
            self.enable_render()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        pygame.display.flip()
        return np.transpose(observation, axes=(1, 0, 2)) if mode == "rgb_array" else None

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        action = np.asarray(action)
        agent = self.agent_selection
        if self.continuous:
            self.move_element(self.elementList[self.agent_name_mapping[agent]], action)
        else:
            self.move_element(self.elementList[self.agent_name_mapping[agent]], action - 1)

        self.space.step(1 / 20.0)
        if self._agent_selector.is_last():
            # ball_min_x = int(self.ball.position[0] - self.ball_radius)
            # if ball_min_x <= self.wall_width + 1:
            #     self.done = True
            self.draw()
            # local_reward = self.get_local_reward(self.lastX, ball_min_x)
            # Opposite order due to moving right to left
            global_reward = 0
            if not self.done:
                global_reward += self.time_penalty
            total_reward = [global_reward * (1 - self.local_ratio)] * self.n_elements  # start with global reward
            # local_elements_to_reward = self.get_nearby_elements()
            # for index in local_elements_to_reward:
            #     total_reward[index] += local_reward * self.local_ratio
            self.rewards = dict(zip(self.agents, total_reward))
            # self.lastX = ball_min_x
            self.frames += 1
        else:
            self._clear_rewards()

        if self.frames >= self.max_cycles:
            self.done = True
        # Clear the list of recent elements for the next reward cycle
        if self.frames % self.recentFrameLimit == 0:
            self.recentelements = set()
        if self._agent_selector.is_last():
            self.dones = dict(zip(self.agents, [self.done for _ in self.agents]))

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()