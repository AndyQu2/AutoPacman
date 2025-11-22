from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType, ActType

from Pacman.constants import SCREENWIDTH, SCREENHEIGHT, SCREENSIZE, UP, DOWN, LEFT, RIGHT, FREIGHT, set_rl_environment, \
    SCATTER
from Pacman.game_controller import GameController


class PacmanEnvironment(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(PacmanEnvironment, self).__init__()

        self.game_controller = GameController()
        self.game_controller.startGame()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.window = None
        self.clock = pygame.time.Clock()
        self._last_score = 0

        self._last_ghost_distances = np.ones(4) * 1000
        self._last_ghost_modes = [SCATTER] * 4

        set_rl_environment(True)

    def _get_obs(self):
        obs = np.zeros(12, dtype=np.float32)

        obs[0] = self.game_controller.pacman.position.x / SCREENWIDTH
        obs[1] = self.game_controller.pacman.position.y / SCREENWIDTH

        ghosts = [self.game_controller.ghosts.blinky,
                  self.game_controller.ghosts.pinky,
                  self.game_controller.ghosts.inky,
                  self.game_controller.ghosts.clyde]

        for i, ghost in enumerate(ghosts):
            obs[2 + i * 2] = ghost.position.x / SCREENWIDTH
            obs[3 + i * 2] = ghost.position.y / SCREENHEIGHT

        if self.game_controller.fruit is not None:
            obs[10] = self.game_controller.fruit.position.x / SCREENWIDTH
            obs[11] = self.game_controller.fruit.position.y / SCREENHEIGHT
        else:
            obs[10] = 0.0
            obs[11] = 0.0

        return obs

    def _get_info(self):
        return {
            'score': self.game_controller.score,
            'lives': self.game_controller.lives,
            'level': self.game_controller.level,
            'pellets_eaten': self.game_controller.pellets.numEaten,
            'pellets_remaining': len(self.game_controller.pellets.pelletList)
        }

    def _is_terminated(self):
        return (self.game_controller.lives <= 0 or
                (self.game_controller.pellets.isEmpty() and
                 self.game_controller.level >= 10))

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        elif self.render_mode == 'human':
            self._render_frame()
        return None

    def _render_frame(self):
        self.game_controller.render()

        if self.render_mode == 'human':
            pygame.display.flip()
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    @staticmethod
    def _calculate_chase_reward(distances, modes):
        reward = 0.0

        for i, (distance, mode) in enumerate(zip(distances, modes)):
            if mode == FREIGHT:
                if distance < 100:
                    chase_reward = 5.0 * (1.0 - distance / 100)
                    reward += chase_reward

                if distance < 50:
                    reward += 3.0

        min_freight_distance = min([dist for dist, mode in zip(distances, modes) if mode == FREIGHT], default=1000)
        if min_freight_distance < 80:
            reward += 2.0

        return reward

    @staticmethod
    def _calculate_avoid_reward(distances, modes):
        reward = 0.0

        min_danger_distance = min([dist for dist, mode in zip(distances, modes) if mode != FREIGHT], default=1000)
        avg_danger_distance = np.mean([dist for dist, mode in zip(distances, modes) if mode != FREIGHT])

        safe_distance = 60
        if min_danger_distance > safe_distance:
            safety_reward = min(3.0, (min_danger_distance - safe_distance) / 20)
            reward += safety_reward

        if avg_danger_distance > 80:
            reward += 1.0

        if min_danger_distance > 100 and avg_danger_distance > 120:
            reward += 2.0

        return reward

    def _get_ghost_info(self):
        pacman_pos = self.game_controller.pacman.position
        ghosts = [self.game_controller.ghosts.blinky,
                  self.game_controller.ghosts.pinky,
                  self.game_controller.ghosts.inky,
                  self.game_controller.ghosts.clyde]

        distances = []
        modes = []
        positions = []

        for ghost in ghosts:
            ghost_pos = ghost.position
            distance = ((pacman_pos.x - ghost_pos.x) ** 2 +
                        (pacman_pos.y - ghost_pos.y) ** 2) ** 0.5
            distances.append(distance)
            modes.append(ghost.mode.current)
            positions.append(ghost_pos)

        return np.array(distances), modes, positions

    def _calculate_reward(self):
        reward = 0.0

        score_diff = self.game_controller.score - self._last_score
        reward += score_diff
        self._last_score = self.game_controller.score

        if score_diff > 0:
            reward += 1.0

        if self.game_controller.fruit is not None and self.game_controller.pacman.collideCheck(self.game_controller.fruit):
            reward += 15.0

        current_ghost_distances, current_ghost_modes, _ = self._get_ghost_info()
        current_ghost_distances = np.array(current_ghost_distances)

        freight_mask = np.array([mode == FREIGHT for mode in current_ghost_modes])
        danger_mask = ~freight_mask

        if np.any(freight_mask):
            freight_distances = current_ghost_distances[freight_mask]
            if len(freight_distances) > 0:
                close_freight = freight_distances < 100
                if np.any(close_freight):
                    distance_ratios = 1.0 - freight_distances[close_freight] / 100
                    reward += np.sum(5.0 * distance_ratios)

                very_close = freight_distances < 50
                if np.any(very_close):
                    reward += 3.0 * np.sum(very_close)

        if np.any(danger_mask):
            danger_distances = current_ghost_distances[danger_mask]
            min_danger_distance = np.min(danger_distances) if len(danger_distances) > 0 else 1000
            avg_danger_distance = np.mean(danger_distances) if len(danger_distances) > 0 else 1000

            if min_danger_distance > 60:
                reward += min(3.0, (min_danger_distance - 60) / 20)

        distance_changes = current_ghost_distances - self._last_ghost_distances
        weights = np.where(freight_mask, -0.1, 0.1)
        reward += np.sum(distance_changes * weights)

        min_distance = np.min(current_ghost_distances)
        if min_distance < 30:
            min_idx = np.argmin(current_ghost_distances)
            if current_ghost_modes[min_idx] != FREIGHT:
                reward -= 20.0 * (1.0 - min_distance / 30)
            else:
                reward += 10.0 * (1.0 - min_distance / 30)

        reward += 0.01

        if not self.game_controller.pacman.alive:
            reward -= 5.0

        if self.game_controller.pellets.isEmpty():
            reward += 50.0

        self._last_ghost_distances = current_ghost_distances
        self._last_ghost_modes = current_ghost_modes

        return reward

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.game_controller.restartGame()
        self.game_controller.pause.paused = False
        self.game_controller.textgroup.hideText()
        self.game_controller.showEntities()
        self._last_score = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        direction_map = {
            0: UP,
            1: DOWN,
            2: LEFT,
            3: RIGHT
        }
        direction = direction_map[action]

        self.game_controller.update_manually(direction)
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = False
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, truncated, info