"""Visualizer for the Parabellum environment"""

import jax.numpy as jnp
import jax
import darkdetect
import pygame
from moviepy.editor import ImageSequenceClip
from typing import Optional
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.viz.visualizer import SMAXVisualizer

# default dict
from collections import defaultdict


# constants

action_to_symbol = {0: "↑", 1: "→", 2: "↓", 3: "←", 4: "Ø"}


class ParabellumVisualizer(SMAXVisualizer):
    def __init__(self, env: MultiAgentEnv, state_seq, reward_seq=None):
        super().__init__(env, state_seq, reward_seq)
        self.bg = (0, 0, 0) if darkdetect.isDark() else (255, 255, 255)
        self.fg = (235, 235, 235) if darkdetect.isDark() else (20, 20, 20)
        self.s = 1000
        self.scale = self.s / self.env.map_width

    def render_agents(self, screen, state):
        time_tuple = zip(state.unit_positions, state.unit_teams, state.unit_types)
        for idx, (pos, team, kind) in enumerate(time_tuple):
            face_col = self.fg if int(team.item()) == 0 else self.bg
            pos = tuple((pos * self.scale).tolist())

            # draw the agent
            pygame.draw.circle(screen, face_col, pos, 7)
            pygame.draw.circle(screen, self.fg, pos, 7, 2)

            # draw the sight range
            # sight_range = self.env.unit_type_sight_ranges[kind] * self.scale
            # pygame.draw.circle(screen, self.mg, pos, sight_range.astype(int), 1)

            # draw attack range
            attack_range = self.env.unit_type_attack_ranges[kind] * self.scale
            pygame.draw.circle(screen, self.fg, pos, attack_range.astype(int), 2)

    def render_action(self, screen, actions):
        for idx, (_, v) in enumerate(actions.items()):
            symb = action_to_symbol.get(v.astype(int).item(), "Ø")
            font = pygame.font.SysFont("Fira Code", jnp.sqrt(self.s).astype(int).item())
            text = font.render(symb, True, self.fg)
            coord = (
                self.s / 2
                + ((idx - len(actions) / 2) * self.env.map_width / 2)
                + self.scale,
                (self.s / 20),
            )
            screen.blit(text, coord)

    def render_obstacles(self, screen, state):
        for c, d in zip(self.env.obstacle_coords, self.env.obstacle_deltas):
            d = tuple(((c + d) * self.scale).tolist())
            c = tuple((c * self.scale).tolist())
            pygame.draw.line(screen, self.fg, c, d, 2)

    def animate(self, save_fname: str = "parabellum.mp4"):
        if not self.have_expanded:
            self.expand_state_seq()
        frames = []  # frames for the video
        pygame.init()  # initialize pygame
        for _, state, action in self.state_seq:  # for every time step
            screen = pygame.Surface((self.s, self.s))  # clear the screen
            screen.fill(self.bg)  # fill the screen with the background color

            self.render_agents(screen, state)  # render the agents
            self.render_action(screen, action)  # render the actions
            self.render_obstacles(screen, state)  # render the obstacles

            # draw bullets
            # for bullet in state.bullets:
            #     pos = (int(bullet[0] * self.scale), int(bullet[1] * self.scale))
            #     pygame.draw.circle(screen, self.fg, pos, 3)

            # draw 4 black rectangles in the padding to cover up overflow of units
            rect = (0, 0, self.s, self.s)
            pygame.draw.rect(screen, self.fg, rect, 2)

            # rotate the screen and append to frames
            frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1))

        # save the images
        clip = ImageSequenceClip(frames, fps=48)
        clip.write_videofile(save_fname, fps=48)
        # clip.write_gif(save_fname.replace(".mp4", ".gif"), fps=24)
        pygame.quit()


# test the visualizer
if __name__ == "__main__":
    from jaxmarl import make
    from jax import random, numpy as jnp

    env = make("parabellum", map_width=128, map_height=128)
    rng, key = random.split(random.PRNGKey(0))
    obs, state = env.reset(key)
    state_seq = []
    for step in range(100):
        rng, key = random.split(rng)
        key_act = random.split(key, len(env.agents))
        action = {
            agent: env.action_space(agent).sample(key_act[i])
            for i, agent in enumerate(env.agents)
        }
        state_seq.append((key, state, action))
        obs, state, reward, done, info = env.step(key, state, action)
    vis = ParabellumVisualizer(env, state_seq)
    vis.animate()
