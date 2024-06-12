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
        self.action_seq = [  # seems there's an error in SMAX expand (this gets around that)
            action for _, _, action in state_seq
        ]

    def render_agents(self, screen, state):
        time_tuple = zip(state.unit_positions, state.unit_teams, state.unit_types)
        for idx, (pos, team, kind) in enumerate(time_tuple):
            face_col = self.fg if int(team.item()) == 0 else self.bg
            pos = tuple((pos * self.scale).tolist())

            # draw the agent
            radius = self.env.unit_type_radiuses[kind] * self.scale * 2
            pygame.draw.circle(screen, face_col, pos, radius.astype(int))
            pygame.draw.circle(screen, self.fg, pos, radius.astype(int), 1)

            # draw the sight range
            # sight_range = self.env.unit_type_sight_ranges[kind] * self.scale
            # pygame.draw.circle(screen, self.mg, pos, sight_range.astype(int), 1)

            # draw attack range
            attack_range = self.env.unit_type_attack_ranges[kind] * self.scale
            pygame.draw.circle(screen, self.fg, pos, attack_range.astype(int), 2)

    def render_action(self, screen, action):
        def coord_fn(idx, n, team):
            return (
                self.s / 20 if team == 0 else self.s - self.s / 20,
                # vertically centered so that n / 2 is above and below the center
                self.s / 2 - (n / 2) * self.s / 20 + idx * self.s / 20,
            )

        for idx in range(self.env.num_allies):
            symb = action_to_symbol.get(action[f"ally_{idx}"].astype(int).item(), "Ø")
            font = pygame.font.SysFont("Fira Code", jnp.sqrt(self.s).astype(int).item())
            text = font.render(symb, True, self.fg)
            coord = coord_fn(idx, self.env.num_allies, 0)
            screen.blit(text, coord)

        for idx in range(self.env.num_enemies):
            symb = action_to_symbol.get(action[f"enemy_{idx}"].astype(int).item(), "Ø")
            font = pygame.font.SysFont("Fira Code", jnp.sqrt(self.s).astype(int).item())
            text = font.render(symb, True, self.fg)
            coord = coord_fn(idx, self.env.num_enemies, 1)
            screen.blit(text, coord)

    def render_obstacles(self, screen):
        for c, d in zip(self.env.obstacle_coords, self.env.obstacle_deltas):
            d = tuple(((c + d) * self.scale).tolist())
            c = tuple((c * self.scale).tolist())
            pygame.draw.line(screen, self.fg, c, d, 2)

    def animate(self, save_fname: str = "parabellum.mp4"):
        if not self.have_expanded:
            self.expand_state_seq()
        frames = []  # frames for the video
        pygame.init()  # initialize pygame
        for idx, (_, state, _) in enumerate(self.state_seq):  # for every time step
            screen = pygame.Surface(
                (self.s, self.s), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            action = self.action_seq[idx // 8]
            # screen = pygame.Surface((self.s, self.s))  # clear the screen
            screen.fill(self.bg)  # fill the screen with the background color

            self.render_agents(screen, state)  # render the agents
            self.render_action(screen, action)  # render the actions
            self.render_obstacles(screen)  # render the obstacles

            # draw bullets
            # for bullet in state.bullets:
            #     pos = (int(bullet[0] * self.scale), int(bullet[1] * self.scale))
            #     pygame.draw.circle(screen, self.fg, pos, 3)

            # draw 4 black rectangles in the padding to cover up overflow of units
            rect = (0, 0, self.s, self.s)
            pygame.draw.rect(screen, self.fg, rect, 2)

            # rotate the screen and append to frames
            frames.append(pygame.surfarray.pixels3d(screen).swapaxes(0, 1))

        # save the images
        clip = ImageSequenceClip(frames, fps=60)
        clip.write_videofile(save_fname, fps=60)
        # clip.write_gif(save_fname.replace(".mp4", ".gif"), fps=24)
        pygame.quit()


# test the visualizer
if __name__ == "__main__":
    from jaxmarl import make
    from jax import random, numpy as jnp

    env = make("parabellum", map_width=256, map_height=256)
    rng, key = random.split(random.PRNGKey(0))
    obs, state = env.reset(key)
    state_seq = []
    for step in range(100):
        rng, key = random.split(rng)
        key_act = random.split(key, len(env.agents))
        actions = {
            agent: jnp.array(1)  # jax.random.randint(key_act[i], (), 0, 5)
            for i, agent in enumerate(env.agents)
        }
        state_seq.append((key, state, actions))
        rng, key_step = random.split(rng)
        obs, state, reward, done, infos = env.step(key_step, state, actions)

    vis = ParabellumVisualizer(env, state_seq)
    vis.animate()
