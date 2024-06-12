"""Parabellum environment based on SMAX"""

import jax.numpy as jnp
import jax
import numpy as np
from jax import random
from jax import jit
from flax.struct import dataclass
import chex
from jaxmarl.environments.smax.smax_env import State, SMAX
from typing import Tuple, Dict
from functools import partial


@dataclass
class Scenario:
    """Parabellum scenario"""

    obstacle_coords: chex.Array
    obstacle_deltas: chex.Array
    unit_types: chex.Array

    num_allies: int = 5
    num_enemies: int = 5

    smacv2_position_generation: bool = False
    smacv2_unit_type_generation: bool = False


# default scenario
scenarios = {
    "default": Scenario(
        jnp.array([[8, 10], [24, 10], [16, 12]]) * 4,
        jnp.array([[0, 12], [0, 12], [0, 8]]) * 4,
        jnp.zeros((10,), dtype=jnp.uint8),
    )
}


class Parabellum(SMAX):
    def __init__(
        self,
        scenario: Scenario = scenarios["default"],
        unit_type_attack_blasts=jnp.array([0, 0, 0, 0, 0, 0]),
        **kwargs,
    ):
        super().__init__(scenario=scenario, **kwargs)
        self.unit_type_attack_blasts = unit_type_attack_blasts
        self.obstacle_coords = scenario.obstacle_coords
        self.obstacle_deltas = scenario.obstacle_deltas
        # overwrite supers _world_step method
        self._world_step = self._world_step

    @partial(jax.jit, static_argnums=(0,))
    # replace the _world_step method
    def _world_step(  # modified version of JaxMARL's SMAX _word_step
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Tuple[chex.Array, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        def update_position(idx, vec):
            # Compute the movements slightly strangely.
            # The velocities below are for diagonal directions
            # because these are easier to encode as actions than the four
            # diagonal directions. Then rotate the velocity 45
            # degrees anticlockwise to compute the movement.
            pos = state.unit_positions[idx]
            new_pos = (
                pos
                + vec
                * self.unit_type_velocities[state.unit_types[idx]]
                * self.time_per_step
            )
            # avoid going out of bounds
            new_pos = jnp.maximum(
                jnp.minimum(new_pos, jnp.array([self.map_width, self.map_height])),
                jnp.zeros((2,)),
            )
            # avoid going into obstacles

            #######################################################################
            #######################################################################

            # if trajectory from  pos to new_pos crosses an obstacle line (have length one)
            # new_pos = pos, else new_pos = new_pos
            # —————————————————
            # |               |
            # |       a1      |
            # |      b1—b2    |
            # |       a2      |
            # |               |
            # —————————————————

            @partial(jax.vmap, in_axes=(None, None, 0, 0))
            def inter_fn(a1, a2, b1, b2):  # TODO: double check this is correct
                # if line from a1 to a2 intersects with b1 to b2
                denom = jnp.linalg.det(jnp.stack([a2 - a1, b2 - b1]))
                t = jnp.linalg.det(jnp.stack([b1 - a1, b2 - b1])) / denom
                u = jnp.linalg.det(jnp.stack([a2 - a1, b1 - a1])) / denom
                return jnp.where(denom == 0, 0, (0 < t) & (t < 1) & (0 < u) & (u < 1))

            print("confirming no errors")
            pos = pos  # a1  (x, y)
            new_pos = new_pos  # a2  (x, y)
            obstacle_start = self.obstacle_coords  # b1 (x, y)
            obstacle_end = obstacle_start + self.obstacle_deltas  # b2  (x, y)
            # if obstacle_start and obstacle_end are empty make them into jnp.array([[-1,-1]])
            inters = jnp.any(inter_fn(pos, new_pos, obstacle_start, obstacle_end))
            new_pos = jax.lax.cond(inters, lambda: pos, lambda: new_pos)

            #######################################################################
            #######################################################################

            return new_pos

        #######################################################################
        #######################################################################
        #######################################################################

        def bystander_fn(attacked_idx):
            idxs = (
                jnp.zeros((self.num_agents,))
                .at[: self.num_allies]
                .set(attacked_idx > self.num_allies)
            )
            idxs *= (
                jnp.linalg.norm(
                    state.unit_positions - state.unit_positions[attacked_idx], axis=-1
                )
                < self.unit_type_attack_blasts[state.unit_types[attacked_idx]]
            )
            return idxs

            #######################################################################
            #######################################################################
            #######################################################################

        def update_agent_health(idx, action, key):  # TODO: add attack blasts
            # for team 1, their attack actions are labelled in
            # reverse order because that is the order they are
            # observed in
            attacked_idx = jax.lax.cond(
                idx < self.num_allies,
                lambda: action + self.num_allies - self.num_movement_actions,
                lambda: self.num_allies - 1 - (action - self.num_movement_actions),
            )
            # deal with no-op attack actions (i.e. agents that are moving instead)
            attacked_idx = jax.lax.select(
                action < self.num_movement_actions, idx, attacked_idx
            )

            #########################################################
            bystanders = bystander_fn(attacked_idx)  # TODO: use
            #########################################################

            attack_valid = (
                (
                    jnp.linalg.norm(
                        state.unit_positions[idx] - state.unit_positions[attacked_idx]
                    )
                    < self.unit_type_attack_ranges[state.unit_types[idx]]
                )
                & state.unit_alive[idx]
                & state.unit_alive[attacked_idx]
            )
            attack_valid = attack_valid & (idx != attacked_idx)
            attack_valid = attack_valid & (state.unit_weapon_cooldowns[idx] <= 0.0)
            health_diff = jax.lax.select(
                attack_valid,
                -self.unit_type_attacks[state.unit_types[idx]],
                0.0,
            )
            # design choice based on the pysc2 randomness details.
            # See https://github.com/deepmind/pysc2/blob/master/docs/environment.md#determinism-and-randomness

            #########################################################
            #########################################################

            bystander_valid = jnp.where(
                attack_valid, bystanders, jnp.zeros((self.num_agents,))
            )
            bystander_health_diff = (
                bystander_valid * -self.unit_type_attacks[state.unit_types[idx]]
            )
            health_diff = health_diff  # (health_diff + bystander_health_diff).squeeze()

            #########################################################
            #########################################################

            cooldown_deviation = jax.random.uniform(
                key, minval=-self.time_per_step, maxval=2 * self.time_per_step
            )
            cooldown = (
                self.unit_type_weapon_cooldowns[state.unit_types[idx]]
                + cooldown_deviation
            )
            cooldown_diff = jax.lax.select(
                attack_valid,
                # subtract the current cooldown because we are
                # going to add it back. This way we effectively
                # set the new cooldown to `cooldown`
                cooldown - state.unit_weapon_cooldowns[idx],
                -self.time_per_step,
            )
            return health_diff, attacked_idx, cooldown_diff

        def perform_agent_action(idx, action, key):
            movement_action, attack_action = action
            new_pos = update_position(idx, movement_action)
            health_diff, attacked_idxes, cooldown_diff = update_agent_health(
                idx, attack_action, key
            )

            return new_pos, (health_diff, attacked_idxes), cooldown_diff

        keys = jax.random.split(key, num=self.num_agents)
        pos, (health_diff, attacked_idxes), cooldown_diff = jax.vmap(
            perform_agent_action
        )(jnp.arange(self.num_agents), actions, keys)
        # Multiple enemies can attack the same unit.
        # We have `(health_diff, attacked_idx)` pairs.
        # `jax.lax.scatter_add` aggregates these exactly
        # in the way we want -- duplicate idxes will have their
        # health differences added together. However, it is a
        # super thin wrapper around the XLA scatter operation,
        # which has this bonkers syntax and requires this dnums
        # parameter. The usage here was inferred from a test:
        # https://github.com/google/jax/blob/main/tests/lax_test.py#L2296
        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        unit_health = jnp.maximum(
            jax.lax.scatter_add(
                state.unit_health,
                jnp.expand_dims(attacked_idxes, 1),
                health_diff,
                dnums,
            ),
            0.0,
        )
        unit_weapon_cooldowns = state.unit_weapon_cooldowns + cooldown_diff
        state = state.replace(
            unit_health=unit_health,
            unit_positions=pos,
            unit_weapon_cooldowns=unit_weapon_cooldowns,
        )
        return state


if __name__ == "__main__":
    scenario = Scenario(
        jnp.array([[0, 0], [1, 1], [2, 2]]),
        jnp.array([0, 1, 2]),
        jnp.array([0, 1, 2]),
    )
    env = Parabellum(scenario)
