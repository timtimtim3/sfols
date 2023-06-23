import numpy as np
import gym
from typing import Dict, List, Tuple, Callable, AnyStr
from gym.envs.classic_control import rendering
import networkx as nx
from itertools import product
import random


# First two ints are room location in the grid, the other 2 are the location within the room.
LocalCoordinate = Tuple[int, int, int, int]
GlobalCoordinate = Tuple[int, int]
Doorways = Tuple[Tuple[LocalCoordinate, LocalCoordinate], ...]

"""

    When the environment (partition_id, global_y, global_x). 
    For any specification the is (room_y, room_x, local_y, local_x)


"""


class HierarchicalOfficeGridworld(gym.Env):
    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

    def __init__(
        self,
        grid_size: Tuple[int, int],
        room_size: int,
        doorways: Doorways,
        initial_states: Tuple[LocalCoordinate, ...],
        objects: Dict[LocalCoordinate, AnyStr],
    ):
        self.grid_size = grid_size
        self.room_size = room_size
        self.initial_states = initial_states
        self.noise = 0

        # We assume there is an underlying graph
        # We further assume that all the rooms are "equivalent" which implies they are of the same size

        # Auxiliary local to global coordinates
        f = lambda u: (self.room_size[0] * u[0] + u[2], self.room_size[1] * u[1] + u[3])

        rooms = []

        for i, j in product(range(self.grid_size[0]), range(self.grid_size[1])):
            
            g = nx.grid_graph(dim=[self.room_size]*2)

            g = nx.relabel_nodes(g, {u: self.local_to_global_coordinates((i, j, *u), self.grid_size, self.room_size) for u in g.nodes})

            rooms.append(g)

        graph = nx.compose_all(rooms)

        for u, v in doorways:
            u = self.local_to_global_coordinates(
                u, self.grid_size, self.room_size
            )
            v = self.local_to_global_coordinates(
                v, self.grid_size, self.room_size
            )

            graph.add_edge(u, v)


        objects = {self.local_to_global_coordinates(o, self.grid_size, self.room_size):objects[o] for o in objects}
        empty = {node : False for node in graph.nodes if node not in objects }
        
        nx.set_node_attributes(graph, objects, name="type")
        nx.set_node_attributes(graph, empty, name="type")

        self.graph = graph

    def reset(self):
        init = [
            self.local_to_global_coordinates(
                s, self.grid_size, self.room_size
            )
            for s in self.initial_states
        ]

        self.state = random.choice(init)
        return np.asarray(self.state, dtype=np.int32)

    def step(self, action):
        (_, y, x) = self.state

        effective_action = action

        if random.uniform(0, 1) < self.noise:
            nactions = [0, 1, 2, 3]
            nactions.remove(action)
            effective_action = np.random.choice(nactions)

        # move
        if effective_action == self.LEFT:
            x -= 1
        elif effective_action == self.UP:
            y -= 1
        elif effective_action == self.RIGHT:
            x += 1
        elif effective_action == self.DOWN:
            y += 1
        else:
            raise Exception("bad action {}".format(action))

        next_state = list(filter(lambda u: u[1:] == (x, y), self.graph.nodes))

        if next_state:
            self.state = next_state

        return (
            self.state,
            0.0,
            False,
            {"phi": np.zeros(4, dtype=np.float32)},
        )

    @staticmethod
    def local_to_global_coordinates(
        loc: LocalCoordinate, grid_size: Tuple[int, int], room_size: int
    ) -> GlobalCoordinate:
        (roomy, roomx, y, x) = loc
        roomidx = np.ravel_multi_index((roomy, roomx), grid_size)

        f = lambda u: (room_size * u[0] + u[2], room_size * u[1] + u[3])

        return (roomidx, *f((roomy, roomx, y, x)))


if __name__ == "__main__":
    # Connect
    doorways = [((0, 0, 1, 2), (0, 1, 1, 0)), ((0, 0, 2, 2), (1, 0, 3, 2))]



    objects = {(0,0,0,0): "coffee1"}

    initial_states = [(0, 0, 1, 1)]

    grid_size = (2, 2)
    room_size = (3, 3)

    env = HierarchicalOfficeGridworld((2, 2), 3, doorways, initial_states, objects)

    env.reset()

    print(env.state)

    print(env.step(3))

    print(nx.get_node_attributes(env.graph, "type"))
