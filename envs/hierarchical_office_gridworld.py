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
        room_side: int,
        doorways: Doorways,
        initial_states: Tuple[LocalCoordinate, ...],
        objs: Dict[LocalCoordinate, AnyStr],
    ):
        self.grid_size = grid_size
        self.room_side = room_side
        self.initial_states = initial_states
        self.noise = 0

        # We assume there is an underlying graph
        # We further assume that all the rooms are "equivalent" which implies they are of the same size

        # Auxiliary local to global coordinates


        # The underlying graph is built as follows;
        # 1) First create each of the subgraphs for each room, where nodes are labeled with global coordinates
        rooms = []

        for i, j in product(range(self.grid_size[0]), range(self.grid_size[1])):
            
            g = nx.grid_graph(dim=[self.room_side]*2)

            g = nx.relabel_nodes(g, {u: self.local_to_global_coordinates((i, j, *u), self.grid_size, self.room_side) for u in g.nodes})

            rooms.append(g) 
        
        # 2) merge the rooms in a single graph and then connect them via doorways.
        graph = nx.compose_all(rooms)

        for u, v in doorways:
            u = self.local_to_global_coordinates(
                u, self.grid_size, self.room_side
            )
            v = self.local_to_global_coordinates(
                v, self.grid_size, self.room_side
            )

            graph.add_edge(u, v)


        # The objects are provided as a dict [LocalCoordinate, Str],
        # in which the LocalCoordinate specifies the coordinate 
            
        self.all_objs = list(set(objs.values()))

        objs = {self.local_to_global_coordinates(o, self.grid_size, self.room_side): objs[o] for o in objs}

        empty = {node : False for node in graph.nodes if node not in objs }
        
        nx.set_node_attributes(graph, empty, name="type")
        nx.set_node_attributes(graph, objs, name="type")


        self.graph = graph

    def reset(self):
        init = [
            self.local_to_global_coordinates(
                s, self.grid_size, self.room_side
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

        next_state = list(filter(lambda u: u[1:] == (y, x), self.graph.nodes))

        if next_state:
            if self.graph.has_edge(tuple(self.state), tuple(next_state)):
                next_state = next_state.pop()
        else: 
            next_state = self.state

        phi =  self._compute_phi(self.state, action, next_state)

        self.state = next_state

        return (
            self.state,
            0.0,
            False,
            {"phi": phi},
        )

    @staticmethod
    def local_to_global_coordinates(
        loc: LocalCoordinate, grid_size: Tuple[int, int], room_side: int
    ) -> GlobalCoordinate:
        (roomy, roomx, y, x) = loc
        roomidx = np.ravel_multi_index((roomy, roomx), grid_size)

        f = lambda u: (room_side * u[0] + u[2], room_side * u[1] + u[3])

        return (roomidx, *f((roomy, roomx, y, x)))
    

    def _compute_phi(self, state, action, next_state):


        next_state = tuple(*next_state)
                    
        obj = self.graph.nodes[next_state]["type"]
        total_objs = len(self.all_objs)

        # Note phi is d + 4 because of the exit states (cardinal positions
        phi = np.zeros(len(self.all_objs)+4, dtype=np.float32)

        if obj:
            idx = self.all_objs.index(obj)
            phi[idx] = 1

        if state[0] != next_state[0]:
            orx, ory = np.unravel_index(state[0], self.grid_size) 
            nrx, nry = np.unravel_index(next_state[0], self.grid_size) 
            diffx = nrx - orx 
            diffy = nry - ory 

            if diffx == 0:
                if diffy < 0:
                    idx = total_objs
                elif diffy > 0:
                    idx = total_objs + 2
            elif diffy == 0:
                if diffx < 0:
                    idx = total_objs + 1
                elif diffx > 0:
                    idx = total_objs + 3
                
            phi[idx] = 1

        return phi


if __name__ == "__main__":
    # Connect tje rooms
    doorways = [((0, 0, 1, 2), (0, 1, 1, 0)), 
                ((0, 0, 2, 2), (1, 0, 3, 2))]


    objects = {(0,0,0,0): "coffee1",
               (1,1,0,0): "coffee1",
               (1,1,0,4): "office1",
               (0,0,2,2): "decoration"}

    initial_states = [(1, 0, 1, 3)]

    grid_size = (2, 2)
    room_side = 5

    env = HierarchicalOfficeGridworld(grid_size, room_side, doorways, initial_states, objects)

    print(env.reset())
    print(env.step(0))
    print(env.step(1))

    print(env.step(1))
    print(env.step(1))
    print(env.step(1))
    print(env.step(1))    
    print(env.step(1))
    print(env.step(1))
    print(env.step(0))
    print(env.step(0))
    print(env.step(3))
    print(env.step(3))
    print(env.step(2))
    print(env.step(2))
    print(env.step(2))
    print(env.step(2))
    print(env.step(2))







