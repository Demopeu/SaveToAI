import numpy as np
from abc import abstractmethod
from typing import Optional
from gymnasium import Env

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import (
    MultiAgentObservation,
    observation_factory,
)
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.objects import Landmark, Obstacle

class GoalEnv(Env):
    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        raise NotImplementedError

class ParkingEnv(AbstractEnv, GoalEnv):
    PARKING_OBS = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        }
    }

    def __init__(self, config: dict = None, render_mode: Optional[str] = None) -> None:
        super().__init__(config, render_mode)
        self.observation_type_parking = None
        self.goal = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update( 
            {
                "observation": {
                    "type": "KinematicsGoal",
                    "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                },
                "action": {"type": "ContinuousAction"},
                "reward_weights": [1, 0.02, 0, 0, 0.01, 0.01],
                "success_goal_reward": 0.07,
                "collision_reward": -6,
                "steering_range": np.deg2rad(20),
                "simulation_frequency": 10,
                "policy_frequency": 1,
                "duration": 150,
                "screen_width": 800,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "scaling": 2,
                "controlled_vehicles": 1,
                "vehicles_count": 6,
                "add_walls": True,
                "max_speed": 3.0,
                "acc_max": 2.0,
                "comfort_acc_max": 1.0,
                "comfort_acc_min": -2.0,
            }
        )
        return config

    def define_spaces(self) -> None:
        super().define_spaces()
        self.observation_type_parking = observation_factory(
            self, self.PARKING_OBS["observation"]
        )

    def _info(self, obs, action) -> dict:
        info = super(ParkingEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(
                self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
                for agent_obs in obs
            )
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs["achieved_goal"], obs["desired_goal"])
        info.update({"is_success": success})
        return info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None,
              start_positions: Optional[list] = None, goal_positions: Optional[list] = None):
        super().reset(seed=seed)
        self._reset(start_positions, goal_positions)
        observation = self.observation_type_parking.observe()
        info = {}
        return observation, info

    def _reset(self, start_positions: Optional[list] = None,goal_positions: Optional[list] = None):
        self._create_road()
        self._create_vehicles(start_positions,goal_positions)

    def _create_road(self, spots: int = 4) -> None:
        net = RoadNetwork()
        width = 28
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 40
        length = 40

        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2

            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [x, y_offset], [x, y_offset + length], width=width, line_types=lt
                ),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [x, -y_offset], [x, -y_offset - length], width=width, line_types=lt
                ),
            )

            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [-x, y_offset], [-x, y_offset + length], width=width, line_types=lt
                ),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [-x, -y_offset], [-x, -y_offset - length], width=width, line_types=lt
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _find_position(self,input_position):
        input_start_position_list = {
            1:1,2:3,3:2,4:0,5:9,6:11,7:10,8:8
        }
        if input_position in input_start_position_list:
            return input_start_position_list[input_position]
        else:
            return 0

    def _create_vehicles(self,
                         start_positions: Optional[list] = None, goal_positions: Optional[list] = None) -> None:
        empty_spots = list(range(len(self.road.network.lanes_list())))

        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            if start_positions and i < len(start_positions):
                start_position = self._find_position(start_positions[i])
                lane_index = empty_spots[start_position]
            else:
                lane_index = empty_spots[self.np_random.choice(np.arange(len(empty_spots)))]
            lane = self.road.network.lanes_list()[lane_index]
            x0, y0 = lane.position(lane.length / 2, 0)

            vehicle = self.action_type.vehicle_class(
            self.road, [x0, y0], lane.heading_at(lane.length / 2), 0)
            vehicle.color = VehicleGraphics.EGO_COLOR
            vehicle.LENGTH = 28
            vehicle.WIDTH = 15
            vehicle.MAX_STEERING_ANGLE = self.config["steering_range"]

            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            if vehicle.lane_index in empty_spots:
                empty_spots.remove(vehicle.lane_index)

        for vehicle in self.controlled_vehicles:

            for i in range(self.config["controlled_vehicles"]):
                if goal_positions and i <len(goal_positions):
                    goal_position = self._find_position(goal_positions[i])
                    lane_index = empty_spots[goal_position]
            lane = self.road.network.lanes_list()[lane_index]
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading)
            vehicle.goal.LENGTH = 10
            vehicle.goal.WIDTH = 10
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        self.goal = self.controlled_vehicles[0].goal


        for lane_index in empty_spots:
            lane = self.road.network.lanes_list()[lane_index]
            position = lane.position(lane.length / 2, 0)
            is_occupied = any(
                np.linalg.norm(np.array(vehicle.position) - np.array(position)) < vehicle.LENGTH
                for vehicle in self.road.vehicles
            )
            is_goal = any(
                np.linalg.norm(np.array(goal.position) - np.array(position)) < goal.LENGTH
                for goal in [v.goal for v in self.controlled_vehicles]
            )
            if not is_occupied and not is_goal:
                obstacle_vehicle = self.action_type.vehicle_class(
                    self.road, position, 0, 0)
                obstacle_vehicle.color = (255, 0, 0)
                obstacle_vehicle.LENGTH = 28
                obstacle_vehicle.WIDTH = 15
                obstacle_vehicle.heading = lane.heading
                self.road.vehicles.append(obstacle_vehicle)

        if self.config["add_walls"]:
            width, height = 156,162
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        reward = sum(
            self.compute_reward(
                agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
            )
            for agent_obs in obs
        )
        reward += self.config["collision_reward"] * sum(
            v.crashed for v in self.controlled_vehicles
        )
        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -self.config["success_goal_reward"]
        )

    def _is_terminated(self) -> bool:
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(
            self._is_success(agent_obs["achieved_goal"], agent_obs["desired_goal"])
            for agent_obs in obs
        )
        return  bool(crashed or success)
    
    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

class ParkingEnvActionRepeat(ParkingEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 100})

class ParkingEnvParkedVehicles(ParkingEnv):
    def __init__(self):
        super().__init__()
