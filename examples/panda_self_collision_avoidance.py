import gym
import os

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

import numpy as np
import os
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf="panda.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "radius"])
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.5, -0.3, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": [0.0, -0.0, 0.4],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 0.0,
            "is_primary_goal": False,
            "indices": [1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.0, 0.0],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = (obst1, obst2)
    env.reset()
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """

    robot_type = 'panda'

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "20 * ca.norm_2(x)**4"
    # damper = {
    #     "alpha_b": 0.5,
    #     "alpha_eta": 0.5,
    #     "alpha_shift": 0.5,
    #     "beta_distant": 0.01,
    #     "beta_close": 6.5,
    #     "radius_shift": 0.1,
    # }
    # planner = ParameterizedFabricPlanner(
    #     degrees_of_freedom,
    #     robot_type,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/panda_for_fk.urdf", "r") as file:
        urdf = file.read()
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        'panda',
        urdf=urdf,
        root_link='panda_link0',
        end_link='panda_link9',
    )
    q = planner.variables.position_variable()

    collision_links = [f"panda_link{i}" for i in range(0,8)]
    self_collision_pairs = {"panda_hand": ["panda_link2"]}
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]

    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_pairs,
        goal,
        limits=panda_limits,
        number_obstacles=2,
    )
    planner.concretize()
    return planner


def run_panda_self_collision(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    action = np.zeros(7)
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][0][0],
            weight_goal_0=goal.sub_goals()[0].weight(),
            x_goal_1=ob_robot['FullSensor']['goals'][1][0][0:2],
            weight_goal_1=goal.sub_goals()[1].weight(),
            x_obst_0=ob_robot['FullSensor']['obstacles'][0][0],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][0][1],
            x_obst_1=ob_robot['FullSensor']['obstacles'][1][0],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][1][1],
            radius_body_panda_link3=np.array([0.02]),
            radius_body_panda_link4=np.array([0.02]),
            radius_body_panda_link5=np.array([0.02]),
            radius_body_panda_link6=np.array([0.02]),
            radius_body_panda_link7=np.array([0.02]),
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_panda_self_collision(n_steps=5000)
