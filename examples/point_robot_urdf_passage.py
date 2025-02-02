import gym
import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.goalComposition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
"""
Fabrics example for a 3D point mass robot.
The fabrics planner uses a 2D point mass to compute actions for a simulated 3D point mass.

To do: tune behavior.
"""

def initalize_environment(render):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
    
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([0.0, 0.1, 0.0])
    vel0 = np.array([0.5, 0.0, 0.0])
    initial_observation = env.reset(pos=pos0, vel=vel0)
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, 1.0, 0.0], "radius": 0.6},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, -1.0, 0.0], "radius": 0.6},
    }
    obst2 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    obstacles = (obst1, obst2) # Add additional obstacles here.
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 0,
                "child_link" : 1,
                "desired_position": [4.5, 0.0],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    # Add walls, the goal and the obstacle to the environment.
    #env.add_walls([0.1, 10, 0.5], [[5.0, 0, 0], [-5.0, 0.0, 0.0], [0.0, 5.0, np.pi/2], [0.0, -5.0, np.pi/2]])
    env.add_goal(goal)
    for obst in obstacles:
        env.add_obstacle(obst)
    return (env, obstacles, goal, initial_observation)


def set_planner(goal: GoalComposition):
    """
    Initializes the fabric planner for the point robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    degrees_of_freedom = 2
    robot_type = "pointRobot"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-0.2 / (x ** 1) * (1 - ca.heaviside(xdot)) * xdot ** 2"
    collision_finsler = "0.1/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    damper_beta: str = "0.5"
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            #collision_geometry=collision_geometry,
            #collision_finsler=collision_finsler,
            #damper_beta=damper_beta,
    )
    collision_links = [1]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_links,
        goal,
        number_obstacles=2,
    )
    planner.concretize()
    return planner


def run_point_robot_urdf(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run point robot example.
    
    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, obstacles, goal, initial_observation) = initalize_environment(render)
    ob = initial_observation
    obst1, obst2 = obstacles
    print(f"Initial observation : {ob}")
    action = np.array([0.0, 0.0, 0.0])
    planner = set_planner(goal)
    # Start the simulation.
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.sub_goals()[0].position())
    sub_goal_0_weight = np.array(goal.sub_goals()[0].weight())
    obst1_position = np.array(obst1.position())
    obst2_position = np.array(obst2.position())
    vel_mags = []
    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        action[0:2] = planner.compute_action(
            q=ob["robot_0"]["joint_state"]["position"][0:2],
            qdot=ob["robot_0"]["joint_state"]["velocity"][0:2],
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            x_obst_0=obst1_position[0:2],
            radius_obst_0=np.array([obst1.radius()]),
            x_obst_1=obst2_position[0:2],
            radius_obst_1=np.array([obst2.radius()]),
            radius_body_1=np.array([0.2])
        )
        ob, *_, = env.step(action)
        vel_mag = np.linalg.norm(ob['robot_0']['joint_state']['velocity'][0:2])
        vel_mags.append(vel_mag)
        print(f"Velocity magnitude at {env.t()}: {np.linalg.norm(ob['robot_0']['joint_state']['velocity'][0:2])}")
    return {}


if __name__ == "__main__":
    import sys
    res = run_point_robot_urdf(n_steps=6000, render=bool(int(sys.argv[1])))



