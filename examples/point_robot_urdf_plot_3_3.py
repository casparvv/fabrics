import gym
import numpy as np
import matplotlib.pyplot as plt
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
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
    pos0 = np.array([1.0, 1.0, 0.0])
    vel0 = np.array([1.0, 0.1, 0.0])
    initial_observation = env.reset(pos=pos0, vel=vel0)
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [3.0, 3.0, 0.0], "radius": 1.0},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    obstacles = [obst1] # Add additional obstacles here.
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 0,
                "child_link" : 1,
                "desired_position": [5.0, 4.0],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
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
    #damper_beta: str = "0.5"
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler,
            #damper_beta=damper_beta,
    )
    collision_links = [1]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
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
    obst1 = obstacles[0]
    print(f"Initial observation : {ob}")
    action = np.array([0.0, 0.0, 0.0])
    planner = set_planner(goal)
    # Start the simulation.
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.sub_goals()[0].position())
    sub_goal_0_weight = np.array(goal.sub_goals()[0].weight())
    obst1_position = np.array(obst1.position())
    vel_mags = []
    positions = np.zeros((n_steps, 2))
    for n_step in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        positions[n_step] = ob["robot_0"]["joint_state"]["position"][0:2]
        action[0:2] = planner.compute_action(
            q=ob["robot_0"]["joint_state"]["position"][0:2],
            qdot=ob["robot_0"]["joint_state"]["velocity"][0:2],
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            x_obst_0=obst1_position[0:2],
            radius_obst_0=np.array([obst1.radius()]),
            radius_body_1=np.array([0.2])
        )
        ob, *_, = env.step(action)
        vel_mag = np.linalg.norm(ob['robot_0']['joint_state']['velocity'][0:2])
        vel_mags.append(vel_mag)
        #print(f"Velocity magnitude at {env.t()}: {np.linalg.norm(ob['robot_0']['joint_state']['velocity'][0:2])}")
    return (positions, sub_goal_0_position, obstacles)


def plot_trajectory(positions, goal=None, obstacles = []):
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    
    for obstacle in obstacles:
        axs.add_patch(plt.Circle(obstacle.position(), radius=obstacle.radius(), color='#D80032', alpha=0.9, label='Obstacle'))
    
    if goal is not None:
        axs.add_patch(plt.Circle(goal, radius=0.25, color='#00A878', label='Goal position'))
    
    # Initial position
    axs.add_patch(plt.Circle([positions[0,0], positions[0,1]], radius=0.25, color='#F4D35E', label='Initial position'))
    
    # Trajectory
    axs.plot(positions[:,0], positions[:,1], color='#27474E', alpha=0.9, linewidth=3, label='Trajectory')
    
    # Arrows on trajectory
    #m = int(len(positions)/200)
    #arrow_width = 0.05; arrow_head_width = 3*arrow_width; arrow_length = 1.5*arrow_head_width
    #for n in (n+sum(range(n)) for n in (range (1, m, int(m/4)))):
    #    axs.arrow(positions[n, 0], positions[n, 1], positions[n + 1, 0] - positions[n, 0], positions[n + 1, 1] - positions[n, 1],
    #                width=arrow_width, head_width=arrow_head_width, head_length=arrow_length, color='#27474E', alpha=0.95)
    
    for item in ([axs.title, axs.xaxis.label, axs.yaxis.label] +
                axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(19)
    plt.rc('font', size=19)
    #axs.axis('equal')
    axs.axis('on')
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.legend(loc=2)
    axs.set_xlim([0, 6])
    axs.set_ylim([0, 6])
    return plt


if __name__ == "__main__":
    positions, goal_position, obstacles = run_point_robot_urdf(n_steps=1000, render=False)
    plot_trajectory(positions, goal=goal_position, obstacles=obstacles).show()

