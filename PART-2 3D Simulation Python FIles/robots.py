import logging
from logging.handlers import RotatingFileHandler
import agentpy as ap
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq
from owlready2 import *
from enum import Enum


# =========================
# Comprehensive Logging Configuration
# =========================

def configure_logging(log_filename='warehouse_simulation.log'):
    """
    Configures logging to direct all log messages to a specified file and
    prevents any log messages from appearing in the console.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = RotatingFileHandler(log_filename, mode='w', maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)


configure_logging()

# =========================
# Define Ontology
# =========================

onto = get_ontology("http://warehouse_simulation.org/onto.owl")

with onto:
    class Entity(Thing): pass


    class Robot(Entity): pass


    class Object(Entity): pass


    class GridCell(Thing): pass


    class has_location(ObjectProperty):
        domain = [Entity]
        range = [GridCell]


    class is_carrying(ObjectProperty):
        domain = [Robot]
        range = [Object]


    class is_stackable(DataProperty, FunctionalProperty):
        domain = [Object]
        range = [bool]


    class intends_to_move_to(ObjectProperty):
        domain = [Robot]
        range = [GridCell]


# =========================
# Define Robot States
# =========================
class RobotState(Enum):
    IDLE = 0
    PICKING_UP = 1
    DROPPING_OFF = 2
    MOVING = 3
    AVOIDING_COLLISION = 4
    REACTIVE_OVERRIDE = 5  # New state for reactive overrides


# =========================
# Define Warehouse Model
# =========================
class Warehouse(ap.Model):
    # Note: Do not include 'steps' in the parameters list to avoid conflicts
    parameters = ['num_robots', 'num_objects']

    parameters = ['num_robots', 'num_objects']

    def setup(self):
        # Initialize maximum steps
        self.max_steps = self.p.get('steps', 100)  # Default to 100 steps if not provided
        self.t = 0  # Initialize time step

        # Initialize a 10x10 grid for 2D movement
        self.grid = ap.Grid(self, (10, 10), track_empty=True)

        # Initialize robots and objects
        self.robots = ap.AgentList(self, self.p['num_robots'], Robot)
        self.objects = ap.AgentList(self, self.p['num_objects'], Object)

        # Initialize reservation table for collision avoidance
        self.reservation_table = {}  # Key: timestep, Value: set of reserved positions

        # Initialize shared intentions map for Hybrid Reasoning
        self.shared_intentions = {}  # Key: timestep, Value: set of positions

        # Initialize stackable cells list (cache for stack levels)
        self.stackable_cells = {}
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                self.stackable_cells[(x, y)] = 0  # Initialize stack level to 0

        # Place objects in random empty cells and update ontology
        for obj in self.objects:
            self.place_object(obj)

        # Place robots in random empty cells and update ontology
        for robot in self.robots:
            self.place_robot(robot)

        # Assign unique priorities to robots (lower ID = higher priority)
        for robot in self.robots:
            robot.priority = robot.id  # Assuming robot IDs are unique and ordered

        # Plot the initial state of the grid
        self.visualize_grid(initial=True)


    def step(self):
        self.shared_intentions[self.t] = set()
        logging.warning(f"--- Step {self.t} ---")

        # Sort robots based on priority (lower ID = higher priority)
        sorted_robots = sorted(self.robots, key=lambda r: r.priority)

        # Each robot announces and executes its planned route
        for robot in sorted_robots:
            robot.step()

        # Log the shared intentions for this step
        logging.debug(f"Shared intentions this step: {self.shared_intentions.get(self.t, set())}")

        # Log the reservation table for this step
        logging.debug(f"Reservation table this step: {self.reservation_table.get(self.t, set())}")

        # Check if all objects are sorted
        if all(obj.sorted for obj in self.objects):
            logging.info(f"All objects sorted in {self.t} steps.")
            self.stop()

        # Check if maximum steps reached
        if self.t >= self.max_steps:
            logging.info(f"Reached maximum steps ({self.max_steps}). Stopping simulation.")
            self.stop()

        super().step()  # Increment self.t

    def place_object(self, obj):
        pos = random.choice(list(self.grid.empty))
        self.grid.add_agents([obj], [pos])
        obj_entity = onto.Object(f"object_{obj.id}")
        grid_cell = onto.GridCell(f"cell_{pos}")
        obj_entity.has_location.append(grid_cell)
        obj.is_stackable = True
        logging.info(f"Placed object at {pos} in ontology")

        # Update stackable_cells cache
        self.stackable_cells[pos] += 1

    def place_robot(self, robot):
        pos = random.choice(list(self.grid.empty))
        self.grid.add_agents([robot], [pos])
        robot_entity = onto.Robot(f"robot_{robot.id}")
        grid_cell = onto.GridCell(f"cell_{pos}")
        robot_entity.has_location.append(grid_cell)
        logging.info(f"Placed robot {robot.id} at {pos} in ontology")


    def end(self):
        logging.info("Simulation finished.")

        # Log number of movements performed by each robot
        for robot in self.robots:
            logging.info(f"Robot {robot.id} performed {robot.movements} movements.")

        # Log final positions of each robot
        logging.info("Final Positions of Robots:")
        for robot in self.robots:
            pos = self.grid.positions[robot]
            logging.info(f"Robot {robot.id} is at position {pos}")

        # Log final positions of each box
        logging.info("Final Positions of Boxes:")
        for box in self.objects:
            pos = self.grid.positions[box]
            logging.info(f"Box {box.id} is at position {pos} - Sorted: {box.sorted}")

        # Perform analysis to suggest potential strategy improvements
        self.analyze_strategies()

        # Plot the final state of the grid
        self.visualize_grid(final=True)


    def analyze_strategies(self):
        """
        Analyzes the simulation data to suggest potential strategies to decrease time spent.
        """
        logging.info("Analyzing strategies to decrease time spent...")

        # Total time taken
        total_time = self.t
        logging.info(f"Total time taken until all objects are sorted: {total_time} steps.")

        # Total movements per robot
        movements = [robot.movements for robot in self.robots]
        average_movements = sum(movements) / len(movements) if movements else 0
        logging.info(f"Total movements per robot: {movements}")
        logging.info(f"Average movements per robot: {average_movements:.2f}")

        # Potential strategy suggestions
        suggestions = []

        # Example suggestions based on average movements
        if average_movements > 20:
            suggestions.append(
                "- *Optimize Pathfinding*: Implement more efficient pathfinding algorithms or heuristics to reduce unnecessary movements.")
        if len(self.robots) > len(self.objects):
            suggestions.append(
                "- *Balance Workload*: Reduce the number of idle robots by balancing the workload among them.")
        if any(robot.state == RobotState.AVOIDING_COLLISION for robot in self.robots):
            suggestions.append(
                "- *Improve Collision Avoidance*: Enhance the collision avoidance mechanism to minimize state transitions to AVOIDING_COLLISION.")
        if total_time > 100:
            suggestions.append(
                "- *Increase Robot Speed*: Allow robots to perform multiple actions per step or increase their speed to complete tasks faster.")

        if suggestions:
            logging.info("Potential strategies to decrease time spent:")
            for suggestion in suggestions:
                logging.info(suggestion)
        else:
            logging.info(
                "No immediate strategies identified to decrease time spent. Consider conducting further analysis.")

    def visualize_grid(self, initial=False, final=False):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Initialize an empty grid
        grid_data = np.full((self.grid.shape[0], self.grid.shape[1]), 'empty', dtype=object)

        # Mark the positions of objects and robots in the grid
        for agent in self.grid.agents:
            pos = self.grid.positions[agent]
            x, y = pos
            if isinstance(agent, Object):
                if grid_data[x, y] == 'robot':
                    grid_data[x, y] = 'robot+box'
                else:
                    grid_data[x, y] = 'box'
            elif isinstance(agent, Robot):
                if grid_data[x, y] == 'box':
                    grid_data[x, y] = 'robot+box'
                else:
                    grid_data[x, y] = 'robot'

        # Define a color map and labels
        colors = {
            'empty': '#f0f0f0',
            'box': '#ffcc00',
            'robot': '#008cff',
            'robot+box': '#ff5733'
        }

        for (x, y), value in np.ndenumerate(grid_data):
            rect = plt.Rectangle((y, x), 1, 1, facecolor=colors[value], edgecolor='black')
            ax.add_patch(rect)
            if value != 'empty':
                ax.text(y + 0.5, x + 0.5, value, ha='center', va='center', fontsize=8, color='black')

        # Set axis limits and labels
        ax.set_xlim(0, self.grid.shape[1])
        ax.set_ylim(0, self.grid.shape[0])
        ax.set_xticks(np.arange(0, self.grid.shape[1], 1))
        ax.set_yticks(np.arange(0, self.grid.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        # Set the title of the plot
        if initial:
            title = f'Warehouse Grid - Initial State'
        elif final:
            title = f'Warehouse Grid - Final State'
        else:
            title = f'Warehouse Grid - Step {self.steps}'
        ax.set_title(title)

        plt.show()


# =========================
# Define Object Agent
# =========================
class Object(ap.Agent):
    def setup(self):
        self.stack_level = 1
        self.sorted = False
        self.picked = False  # Indicates if the object has been picked up

    def increment_stack(self):
        self.stack_level += 1
        if self.stack_level >= 5:
            self.sorted = True
            logging.info(f"Object at {self.model.grid.positions[self]} sorted.")


# =========================
# Define Robot Agent with Hybrid Reasoning
# =========================
class Robot(ap.Agent):
    orientations = ['N', 'E', 'S', 'W']  # Focused on 2D orientations

    def setup(self):
        self.movements = 0
        self.carrying = None
        self.orientation = 'N'
        self.previous_positions = []  # Stack to track multiple previous positions
        self.state = RobotState.IDLE
        self.target = None
        self.path = []
        self.priority = self.id  # Assign priority based on robot ID
        self.max_retries = 3  # For pathfinding retries

        # Hybrid Reasoning Variables
        self.deliberative_goal = None  # Long-term goal
        self.reactive_goal = None  # Short-term goal from reactive layer

    def step(self):
        logging.debug(f"Robot {self.id} is in state {self.state.name}")

        # Reactive Reasoning: Handle immediate environmental changes
        self.reactive_reasoning()

        # Deliberative Reasoning: Plan long-term goals
        self.deliberative_reasoning()

        # Execute based on state
        if self.state == RobotState.IDLE:
            self.execute_idle_state()
        elif self.state == RobotState.PICKING_UP:
            self.execute_picking_up_state()
        elif self.state == RobotState.DROPPING_OFF:
            self.execute_dropping_off_state()
        elif self.state == RobotState.MOVING:
            self.reactive_move()
        elif self.state == RobotState.AVOIDING_COLLISION:
            self.reactive_avoid_collision()
        elif self.state == RobotState.REACTIVE_OVERRIDE:
            self.execute_reactive_override_state()
        else:
            logging.warning(f"Robot {self.id} is in an undefined state.")

    # =========================
    # Reactive Reasoning Layer
    # =========================
    def reactive_reasoning(self):
        """
        Handles immediate, environment-based decisions such as collision avoidance
        and movement in free cells.
        """
        # Check if next position in path is free
        if self.path:
            next_pos = self.path[0]
            if not self.is_position_free(next_pos):
                logging.warning(f"Robot {self.id}: Detected potential collision at {next_pos}.")
                self.state = RobotState.REACTIVE_OVERRIDE
                self.reactive_goal = 'COLLISION_AVOIDANCE'

    def execute_reactive_override_state(self):
        """
        Executes actions based on reactive goals, overriding deliberative plans.
        """
        if self.reactive_goal == 'COLLISION_AVOIDANCE':
            logging.info(f"Robot {self.id}: Executing collision avoidance maneuvers.")
            # Simple avoidance: try moving backward
            if self.move_backward():
                logging.info(f"Robot {self.id}: Moved backward to avoid collision.")
                self.state = RobotState.IDLE
            else:
                logging.warning(f"Robot {self.id}: Unable to move backward. Attempting to replan path.")
                # Attempt to replan path without the conflicting position
                self.replan_path()
                self.state = RobotState.MOVING
            self.reactive_goal = None  # Reset reactive goal

    def is_position_free(self, pos):
        """
        Checks if the given position is free (no robot or reserved).
        """
        # Check if any robot is at the position
        for agent in self.model.grid.agents:
            if isinstance(agent, Robot) and self.model.grid.positions[agent] == pos and agent != self:
                return False
        # Check reservation table
        for t in range(self.model.t, self.model.t + len(self.path) + 1):
            if pos in self.model.reservation_table.get(t, set()):
                return False
        return True


    def replan_path(self):
        """
        Attempts to replan the current path to avoid obstacles.
        """
        if self.target:
            robot_pos = self.model.grid.positions[self]
            if isinstance(self.target, Object):
                goal_pos = self.model.grid.positions[self.target]
            elif isinstance(self.target, tuple):
                goal_pos = self.target
            else:
                logging.error(f"Robot {self.id}: Invalid target for replanning.")
                return

            new_path = self.a_star(robot_pos, goal_pos)
            if new_path and self.announce_route(new_path):
                self.path = new_path
                self.state = RobotState.MOVING
                logging.info(f"Robot {self.id}: Replanned path to {goal_pos}: {new_path}")
            else:
                logging.warning(f"Robot {self.id}: Unable to replan path to {goal_pos}.")
                self.state = RobotState.IDLE

    # =========================
    # Deliberative Reasoning Layer
    # =========================
    def deliberative_reasoning(self):
        """
        Plans long-term goals such as organizing stacks in a specific pattern.
        Utilizes ontology-based data to enhance deliberation.
        """
        if not self.carrying and self.state == RobotState.IDLE:
            # Example: Decide on a pattern for stacking (e.g., centralizing stacks)
            self.deliberative_goal = 'ORGANIZE_STACKS'

        if self.deliberative_goal == 'ORGANIZE_STACKS':
            logging.info(f"Robot {self.id}: Planning to organize stacks.")
            self.deliberative_goal = None  # Reset after planning
            self.deliberative_decide_next_action()

    def deliberative_decide_next_action(self):
        """
        Decides the next action based on deliberative goals.
        """
        if not self.carrying:
            # Choose the nearest unsorted, unpicked object
            robot_pos = self.model.grid.positions[self]
            available_objects = [obj for obj in self.model.objects if not obj.sorted and not obj.picked]
            if available_objects:
                nearest_obj = min(available_objects,
                                  key=lambda obj: self.manhattan_distance(robot_pos, self.model.grid.positions[obj]))
                self.target = nearest_obj
                logging.info(f"Robot {self.id} decided to pick up object at {self.model.grid.positions[nearest_obj]}")
                self.state = RobotState.PICKING_UP
            else:
                logging.info(f"Robot {self.id} found no available objects to pick up.")
        else:
            # Carrying an object, decide where to drop it off
            self.state = RobotState.DROPPING_OFF

    def deliberative_pickup_object(self):
        if self.target:
            robot_pos = self.model.grid.positions[self]
            object_pos = self.model.grid.positions[self.target]
            logging.debug(f"Robot {self.id} planning path from {robot_pos} to {object_pos}")
            path = self.a_star(robot_pos, object_pos)
            if path:
                logging.debug(f"Robot {self.id} found path: {path}")
            else:
                logging.warning(f"Robot {self.id} could not find a path to object at {object_pos}")
            if path and self.announce_route(path):
                self.path = path
                self.state = RobotState.MOVING
            else:
                logging.warning(f"Robot {self.id} could not reserve path to object at {object_pos}.")
                self.state = RobotState.IDLE

    def deliberative_dropoff_object(self):
        # Define drop-off location as the center of the grid
        grid_width, grid_height = self.model.grid.shape
        center_positions = self.get_center_positions(grid_width, grid_height, 1)  # z=0 for 2D

        # Find the nearest center position with stack level less than 5
        robot_pos = self.model.grid.positions[self]
        available_centers = [pos for pos in center_positions if self.get_stack_level(pos) < 5]

        if available_centers:
            closest_center = min(available_centers, key=lambda pos: self.manhattan_distance(robot_pos, pos))
            self.target = closest_center
            logging.info(f"Robot {self.id} decided to drop off object at {closest_center}")
            path = self.a_star(robot_pos, closest_center)
            if path:
                logging.debug(f"Robot {self.id} found path: {path}")
            else:
                logging.warning(f"Robot {self.id} could not find a path to drop-off at {closest_center}")
            if path and self.announce_route(path):
                self.path = path
                self.state = RobotState.MOVING
            else:
                logging.warning(f"Robot {self.id} could not reserve path to drop-off at {closest_center}.")
                self.state = RobotState.IDLE
        else:
            logging.info(f"Robot {self.id} found no available drop-off positions.")
            self.state = RobotState.IDLE

    # =========================
    # Execute State Methods
    # =========================
    def execute_idle_state(self):
        """
        Executes actions when the robot is in IDLE state.
        """
        self.deliberative_decide_next_action()

    def execute_picking_up_state(self):
        """
        Executes actions when the robot is in PICKING_UP state.
        """
        self.deliberative_pickup_object()

    def execute_dropping_off_state(self):
        """
        Executes actions when the robot is in DROPPING_OFF state.
        """
        self.deliberative_dropoff_object()

    # =========================
    # Reactive Movement Methods
    # =========================
    def reactive_move(self):
        if self.path:
            next_pos = self.path.pop(0)
            logging.debug(f"Robot {self.id} attempting to move to {next_pos}")
            success = self.move_to(next_pos)
            if success:
                logging.debug(f"Robot {self.id} successfully moved to {next_pos}")
                if not self.path:
                    # Reached the target
                    if isinstance(self.target, Object):
                        self.pick_up_item(self.target)
                        self.state = RobotState.IDLE
                    elif isinstance(self.target, tuple):
                        self.drop_off_item(self.target)
                        self.state = RobotState.IDLE
            else:
                # Collision detected or path blocked
                logging.warning(f"Robot {self.id} encountered an obstacle while moving to {next_pos}.")
                self.state = RobotState.AVOIDING_COLLISION

    def reactive_avoid_collision(self):
        # Implement a retry mechanism
        if hasattr(self, 'retry_count'):
            self.retry_count += 1
        else:
            self.retry_count = 1

        if self.retry_count <= self.max_retries:
            logging.info(f"Robot {self.id} is retrying to find a path. Attempt {self.retry_count}/{self.max_retries}")
            # Attempt to replan the path
            if isinstance(self.target, Object):
                target_pos = self.model.grid.positions[self.target]
            elif isinstance(self.target, tuple):
                target_pos = self.target
            else:
                logging.error(f"Robot {self.id} has an invalid target.")
                self.state = RobotState.IDLE
                return

            robot_pos = self.model.grid.positions[self]
            new_path = self.a_star(robot_pos, target_pos)
            if new_path and self.announce_route(new_path):
                self.path = new_path
                self.retry_count = 0  # Reset retry count on success
                self.state = RobotState.MOVING
            else:
                logging.warning(f"Robot {self.id} could not replan path to {target_pos}.")
                self.state = RobotState.IDLE
        else:
            logging.error(f"Robot {self.id} exceeded maximum retries to find a path to {self.target}.")
            self.state = RobotState.IDLE
            self.retry_count = 0  # Reset retry count

    # =========================
    # Utility Methods
    # =========================
    def announce_route(self, path):
        logging.debug(f"Robot {self.id} announcing route: {path}")
        # Check for conflicts in the reservation table and shared intentions map
        for step_offset, step_pos in enumerate(path, start=self.model.t):
            if step_pos in self.model.reservation_table.get(step_offset, set()) or \
                    step_pos in self.model.shared_intentions.get(step_offset, set()):
                logging.warning(
                    f"Robot {self.id} found a conflict at {step_pos} for step {step_offset}. Recalculating.")
                return False
        # Reserve the path in the reservation table and shared intentions map
        for step_offset, step_pos in enumerate(path, start=self.model.t):
            self.model.reservation_table.setdefault(step_offset, set()).add(step_pos)
            self.model.shared_intentions.setdefault(step_offset, set()).add(step_pos)
            logging.debug(f"Robot {self.id} reserved position {step_pos} for step {step_offset}.")
        return True


    def move_to(self, next_pos):
        if self.is_valid_position(next_pos):
            # Rotate towards the next position
            self.rotate_towards(next_pos)
            # Move forward
            self.move_forward()
            return True
        else:
            logging.warning(f"Robot {self.id}: Position {next_pos} is invalid or already reserved.")
            return False

    def rotate_towards(self, next_pos):
        robot_pos = self.model.grid.positions[self]
        dx = next_pos[0] - robot_pos[0]
        dy = next_pos[1] - robot_pos[1]

        if dx == 1 and dy == 0:
            desired_direction = 'S'
        elif dx == -1 and dy == 0:
            desired_direction = 'N'
        elif dx == 0 and dy == 1:
            desired_direction = 'E'
        elif dx == 0 and dy == -1:
            desired_direction = 'W'
        else:
            logging.error(f"Robot {self.id}: Invalid direction from {robot_pos} to {next_pos}")
            return

        self.rotate_to(desired_direction)

    def move_forward(self):
        robot_pos = self.model.grid.positions[self]
        dx, dy = 0, 0

        if self.orientation == 'N':
            dx, dy = -1, 0
        elif self.orientation == 'S':
            dx, dy = 1, 0
        elif self.orientation == 'E':
            dx, dy = 0, 1
        elif self.orientation == 'W':
            dx, dy = 0, -1

        next_pos = (robot_pos[0] + dx, robot_pos[1] + dy)

        if self.is_valid_position(next_pos):
            self.previous_positions.append(robot_pos)  # Push current position to history before moving
            self.model.grid.move_to(self, next_pos)
            self.movements += 1
            logging.info(f"Robot {self.id} moved forward to {next_pos}, facing {self.orientation}")
            return True
        else:
            logging.warning(f"Robot {self.id}: Cannot move forward to {next_pos}, position is invalid or occupied.")
            return False

    def move_backward(self):
        if self.previous_positions:
            previous_pos = self.previous_positions.pop()
            # Determine direction to previous_pos
            dx = previous_pos[0] - self.model.grid.positions[self][0]
            dy = previous_pos[1] - self.model.grid.positions[self][1]

            # Determine desired direction to move backward
            if dx == 1 and dy == 0:
                desired_direction = 'S'
            elif dx == -1 and dy == 0:
                desired_direction = 'N'
            elif dx == 0 and dy == 1:
                desired_direction = 'E'
            elif dx == 0 and dy == -1:
                desired_direction = 'W'
            else:
                logging.error(f"Robot {self.id}: Cannot determine direction to move backward to {previous_pos}")
                return False

            # Rotate to desired direction
            self.rotate_to(desired_direction)

            # Move forward (which is effectively moving backward)
            if self.move_forward():
                logging.info(f"Robot {self.id} moved backward to {previous_pos}, facing {self.orientation}")
                return True
            else:
                logging.warning(
                    f"Robot {self.id}: Cannot move backward to {previous_pos}, position is invalid or occupied.")
                return False
        else:
            logging.warning(f"Robot {self.id}: Cannot move backward, no valid previous position.")
            return False

    def rotate_left(self):
        current_index = self.orientations.index(self.orientation)
        new_orientation = self.orientations[(current_index - 1) % len(self.orientations)]
        logging.debug(f"Robot {self.id} rotated left from {self.orientation} to {new_orientation}")
        self.orientation = new_orientation

    def rotate_right(self):
        current_index = self.orientations.index(self.orientation)
        new_orientation = self.orientations[(current_index + 1) % len(self.orientations)]
        logging.debug(f"Robot {self.id} rotated right from {self.orientation} to {new_orientation}")
        self.orientation = new_orientation

    def rotate_to(self, desired_orientation):
        if self.orientation == desired_orientation:
            return  # No rotation needed

        orientations = self.orientations
        current_index = orientations.index(self.orientation)
        desired_index = orientations.index(desired_orientation)

        # Calculate the number of left and right rotations needed
        left_steps = (current_index - desired_index) % len(orientations)
        right_steps = (desired_index - current_index) % len(orientations)

        # Choose the direction with minimal steps
        if left_steps <= right_steps:
            for _ in range(left_steps):
                self.rotate_left()
        else:
            for _ in range(right_steps):
                self.rotate_right()

    def pick_up_item(self, item):
        if not self.carrying and not item.picked:
            self.carrying = item
            item.picked = True
            logging.info(f"Robot {self.id} picked up object at {self.model.grid.positions[item]}")
        else:
            logging.warning(f"Robot {self.id} cannot pick up object at {self.model.grid.positions[item]}")

    def drop_off_item(self, drop_pos):
        if self.carrying:
            current_stack = self.get_stack_level(drop_pos)
            if current_stack < 5:
                logging.info(f"Robot {self.id} dropped off item at {drop_pos}")
                # Move the object to the drop position
                self.model.grid.move_to(self.carrying, drop_pos)
                self.carrying.sorted = True  # Mark object as sorted
                self.carrying = None  # Release the item
                self.increment_stack(drop_pos)
            else:
                logging.warning(f"Robot {self.id}: Drop-off position {drop_pos} is full. Cannot drop off here.")
                self.state = RobotState.AVOIDING_COLLISION

    def get_stack_level(self, pos):
        stack = [obj for obj in self.model.objects if self.model.grid.positions.get(obj) == pos and obj.sorted]
        return len(stack)

    def increment_stack(self, pos):
        # Find all sorted objects at the drop position and increment their stack levels
        for obj in self.model.objects:
            if self.model.grid.positions.get(obj) == pos and obj.sorted:
                obj.increment_stack()
                logging.info(f"Robot {self.id}: Incremented stack level for object {obj.id} at {pos}")

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        logging.warning(f"Robot {self.id} could not find a path from {start} to {goal}")
        return None

    def get_neighbors(self, position):
        x, y = position
        neighbors = [
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)  # 2D neighbors
        ]
        return [pos for pos in neighbors if self.is_valid_position(pos)]

    def is_valid_position(self, pos):
        # Check if position is within grid bounds
        if pos not in self.model.grid.all:
            logging.debug(f"Robot {self.id}: Position {pos} is out of bounds.")
            return False
        # Check if position is reserved in the current or future timesteps
        for t in range(self.model.t, self.model.t + len(self.path) + 1):
            if pos in self.model.reservation_table.get(t, set()):
                logging.debug(f"Robot {self.id}: Position {pos} is reserved for timestep {t}.")
                return False
        # Position is valid and not reserved
        return True

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        logging.debug(f"Robot {self.id}: Reconstructed path: {path}")
        return path[1:]

    def get_center_positions(self, grid_width, grid_height, grid_levels):
        center_x = (grid_width - 1) / 2
        center_y = (grid_height - 1) / 2

        if grid_width % 2 == 0 and grid_height % 2 == 0:
            center_positions = [
                (int(center_x), int(center_y)),
                (int(center_x) + 1, int(center_y)),
                (int(center_x), int(center_y) + 1),
                (int(center_x) + 1, int(center_y) + 1)
            ]
        elif grid_width % 2 == 0:
            center_positions = [
                (int(center_x), int(center_y)),
                (int(center_x) + 1, int(center_y))
            ]
        elif grid_height % 2 == 0:
            center_positions = [
                (int(center_x), int(center_y)),
                (int(center_x), int(center_y) + 1)
            ]
        else:
            center_positions = [(int(center_x), int(center_y))]

        return center_positions


# =========================
# Running the Simulation
# =========================
if __name__ == "__main__":
    parameters = {'steps': 100, 'num_robots': 5, 'num_objects': 10}
    model = Warehouse(parameters)
    results = model.run()