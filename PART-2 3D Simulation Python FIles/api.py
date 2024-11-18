import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
from flask_cors import CORS
from robots import Warehouse, configure_logging

configure_logging()

app = Flask(__name__)
CORS(app)

simulation_model = None
parameters = {}

@app.route('/initialize', methods=['POST'])
def initialize_simulation():
    global simulation_model, parameters
    try:
        logging.info("Initializing simulation...")
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        num_robots = data.get('num_robots', 3)
        num_objects = data.get('num_objects', 5)
        steps = data.get('steps', 100)  # Default to 100 steps if not provided

        parameters = {'num_robots': num_robots, 'num_objects': num_objects, 'steps': steps}
        simulation_model = Warehouse(parameters)
        simulation_model.setup()
        logging.info("Simulation initialized successfully.")

        return jsonify({"message": "Simulation initialized successfully", "parameters": parameters}), 200
    except Exception as e:
        logging.error(f"Error initializing simulation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/step', methods=['POST'])
def run_step():
    global simulation_model
    try:
        if simulation_model is None:
            logging.error("Simulation not initialized.")
            return jsonify({"error": "Simulation not initialized"}), 400

        if simulation_model.running:
            logging.info(f"Executing Step {simulation_model.t}...")
            simulation_model.step()
            is_complete = not simulation_model.running
            logging.info(f"Step {simulation_model.t - 1} executed. Simulation running: {simulation_model.running}")
        else:
            is_complete = True
            logging.info("Simulation already completed.")

        return jsonify({"message": f"Step {simulation_model.t - 1} executed successfully", "complete": is_complete}), 200
    except Exception as e:
        logging.error(f"Error during step execution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/positions', methods=['GET'])
def get_positions():
    global simulation_model
    try:
        if simulation_model is None:
            return jsonify({"error": "Simulation not initialized"}), 400

        # Get positions with simplified format
        positions = {
            "robots": [
                {
                    "id": robot.id,
                    "position": [
                        float(simulation_model.grid.positions[robot][0]) * 10,  # Scale up for visibility
                        float(simulation_model.grid.positions[robot][1]) * 10   # Scale up for visibility
                    ],
                    "picked": robot.carrying is not None
                }
                for robot in simulation_model.robots
            ],
            "boxes": [
                {
                    "id": obj.id,
                    "position": [
                        float(simulation_model.grid.positions[obj][0]) * 10,  # Scale up for visibility
                        float(simulation_model.grid.positions[obj][1]) * 10   # Scale up for visibility
                    ],
                    "picked": obj.picked
                }
                for obj in simulation_model.objects
            ]
        }
        return jsonify(positions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    configure_logging()
    app.logger.handlers = logging.getLogger().handlers
    app.logger.setLevel(logging.DEBUG)
    app.run(debug=True, host='0.0.0.0', port=5555)