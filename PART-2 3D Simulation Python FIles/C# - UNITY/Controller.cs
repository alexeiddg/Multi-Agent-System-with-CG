using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;

public class UnityFlaskIntegration : MonoBehaviour
{
    // URLs and configuration
    public string initializeUrl = "http://127.0.0.1:5555/initialize";
    public string stepUrl = "http://127.0.0.1:5555/step";
    public string positionsUrl = "http://127.0.0.1:5555/positions";
    public int numRobots = 5;
    public int numObjects = 5;
    public int steps = 100;

    // Prefabs
    public GameObject robotPrefab;
    public GameObject boxPrefab;

    // Movement settings
    public float moveSpeed = 2.0f;
    public float stepSize = 2.0f;  // How far each movement step should be

    // Track game objects
    private Dictionary<int, GameObject> robots = new Dictionary<int, GameObject>();
    private Dictionary<int, GameObject> boxes = new Dictionary<int, GameObject>();
    private Dictionary<int, Vector3> lastKnownPositions = new Dictionary<int, Vector3>();
    private bool isComplete = false;

    // Parent objects for organization
    private GameObject robotsParent;
    private GameObject boxesParent;

    void Start()
    {
        // Create empty parent objects for organization
        robotsParent = new GameObject("Robots");
        boxesParent = new GameObject("Boxes");

        // Initialize robots at their starting positions
        for (int i = 0; i < numRobots; i++)
        {
            Vector3 startPos = new Vector3(-135.7f + (i * 2), 0, 102.69f);
            GameObject robot = Instantiate(robotPrefab, startPos, Quaternion.identity);
            robot.transform.parent = robotsParent.transform;
            robot.name = $"Robot_{i}";
            robots.Add(i, robot);
            lastKnownPositions.Add(i, startPos);
        }

        // Initialize boxes at their starting positions
        for (int i = 0; i < numObjects; i++)
        {
            Vector3 startPos = new Vector3(-138.23f + (i * 2), 0, 107.43f);
            GameObject box = Instantiate(boxPrefab, startPos, Quaternion.identity);
            box.transform.parent = boxesParent.transform;
            box.name = $"Box_{i}";
            boxes.Add(i, box);
        }

        StartCoroutine(InitializeSimulation());
    }

    IEnumerator InitializeSimulation()
    {
        InitializeData initData = new InitializeData
        {
            num_robots = numRobots,
            num_objects = numObjects,
            steps = steps
        };

        string jsonData = JsonConvert.SerializeObject(initData);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);

        UnityWebRequest request = new UnityWebRequest(initializeUrl, "POST");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log("Simulation initialized: " + request.downloadHandler.text);
            StartCoroutine(SimulateAndFetchPositions());
        }
        else
        {
            Debug.LogError("Failed to initialize simulation: " + request.error);
        }
    }

    IEnumerator SimulateAndFetchPositions()
    {
        while (!isComplete)
        {
            // Step the simulation
            UnityWebRequest stepRequest = new UnityWebRequest(stepUrl, "POST");
            stepRequest.uploadHandler = new UploadHandlerRaw(new byte[0]);
            stepRequest.downloadHandler = new DownloadHandlerBuffer();
            stepRequest.SetRequestHeader("Content-Type", "application/json");
            yield return stepRequest.SendWebRequest();

            if (stepRequest.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Failed to step simulation: " + stepRequest.error);
                yield break;
            }

            // Parse the response
            StepResponse stepResponse = JsonConvert.DeserializeObject<StepResponse>(stepRequest.downloadHandler.text);
            if (stepResponse.complete)
            {
                Debug.Log("Simulation completed.");
                isComplete = true;
            }

            // Fetch updated positions
            UnityWebRequest positionsRequest = UnityWebRequest.Get(positionsUrl);
            yield return positionsRequest.SendWebRequest();

            if (positionsRequest.result == UnityWebRequest.Result.Success)
            {
                PositionData data = JsonConvert.DeserializeObject<PositionData>(positionsRequest.downloadHandler.text);
                UpdatePositions(data);
            }
            else
            {
                Debug.LogError("Failed to fetch positions: " + positionsRequest.error);
                yield break;
            }

            yield return new WaitForSeconds(1.0f); // Poll every second
        }
    }

    void UpdatePositions(PositionData data)
    {
        Debug.Log($"Received position update - Robots: {data.robots.Count}, Boxes: {data.boxes.Count}");

        // Update Robots
        foreach (var robotData in data.robots)
        {
            if (robots.TryGetValue(robotData.id, out GameObject robot))
            {
                Vector3 lastPos = lastKnownPositions[robotData.id];
                Vector3 currentPos = robot.transform.position;

                // Determine movement direction based on simulation data
                bool movedRight = robotData.position[0] > lastPos.x;
                bool movedLeft = robotData.position[0] < lastPos.x;
                bool movedForward = robotData.position[1] > lastPos.z;
                bool movedBack = robotData.position[1] < lastPos.z;

                // Calculate new target position
                Vector3 targetPos = currentPos;
                if (movedRight) targetPos += Vector3.right * stepSize;
                if (movedLeft) targetPos += Vector3.left * stepSize;
                if (movedForward) targetPos += Vector3.forward * stepSize;
                if (movedBack) targetPos += Vector3.back * stepSize;

                // Move robot if position changed
                if (targetPos != currentPos)
                {
                    StartCoroutine(SmoothMove(robot, targetPos));
                    lastKnownPositions[robotData.id] = new Vector3(robotData.position[0], 0, robotData.position[1]);
                }

                // Handle box pickup
                if (robotData.carrying.HasValue)
                {
                    if (boxes.TryGetValue(robotData.carrying.Value, out GameObject box))
                    {
                        box.transform.parent = robot.transform;
                        box.transform.localPosition = new Vector3(0, 1f, 0); // Place box above robot
                    }
                }
            }
            else
            {
                // Robot not found, create it
                Vector3 startPos = new Vector3(robotData.position[0], 0, robotData.position[1]);
                GameObject newRobot = Instantiate(robotPrefab, startPos, Quaternion.identity);
                newRobot.transform.parent = robotsParent.transform;
                newRobot.name = $"Robot_{robotData.id}";
                robots.Add(robotData.id, newRobot);
                lastKnownPositions.Add(robotData.id, startPos);
            }
        }

        // Update Boxes
        foreach (var boxData in data.boxes)
        {
            if (boxes.TryGetValue(boxData.id, out GameObject box))
            {
                // Only move boxes that aren't being carried
                if (!boxData.picked.GetValueOrDefault(false))
                {
                    box.transform.parent = null; // Unparent from robot if it was carried
                    if (boxData.sorted.GetValueOrDefault(false))
                    {
                        // Move sorted boxes to a specific area
                        Vector3 sortedPos = new Vector3(box.transform.position.x, 0, box.transform.position.z + 2f);
                        StartCoroutine(SmoothMove(box, sortedPos));
                    }
                }
            }
            else
            {
                // Box not found, create it
                Vector3 startPos = new Vector3(boxData.position[0], 0, boxData.position[1]);
                GameObject newBox = Instantiate(boxPrefab, startPos, Quaternion.identity);
                newBox.transform.parent = boxesParent.transform;
                newBox.name = $"Box_{boxData.id}";
                boxes.Add(boxData.id, newBox);
            }
        }
    }

    IEnumerator SmoothMove(GameObject obj, Vector3 targetPos)
    {
        float elapsedTime = 0;
        Vector3 startPos = obj.transform.position;

        while (elapsedTime < 1.0f)
        {
            elapsedTime += Time.deltaTime * moveSpeed;
            obj.transform.position = Vector3.Lerp(startPos, targetPos, elapsedTime);
            yield return null;
        }

        obj.transform.position = targetPos;
    }

    void UpdateRobotVisuals(GameObject robotObj, AgentData robotData)
    {
        // Update robot appearance based on state
        if (robotData.carrying.HasValue)
        {
            // Add visual indication that robot is carrying a box
            // For example, change color or attach a box model to the robot
        }
    }

    void UpdateBoxVisuals(GameObject boxObj, AgentData boxData)
    {
        // Update box appearance based on state
        if (boxData.sorted == true)
        {
            // Add visual indication that box is sorted
            // For example, change color to green
        }
        if (boxData.picked == true)
        {
            // Add visual indication that box is being carried
            // For example, make it invisible or attach it to the robot
        }
    }
}

[System.Serializable]
public class InitializeData
{
    public int num_robots;
    public int num_objects;
    public int steps;
}

[System.Serializable]
public class PositionData
{
    public List<AgentData> robots;
    public List<AgentData> boxes;
}

[System.Serializable]
public class AgentData
{
    public int id;
    public float[] position;
    public string state;        // For robots
    public int? carrying;       // For robots
    public string orientation;  // For robots
    public bool? sorted;        // For boxes
    public bool? picked;        // For boxes
    public int? stack_level;    // For boxes
}

[System.Serializable]
public class StepResponse
{
    public string message;
    public bool complete;
}