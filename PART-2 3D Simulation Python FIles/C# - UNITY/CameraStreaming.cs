
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class CameraStreaming : MonoBehaviour
{
    public Camera robotCamera; 
    
    private string serverUrl = "http://127.0.0.1:8000/stream"; 

    void Start()
    {
        // Check if the camera is assigned in the script
        if (robotCamera == null)
        {
            Debug.LogError("The camera is not assigned in the script. Please assign a camera in the inspector.");
            return;
        }

        // Start the coroutine to stream to the server
        StartCoroutine(StreamToServer());
    }

    IEnumerator StreamToServer()
    {
        while (true)
        {
            // Capture the current frame from the camera
            RenderTexture currentRT = robotCamera.targetTexture;
            RenderTexture.active = currentRT;

            Texture2D image = new Texture2D(currentRT.width, currentRT.height, TextureFormat.RGB24, false);
            image.ReadPixels(new Rect(0, 0, currentRT.width, currentRT.height), 0, 0);
            image.Apply();

            // Convert the image to bytes
            byte[] imageBytes = image.EncodeToPNG();
            Destroy(image);

            Debug.Log($"Frame captured and converted to bytes. Size: {imageBytes.Length} bytes");

            // Send the image to the server (POST request)
            UnityWebRequest request = new UnityWebRequest(serverUrl, "POST");
            request.uploadHandler = new UploadHandlerRaw(imageBytes);
            request.SetRequestHeader("Content-Type", "application/octet-stream");
            request.downloadHandler = new DownloadHandlerBuffer();

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"Error sending frame: {request.error}");
            }
            else
            {
                Debug.Log("Frame successfully sent to the server");
            }

            // Frequency of sending frames
            yield return new WaitForSeconds(0.1f);
        }
    }
}
