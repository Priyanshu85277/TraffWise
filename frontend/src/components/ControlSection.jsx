import React, { useState, useEffect } from "react";
import {
  FaCamera,
  FaCog,
  FaPowerOff,
  FaList,
  FaPause,
  FaPlay,
} from "react-icons/fa";
import "./ControlSection.css";
import { useNavigate } from "react-router-dom";
import { useData } from "../contexts/DataProvider";
import { API_URL } from "../api";

export default function ControlSection({ togglePanel, activePanel, cameraId }) {
  const navigate = useNavigate();
  const [isOn, setIsOn] = useState(true);
  const { cameras } = useData();
  const [selectedModel, setSelectedModel] = useState("yolo11");
  const [selectedCamera, setSelectedCamera] = useState("1");
  const [, setError] = useState(null);
  const [captureStatus, setCaptureStatus] = useState(null);
  const [isPaused, setIsPaused] = useState(false);

  const toggleOnOff = async () => {
    setError(null);
    const newState = !isOn;

    try {
      const response = await fetch(`${API_URL}/toggle_annotations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ show_annotations: newState }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error toggling annotations:", errorData);
        setError(
          `Failed to toggle annotations: ${errorData.detail || "Unknown error"}`
        );
        return;
      }

      const data = await response.json();
      setIsOn(data.show_annotations);
      console.log(
        `Annotations ${data.show_annotations ? "enabled" : "disabled"}`
      );
    } catch (err) {
      console.error("Network error during annotation toggle:", err);
      setError("Network error when toggling annotations");
    }
  };

  useEffect(() => {
    if (cameraId) setSelectedCamera(String(cameraId));
  }, [cameraId]);

  const handleModelChange = async (e) => {
    const model_type = e.target.value;
    setSelectedModel(model_type);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/set_model`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model_type }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error switching model:", errorData);
        setError(
          `Failed to switch model: ${errorData.detail || "Unknown error"}`
        );
      }
    } catch (err) {
      console.error("Network error:", err);
      setError("Network error when switching model");
    }
  };

  const handleCameraChange = (e) => {
    const camera_id = e.target.value;
    setSelectedCamera(camera_id);
    navigate(`/monitor/${camera_id}`);
  };

  const handleCapture = async () => {
    setError(null);
    setCaptureStatus("Capturing...");

    try {
      const response = await fetch(`${API_URL}/capture_frame`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          camera_id: selectedCamera,
          model_type: selectedModel,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error capturing frame:", errorData);
        setError(
          `Failed to capture frame: ${errorData.detail || "Unknown error"}`
        );
        setCaptureStatus(null);
        return;
      }

      const data = await response.json();
      setCaptureStatus("Captured!");
      console.log(`Frame captured successfully: ${data.filename}`);

      setTimeout(() => {
        setCaptureStatus(null);
      }, 3000);
    } catch (err) {
      console.error("Network error during capture:", err);
      setError("Network error when capturing frame");
      setCaptureStatus(null);
    }
  };

  const handlePlayPause = async () => {
    setError(null);
    try {
      const response = await fetch(`${API_URL}/toggle_pause`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Error toggling pause:", errorData);
        setError(
          `Failed to toggle pause: ${errorData.detail || "Unknown error"}`
        );
        return;
      }

      const data = await response.json();
      setIsPaused(data.paused);
      console.log(`Video ${data.paused ? "paused" : "resumed"}`);
    } catch (err) {
      console.error("Network error during pause toggle:", err);
      setError("Network error when toggling pause");
    }
  };

  return (
    <div className="control-section">
      <select
        className="control-dropdown"
        value={selectedModel}
        onChange={handleModelChange}
      >
        <option value="yolo11">YOLO11</option>
        <option value="yolo26">YOLO26n</option>
        <option value="rtdetrv2">RTDETRv2</option>
        <option value="faster_rcnn">Faster-RCNN</option>
      </select>

      <select
        className="control-dropdown"
        value={selectedCamera}
        onChange={handleCameraChange}
      >
        {cameras.map((cam) => (
          <option key={cam.id} value={String(cam.id)}>
            {cam.name}
          </option>
        ))}
      </select>

      <button
        className={`control-btn ${isOn ? "active" : ""}`}
        onClick={toggleOnOff}
      >
        <FaPowerOff /> {isOn ? "ON" : "OFF"}
      </button>

      <button
        className={`control-btn ${activePanel === "edit" ? "active" : ""}`}
        onClick={() => togglePanel("edit")}
      >
        <FaCog /> Settings
      </button>

      <button
        className="control-btn"
        onClick={handleCapture}
        disabled={captureStatus === "Capturing..."}
      >
        <FaCamera /> {captureStatus || "Capture"}
      </button>

      <button
        className={`control-btn ${isPaused ? "active" : ""}`}
        onClick={handlePlayPause}
      >
        {isPaused ? <FaPlay /> : <FaPause />} {isPaused ? "Resume" : "Pause"}
      </button>

      <button
        className={`control-btn ${activePanel === "log" ? "active" : ""}`}
        onClick={() => togglePanel("log")}
      >
        <FaList /> Logs
      </button>
    </div>
  );
}

// import React, { useState, useEffect } from "react";
// import {
//   FaCamera,
//   FaCog,
//   FaPowerOff,
//   FaList,
//   FaPause,
//   FaPlay,
// } from "react-icons/fa";
// import "./ControlSection.css";
// import { useLocation, useNavigate } from "react-router-dom";
// import { useData } from "../contexts/DataProvider";

// // Object to store camera-specific settings
// const cameraSettings = {};

// export default function ControlSection({ togglePanel, activePanel, cameraId }) {
//   const navigate = useNavigate();
//   const location = useLocation();
//   const [isOn, setIsOn] = useState(true);
//   const { cameras } = useData();
//   const [selectedModel, setSelectedModel] = useState("yolo11");
//   const [selectedCamera, setSelectedCamera] = useState("1");
//   const [error, setError] = useState(null);
//   const [captureStatus, setCaptureStatus] = useState(null);
//   const [isPaused, setIsPaused] = useState(false);

//   const toggleOnOff = async () => {
//     setError(null);
//     const newState = !isOn;

//     try {
//       const response = await fetch(`${API_URL}/toggle_annotations", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ show_annotations: newState }),
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         console.error("Error toggling annotations:", errorData);
//         setError(
//           `Failed to toggle annotations: ${errorData.detail || "Unknown error"}`
//         );
//         return;
//       }

//       const data = await response.json();
//       setIsOn(data.show_annotations);

//       // Save camera-specific annotation setting
//       saveCurrentCameraSettings({ isOn: data.show_annotations });

//       console.log(
//         `Annotations ${data.show_annotations ? "enabled" : "disabled"}`
//       );
//     } catch (err) {
//       console.error("Network error during annotation toggle:", err);
//       setError("Network error when toggling annotations");
//     }
//   };

//   const isMonitorPage = location.pathname.includes("/monitor");

//   // Save current camera settings
//   const saveCurrentCameraSettings = (settings) => {
//     const currentCameraId = selectedCamera;
//     cameraSettings[currentCameraId] = {
//       ...(cameraSettings[currentCameraId] || {}),
//       ...settings,
//     };
//     console.log(
//       `Saved settings for camera ${currentCameraId}:`,
//       cameraSettings[currentCameraId]
//     );
//   };

//   // Load settings for a specific camera
//   const loadCameraSettings = (cameraId) => {
//     const settings = cameraSettings[cameraId];
//     if (settings) {
//       console.log(`Loading saved settings for camera ${cameraId}:`, settings);
//       if (settings.isOn !== undefined) setIsOn(settings.isOn);
//       if (settings.model) setSelectedModel(settings.model);
//       return true;
//     }
//     return false;
//   };

//   useEffect(() => {
//     const initializeCamera = async () => {
//       if (isMonitorPage) {
//         const newCameraId = cameraId ? String(cameraId) : "1";
//         setSelectedCamera(newCameraId);

//         // Try to load saved settings for this camera
//         const hasSettings = loadCameraSettings(newCameraId);

//         // If no saved settings, initialize with defaults
//         if (!hasSettings) {
//           setIsOn(true);
//           setSelectedModel("yolo11");
//         }

//         try {
//           // Apply current model setting
//           await fetch(`${API_URL}/set_model", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ model_type: selectedModel }),
//           });

//           // Set camera without resetting settings
//           await fetch(`${API_URL}/set_camera", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ camera_id: newCameraId }),
//           });

//           // Apply current annotation setting
//           await fetch(`${API_URL}/toggle_annotations", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ show_annotations: isOn }),
//           });

//           console.log("Camera initialized with settings:", {
//             camera: newCameraId,
//             model: selectedModel,
//             annotations: isOn,
//           });
//         } catch (err) {
//           console.error("Error initializing camera:", err);
//           setError("Failed to initialize camera");
//         }
//       }
//     };

//     initializeCamera();

//     return () => {
//       // Save settings when unmounting
//       if (selectedCamera) {
//         saveCurrentCameraSettings({
//           isOn,
//           model: selectedModel,
//           isPaused,
//         });
//       }
//     };
//   }, [isMonitorPage, cameraId]);

//   useEffect(() => {
//     const fetchPauseState = async () => {
//       try {
//         const response = await fetch(`${API_URL}/toggle_pause", {
//           method: "POST",
//           headers: {
//             "Content-Type": "application/json",
//           },
//           // Send empty body to avoid toggling the state
//           body: JSON.stringify({}),
//         });

//         if (response.ok) {
//           const data = await response.json();
//           setIsPaused(data.paused);
//         }
//       } catch (err) {
//         console.error("Error fetching pause state:", err);
//       }
//     };

//     // if (isMonitorPage) {
//     //   fetchPauseState();
//     // }
//   }, [location.pathname]);

//   useEffect(() => {
//     if (cameraId) setSelectedCamera(String(cameraId));
//   }, [cameraId]);

//   const handleModelChange = async (e) => {
//     const model_type = e.target.value;
//     setSelectedModel(model_type);
//     setError(null);

//     try {
//       const response = await fetch(`${API_URL}/set_model", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ model_type }),
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         console.error("Error switching model:", errorData);
//         setError(
//           `Failed to switch model: ${errorData.detail || "Unknown error"}`
//         );
//       } else {
//         // Save camera-specific model setting
//         saveCurrentCameraSettings({ model: model_type });
//       }
//     } catch (err) {
//       console.error("Network error:", err);
//       setError("Network error when switching model");
//     }
//   };

//   const handleCameraChange = async (e) => {
//     const camera_id = e.target.value;

//     // Save current camera settings before switching
//     saveCurrentCameraSettings({
//       isOn,
//       model: selectedModel,
//       isPaused,
//     });

//     setSelectedCamera(camera_id);
//     navigate(`/monitor/${camera_id}`);
//     setError(null);

//     try {
//       const response = await fetch(`${API_URL}/set_camera", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ camera_id }),
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         console.error("Error switching camera:", errorData);
//         setError(
//           `Failed to switch camera: ${errorData.detail || "Unknown error"}`
//         );
//       } else {
//         // Load settings for the selected camera
//         loadCameraSettings(camera_id);
//       }
//     } catch (err) {
//       console.error("Network error:", err);
//       setError("Network error when switching camera");
//     }
//   };

//   const handleCapture = async () => {
//     setError(null);
//     setCaptureStatus("Capturing...");

//     try {
//       const response = await fetch(`${API_URL}/capture_frame", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           camera_id: selectedCamera,
//           model_type: selectedModel,
//         }),
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         console.error("Error capturing frame:", errorData);
//         setError(
//           `Failed to capture frame: ${errorData.detail || "Unknown error"}`
//         );
//         setCaptureStatus(null);
//         return;
//       }

//       const data = await response.json();
//       setCaptureStatus("Captured!");
//       console.log(`Frame captured successfully: ${data.filename}`);

//       setTimeout(() => {
//         setCaptureStatus(null);
//       }, 3000);
//     } catch (err) {
//       console.error("Network error during capture:", err);
//       setError("Network error when capturing frame");
//       setCaptureStatus(null);
//     }
//   };

//   const handlePlayPause = async () => {
//     setError(null);
//     try {
//       const response = await fetch(`${API_URL}/toggle_pause", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         console.error("Error toggling pause:", errorData);
//         setError(
//           `Failed to toggle pause: ${errorData.detail || "Unknown error"}`
//         );
//         return;
//       }

//       const data = await response.json();
//       setIsPaused(data.paused);

//       // Save camera-specific pause state
//       saveCurrentCameraSettings({ isPaused: data.paused });

//       console.log(`Video ${data.paused ? "paused" : "resumed"}`);
//     } catch (err) {
//       console.error("Network error during pause toggle:", err);
//       setError("Network error when toggling pause");
//     }
//   };

//   return (
//     <div className="control-section">
//       <select
//         className="control-dropdown"
//         value={selectedModel}
//         onChange={handleModelChange}
//       >
//         <option value="yolo11">YOLO11</option>
//         <option value="rtdetrv2">RTDETRv2</option>
//         <option value="faster_rcnn">Faster-RCNN</option>
//       </select>

//       <select
//         className="control-dropdown"
//         value={selectedCamera}
//         onChange={handleCameraChange}
//       >
//         {cameras.map((cam) => (
//           <option key={cam.id} value={String(cam.id)}>
//             {cam.name}
//           </option>
//         ))}
//       </select>

//       <button
//         className={`control-btn ${isOn ? "active" : ""}`}
//         onClick={toggleOnOff}
//       >
//         <FaPowerOff /> {isOn ? "ON" : "OFF"}
//       </button>

//       <button
//         className={`control-btn ${activePanel === "edit" ? "active" : ""}`}
//         onClick={() => togglePanel("edit")}
//       >
//         <FaCog /> Settings
//       </button>

//       <button
//         className="control-btn"
//         onClick={handleCapture}
//         disabled={captureStatus === "Capturing..."}
//       >
//         <FaCamera /> {captureStatus || "Capture"}
//       </button>

//       <button
//         className={`control-btn ${isPaused ? "active" : ""}`}
//         onClick={handlePlayPause}
//       >
//         {isPaused ? <FaPlay /> : <FaPause />} {isPaused ? "Resume" : "Pause"}
//       </button>

//       <button
//         className={`control-btn ${activePanel === "log" ? "active" : ""}`}
//         onClick={() => togglePanel("log")}
//       >
//         <FaList /> Logs
//       </button>
//     </div>
//   );
// }
