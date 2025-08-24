# Intelligent Modeling and Control of Steam Plant Operations with AI

Lake Superior State University (LSSU)  

---

## Project Overview

This project aims to reduce natural gas consumption at LSSU by using Artificial Intelligence (AI) to optimize how the Central Heating Plant (CHP) operates. The CHP manually controls three fire-tube steam boilers that heat 17 campus buildings. Currently, boilers operate continuously at high capacity regardless of building-specific demand or external factors like weather or occupancy.

The project will develop a neural network–driven prediction and recommendation system that:

- Predicts daily heating demand (starting with the Library building)
- Recommends optimal setpoints for CHP operators
- Integrates with historical, real-time, and forecasted data
- Minimizes energy waste during low-demand periods

---
Current Status: 
---

## Goals

- Reduce energy waste by forecasting heating requirements instead of reactive heating.
- Develop predictive AI models to assist human operators with daily scheduling.
- Integrate occupancy, weather, and usage patterns into a reliable demand forecast.
- Prototype system in the Library before scaling campus-wide.

---

## Current Role & Lead Developer

**David Rivard** (CS Major, Student Lead — Fall 2025 → Spring 2026)  
Responsibilities:

- Refactor the AI model to prevent overfitting and improve generalization.
- Develop a minimal viable model for daily heating prediction.
- Implement and test OCR pipelines for reading analog gas meter values using real-time camera images.
- Clean and organize legacy code into a modular GitHub structure.
- Manage version control, documentation, and reproducibility.
- Prepare and assist with publication of a future research paper.

---

## AI Model Architecture (Planned / Current)

| Component          | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| Model Type         | TensorFlow-based regression models (currently under refinement)        |
| Input Variables    | Weather data, occupancy trends, time/day, historical heating logs      |
| Target Variable    | Predicted steam demand or setpoint adjustment                          |
| Output             | Daily setpoint or load recommendation for CHP operators                |
| Validation         | Train/test split by season; cross-checked with actual steam usage      |
| Evaluation Metrics | Mean Squared Error (MSE), prediction error plots, temporal correlation |

---

## OCR Pipeline for Gas Meter Reading

| Component  | Description                                                                |
| ---------- | -------------------------------------------------------------------------- |
| Source     | Real-time images from fixed camera at CHP gas meter                        |
| Problem    | Analog dial readings suffer from lighting issues, character similarity     |
| Tools      | Tesseract OCR with OpenCV preprocessing; fallback image differencing       |
| Safeguards | Multiple frame analysis; plausibility checks vs. prior reading             |
| Next Steps | Train digit-only OCR model if needed; fix flashlight trigger control       |

---

## Data Sources

| Source         | Description                                               |
| -------------- | --------------------------------------------------------- |
| Metasys        | Johnson Controls BMS — historical sensor and boiler data  |
| NOAA XML       | Weather forecasts and history                             |
| Occupancy Cams | Cameras in Library used to estimate hourly human activity |
| Indoor Sensors | Temperature & humidity sensors inside prototype building  |
| Gas Meter      | Analog dial image processed by OCR                        |
| Google Drive   | Main repository of backups, logs, scripts, and data       |

---

## Repository Structure

```
project-root/
├── ocr/
│   ├── meter_reader.py
│   ├── tesseract_pipeline.py
│   └── test_images/
├── models/
│   ├── dt_library_ai.py
│   ├── model_tester.py
│   └── saved_models/
├── data_ingest/
│   ├── fetch_noaa_data.py
│   ├── metasys_pull.py
│   └── occupancy_watchdog.py
├── notebooks/
│   └── exploratory_visuals.ipynb
├── utils/
│   ├── image_preprocessing.py
│   └── flash_controller.py
├── results/
│   ├── graphs/
│   └── template/
├── requirements.txt
└── README.md
```

---

## Validation & Visualization

All trained models must be tested via `model_tester.py`, which:

- Loads a model and test set
- Outputs prediction plots and MSE
- Stores visualizations for reporting and reviews

---

## Known Issues / Work in Progress

| Area            | Status      | Notes                                                             |
| --------------- | ----------- | ----------------------------------------------------------------- |
| AI Overfitting  | In Progress | Working on baseline model with reduced features                   |
| OCR Accuracy    | Critical    | Tesseract implementation in progress; flashlight control buggy    |
| Legacy Code     | Refactor    | Code cleanup and Git migration in progress                        |
| Data Interrupts | Ongoing     | Some sensors fail during holidays or off-hours; watchdog in place |
| Flashlight      | Investigate | Flash runs continuously—should only trigger during image capture  |

---

## Communication

- Lead Faculty: Dr. Mahmood (keyholder, project PI)
- Project Email (TBD): [dte_project@lssu.edu](mailto:dte_project@lssu.edu)
- Documents: All meetings, minutes, proposals in shared Google Drive

---

## Research Publication (Planned)

A research paper is in preparation for Spring 2026. Key contributions:

- AI-driven heating load forecasting in mixed-use academic buildings
- Application of computer vision for utility metering in retrofitted systems
- Comparative performance vs. baseline CHP operation

---

## Project Timeline (Fall 2025)

| Milestone                        | Target Date        |
| -------------------------------- | ------------------ |
| MVP AI Model Working             | Mid-September 2025 |
| OCR Pipeline Operational         | Early October 2025 |
| Model Integration with Live Data | November 2025      |
| Paper Drafting Begins            | December 2025      |

---

## License

Licensing TBD based on final publication and data-sharing agreements.

---

## Acknowledgments

- DTE Energy for project funding
- Cloverland Electric Cooperative
- LSSU Faculty: Dr. Mahmood, Dr. Smith, and the Mechanical Engineering Team
- Riley, prior student engineer, for initial system implementation
