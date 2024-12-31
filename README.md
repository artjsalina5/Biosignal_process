# Stress Detection Using Empatica E4 and Hexoskin Biosignals

## Overview

This project focuses on developing a scalable **stress detection model** that integrates biosignals from the **Empatica E4** and **Hexoskin** wearable devices. Unlike existing methods, which primarily focus on individual-level detection, our approach is designed to scale for **team task training simulators**. These simulators aim to identify stress and fatigue-related performance issues during critical tasks, such as procedural compliance in military or high-stakes environments.

By leveraging **tinygrad**, we aim to build a computationally efficient stress detection pipeline that can:
1. Scale to monitor entire teams in real-time.
2. Identify specific moments of procedural deviation (e.g., when fatigue or stress compromises performance).
3. Incorporate long-term baselines to flag serious mental health concerns (e.g., burnout or suicide risk).

---

## Project Goals

1. **Signal Integration**: Combine biosignals from Empatica E4 (e.g., EDA, PPG, IBI) and Hexoskin (e.g., respiration, ECG) into a unified stress detection model.
2. **Team-Based Scalability**: Develop a pipeline capable of monitoring multiple users simultaneously during task training.
3. **Procedural Compliance Detection**: Pinpoint deviations from expected behavior due to stress, fatigue, or mental health deterioration.
4. **Long-Term Monitoring**: Integrate baseline data to track individuals over time and flag significant deviations.

---

## Primary Contributor

This project is spearheaded by **Arturo Salinas-Aguayo**, a Computer Engineering student with extensive experience in biosignal processing and lightweight neural networks. Arturo's expertise uniquely positions this project to bridge computational efficiency with real-world applicability in high-stakes environments.

GitHub Profile: [Arturo Salinas-Aguayo](https://github.com/artjsalina5)

---

## Key Features

1. **Multivariate Signal Fusion**:
   - Real-time synchronization of Empatica E4 and Hexoskin biosignals.
   - Unified processing of ECG, EDA, PPG, and respiration data.

2. **Tinygrad-Based Stress Model**:
   - A lightweight, scalable neural network tailored for real-time deployment.
   - Incorporates Bayesian uncertainty estimation to enhance reliability.

3. **Team Training Simulators**:
   - Multi-user monitoring to evaluate team-level task performance under stress.
   - Detection of procedural compliance deviations (e.g., failure to follow steps in high-stress scenarios).

4. **Baseline & Long-Term Monitoring**:
   - Compare current stress levels with individual baselines.
   - Track changes over time to flag fatigue, mental health deterioration, or burnout.

---

## Current Methods and Limitations

### Existing Approaches

1. **Single-User Stress Detection**:
   - Primarily focused on detecting stress at an individual level in controlled lab environments.
   - Methods like heart rate variability (HRV) analysis, skin conductance levels (SCL), and respiratory patterns are commonly used.
   - Tools like the **Empatica Research Suite** and **Hexoskin SDK** provide raw data but lack integrated stress models or real-time analytics.

2. **Team-Based Monitoring**:
   - Limited computational resources make scaling for multiple users challenging.
   - Systems like **TeamSense** focus on basic physiological aggregation (e.g., average HRV) without detailed individual or task-specific insights.

3. **Procedural Compliance Systems**:
   - Relies on manual observations or post-task analysis, which are time-intensive and prone to bias.
   - Rarely integrates physiological data, missing the connection between stress and deviations in performance.

### Shortcomings in Biosignal Analysis Theory for Stress/Arousal

1. **Generalization**:
   - Current models often assume universal physiological responses to stress (e.g., increased heart rate or skin conductance). 
   - Stress responses vary significantly across individuals due to factors like fitness, mental health, and context, making one-size-fits-all models unreliable.

2. **Contextual Sensitivity**:
   - Biosignals alone cannot differentiate between physical exertion and psychological stress. For example, an elevated heart rate may result from running or anxiety.

3. **Temporal Dynamics**:
   - Most models analyze signals in static time windows, missing the importance of long-term trends and baseline shifts.

4. **Signal Interference**:
   - Biosignals are highly susceptible to artifacts (e.g., motion, environmental noise). Current preprocessing methods often discard valuable data to ensure "clean" signals.

5. **Lack of Task-Specific Models**:
   - Stress levels vary depending on task difficulty, cognitive load, and team interactions. Current approaches rarely incorporate task-specific baselines or interdependencies.

6. **Scalability**:
   - Current models do not address the complexity of real-time team-based analysis, where multiple individuals must be monitored simultaneously.

---

## How Our Approach Differs

1. **Scalability**:
   - Lightweight **tinygrad** models optimized for simultaneous monitoring of multiple team members.
   - Efficient signal synchronization ensures real-time processing.

2. **Procedural Compliance Detection**:
   - Automatically flags deviations during task execution (e.g., skipping steps in high-stress scenarios).
   - Links biosignal metrics (e.g., high HRV, irregular breathing) to specific moments of non-compliance.

3. **Baseline Comparisons**:
   - Incorporates individual baselines to assess relative stress levels.
   - Tracks changes over time to flag fatigue, burnout, or mental health deterioration.

4. **Psychological Safety Monitoring**:
   - Models long-term trends to identify sailors or team members at risk of mental health crises.
   - Combines subjective inputs ("I’m fine") with objective physiological data to override potential bias.

---

## Workflow

### 1. Biosignal Acquisition
   - **Empatica E4**: Collect signals such as:
     - **EDA (Electrodermal Activity)**: Skin conductance levels.
     - **PPG (Photoplethysmography)**: Pulse rate variability.
     - **IBI (Inter-Beat Interval)**: Heart rate variability.
   - **Hexoskin**: Collect signals such as:
     - **ECG (Electrocardiogram)**: R-peak intervals for HRV.
     - **Respiration**: Breathing rate and tidal volume.

### 2. Signal Synchronization
   - Align Empatica E4 and Hexoskin signals using timestamps.
   - Segment signals by task markers (e.g., "easy," "baseline," "matb hard").

### 3. Feature Extraction
   - Extract stress-relevant metrics (e.g., HRV, SCL, PRV) for each segment.
   - Compute task-specific baselines.

### 4. Model Development
   - Use **tinygrad** to train a neural network that:
     - Learns latent representations of biosignals.
     - Incorporates Bayesian uncertainty for robust predictions.

### 5. Team-Based Stress Analysis
   - Monitor stress levels for each team member during training.
   - Aggregate data to detect task-level stress trends and procedural compliance issues.

### 6. Long-Term Monitoring
   - Compare individual baselines over weeks/months.
   - Flag sailors in the "red zone" based on cumulative stress and deviations from their baseline.

---

## Potential Applications

### 1. Military Training
   - Identify sailors who may underperform due to fatigue or high stress.
   - Pinpoint procedural compliance failures caused by physiological deterioration.

### 2. Crisis Management
   - Monitor teams in high-stakes scenarios (e.g., nuclear reactor control, combat operations).
   - Provide real-time feedback to supervisors on team stress levels.

### 3. Mental Health Intervention
   - Flag sailors at risk of mental health crises based on longitudinal stress data.
   - Enable proactive interventions, reducing risks of burnout or suicide.

---

## Future Directions

1. **Advanced Behavioral Models**:
   - Incorporate behavioral data (e.g., task completion time, error rates) into the stress model.

2. **Improved Scalability**:
   - Explore hardware acceleration (e.g., NVIDIA Jetson or Google Coral) for larger team simulations.

3. **Field Testing**:
   - Deploy the model in real-world scenarios (e.g., submarine crews, tactical operations) to validate its performance.

4. **Individualized Insights**:
   - Develop adaptive models that account for user-specific factors (e.g., age, fitness level, baseline variability).

---

## Acknowledgments

This project builds on years of research in biosignal processing, wearable technology, and team performance optimization. Special thanks to contributors in the fields of computational neuroscience and embedded systems for inspiring this scalable solution.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.