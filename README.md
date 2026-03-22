# 🎮 Virtual Gesture Steering Wheel

A real-time computer vision application that transforms hand gestures captured via webcam into keyboard inputs — turning your hands into a virtual steering wheel for any driving game or simulator.

---

## 🛠️ Tech Stack

**CV & Detection:** Python, OpenCV, MediaPipe (Hand Landmark Detection)  
**Input Simulation:** pynput (keyboard controller)  
**Math:** NumPy, Python `math` (EWMA smoothing, nonlinear curve)

---

## ✨ Features

- 🤚 **Gesture Recognition** — Detects fist (accelerate), open palm (brake), and hand height difference (steering) in real time
- 🎯 **Virtual Steering** — Raising one hand and lowering the other simulates a steering wheel; mapped to Left/Right arrow keys
- 📐 **Calibration** — Press `c` with both hands level to set a neutral baseline, eliminating drift
- 📉 **Smooth Input** — EWMA smoothing + nonlinear soft curve (deadzone + power exponent) for realistic, non-jittery control
- 🖥️ **Live HUD** — Steering gauge bar, per-hand state markers, speed/brake status, and a rotating steering wheel animation rendered over the webcam feed
- ⌨️ **Debounced Key Presses** — Frame-count debouncing prevents accidental key triggers

---

## 🕹️ Controls

| Gesture | Action |
|---|---|
| Raise right hand, lower left | Turn Left |
| Raise left hand, lower right | Turn Right |
| Fist (either hand) | Accelerate (↑) |
| Open Palm (either hand) | Brake (Space) |
| `c` key | Calibrate neutral |
| `q` key | Quit |

---

## 🚀 How to Run

```bash
pip install opencv-python mediapipe pynput
python gesture_wheel.py
```

Requires a working webcam. Run alongside any driving game that uses arrow key controls.

---

## 📁 Project Structure

```
└── gesture_wheel.py    # Complete application — detection, input, and UI in one file
```
