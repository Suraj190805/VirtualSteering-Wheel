# gesture_wheel.py
"""
Medium-sensitivity virtual steering wheel (balanced between snappy and smooth).
- Calibration: press 'c' while holding hands level to set neutral.
- Controls:
    * Raise one hand and lower the other -> steering (realistic wheel motion)
    * Make a fist (any hand) -> accelerate (Up key)
    * Show open palm (any hand) -> brake (Space)
    * q -> quit
Tune knobs near the top if you want a slightly different feel.
"""

import cv2
import mediapipe as mp
import time
import math
from pynput.keyboard import Controller, Key

# ---------- CONFIG / MEDIUM TUNING ----------
LEFT_KEY = Key.left
RIGHT_KEY = Key.right
ACCEL_KEY = Key.up
BRAKE_KEY = Key.space

MIN_DETECT_CONF = 0.5
MIN_TRACK_CONF = 0.5

# Medium (averaged) tuning
ALPHA = 0.31               # smoothing EWMA (0..1)  -- medium responsiveness
STEER_DEADZONE = 0.065     # ignore tiny delta motions
STEER_THRESHOLD = 0.16     # magnitude to trigger left/right holds
SENSITIVITY = 0.775        # global steering gain (0.0 - 1.0) — medium

# Keep nonlinear soft curve exponent from earlier (reduces small inputs)
CURVE_EXP = 1.6

# throttle tuning
DEBOUNCE_FRAMES = 3
NO_HAND_TIMEOUT = 0.6

# UI colors
TEXT_COLOR = (255, 255, 255)
FIST_COLOR = (0, 180, 0)
PALM_COLOR = (0, 120, 255)
NEUTRAL_COLOR = (180, 180, 180)
STEER_BAR_COLOR = (255, 200, 0)

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=MIN_DETECT_CONF,
                       min_tracking_confidence=MIN_TRACK_CONF)
mp_draw = mp.solutions.drawing_utils

# ---------- State ----------
keyboard = Controller()
cap = cv2.VideoCapture(0)

holding_left = holding_right = False
holding_accel = holding_brake = False

smoothed_turn = 0.0
fist_count = palm_count = 0
last_detect_time = time.time()

# calibration state
neutral_offset = 0.0
calibrated = False

# ---------- Helpers ----------
def press(key):
    try: keyboard.press(key)
    except: pass

def release(key):
    try: keyboard.release(key)
    except: pass

def is_fist_or_palm(hand_landmarks, img_w, img_h):
    """Return 'fist'/'palm'/'neutral' using landmark geometry."""
    tips = [4,8,12,16,20]
    mcps = [2,5,9,13,17]
    lm = hand_landmarks.landmark
    wrist = (lm[0].x*img_w, lm[0].y*img_h)
    mid_mcp = (lm[9].x*img_w, lm[9].y*img_h)
    hand_size = math.hypot(mid_mcp[0]-wrist[0], mid_mcp[1]-wrist[1]) or 1.0
    folded, open_ = 0,0
    for t,m in zip(tips,mcps):
        tx,ty = lm[t].x*img_w, lm[t].y*img_h
        mx,my = lm[m].x*img_w, lm[m].y*img_h
        ratio = math.hypot(tx-mx,ty-my)/hand_size
        if ratio < 0.45: folded += 1
        else: open_ += 1
    if folded >= 4: return "fist"
    if open_ >= 4: return "palm"
    return "neutral"

def apply_deadzone_and_curve(raw, deadzone, exp):
    """Apply deadzone, normalize, apply soft curve; returns [-1,1]."""
    if abs(raw) <= deadzone:
        return 0.0
    sign = 1.0 if raw > 0 else -1.0
    mag = (abs(raw) - deadzone) / (1.0 - deadzone)
    curved = mag ** exp
    return sign * curved

def draw_steering_wheel(frame, center, radius, angle_deg):
    """
    Draw a cosmetic steering wheel overlay on the provided frame.
    - center: (x,y) in pixels
    - radius: radius in pixels
    - angle_deg: rotation in degrees (clockwise positive)
    This function uses only OpenCV drawing functions for portability.
    """
    cx, cy = int(center[0]), int(center[1])
    # outer rim
    cv2.circle(frame, (cx, cy), radius, (200,200,200), 6, lineType=cv2.LINE_AA)
    # inner rim
    inner_r = int(radius * 0.6)
    cv2.circle(frame, (cx, cy), inner_r, (120,120,120), 3, lineType=cv2.LINE_AA)
    # spokes - draw 3 spokes rotated by angle_deg
    spoke_lengths = [radius - 8, int(radius*0.7), int(radius*0.5)]
    spoke_angles = [0, 120, 240]  # base angles
    for base in spoke_angles:
        ang = math.radians(base + angle_deg)
        x1 = int(cx + math.cos(ang) * (inner_r - 4))
        y1 = int(cy + math.sin(ang) * (inner_r - 4))
        x2 = int(cx + math.cos(ang) * (radius - 8))
        y2 = int(cy + math.sin(ang) * (radius - 8))
        cv2.line(frame, (x1, y1), (x2, y2), (255,200,80), 4, lineType=cv2.LINE_AA)
    # small center cap
    cv2.circle(frame, (cx, cy), int(radius*0.12), (180,180,180), -1, lineType=cv2.LINE_AA)
    # subtle shadow/outline for better contrast
    cv2.circle(frame, (cx, cy), radius+2, (40,40,40), 1, lineType=cv2.LINE_AA)

print("Starting medium-sensitivity virtual steering wheel. Press 'c' to calibrate, 'q' to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera not available.")
            break
        h,w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        left_y = right_y = None
        hand_states = []
        current_time = time.time()

        if results.multi_hand_landmarks:
            last_detect_time = current_time
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                state = is_fist_or_palm(hand_landmarks, w, h)
                wrist = hand_landmarks.landmark[0]
                cy = wrist.y   # normalized 0..1
                cx = wrist.x
                if label == "Left":
                    left_y = cy
                else:
                    right_y = cy
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_states.append((label, state, int(cx*w), int(cy*h)))
        else:
            # no hands detected — steering will trend to 0 via smoothing
            pass

        # ---------- Calibration hint ----------
        cv2.putText(frame, "Press 'c' to calibrate neutral (both hands level). 'q' to quit.",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # ---------- Steering computation ----------
        steer_val = 0.0
        if left_y is not None and right_y is not None:
            raw_delta = right_y - left_y
            centered = raw_delta - neutral_offset
            centered = max(-1.0, min(1.0, centered))
            processed = apply_deadzone_and_curve(centered, STEER_DEADZONE, CURVE_EXP)
            steer_val = processed * SENSITIVITY
        else:
            steer_val = 0.0

        # smoothing EWMA
        smoothed_turn = ALPHA * steer_val + (1 - ALPHA) * smoothed_turn

        # decide holds
        want_left = smoothed_turn > STEER_THRESHOLD
        want_right = smoothed_turn < -STEER_THRESHOLD

        # apply steering keys (mutually exclusive)
        if want_left and not holding_left:
            press(LEFT_KEY); holding_left = True
        if not want_left and holding_left:
            release(LEFT_KEY); holding_left = False
        if want_right and not holding_right:
            press(RIGHT_KEY); holding_right = True
        if not want_right and holding_right:
            release(RIGHT_KEY); holding_right = False

        # ---------- Throttle / Brake ----------
        any_fist = any(s=="fist" for _,s,_,_ in hand_states)
        any_palm = any(s=="palm" for _,s,_,_ in hand_states)
        if any_fist: fist_count += 1
        else: fist_count = 0
        if any_palm: palm_count += 1
        else: palm_count = 0

        want_accel = fist_count >= DEBOUNCE_FRAMES
        want_brake = (not want_accel) and (palm_count >= DEBOUNCE_FRAMES)

        if want_accel and not holding_accel:
            press(ACCEL_KEY); holding_accel = True
        if not want_accel and holding_accel:
            release(ACCEL_KEY); holding_accel = False
        if want_brake and not holding_brake:
            press(BRAKE_KEY); holding_brake = True
        if not want_brake and holding_brake:
            release(BRAKE_KEY); holding_brake = False

        # ---------- Draw UI ----------
        # steering gauge
        bar_y = int(h * 0.9)
        bar_w = int(w * 0.7)
        bar_x = (w - bar_w) // 2
        cx_pos = int(bar_x + (smoothed_turn + 1) / 2 * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y - 12), (bar_x + bar_w, bar_y + 12), (50,50,50), -1)
        cv2.circle(frame, (cx_pos, bar_y), 14, STEER_BAR_COLOR, -1)
        cv2.putText(frame, f"Steer:{smoothed_turn:+.2f}  Sens:{SENSITIVITY:.3f}", (bar_x, bar_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        # per-hand state markers
        for label, state, cx, cy in hand_states:
            color = FIST_COLOR if state == "fist" else PALM_COLOR if state == "palm" else NEUTRAL_COLOR
            cv2.circle(frame, (cx, cy), 12, color, -1)
            cv2.putText(frame, f"{label}:{state}", (max(5, cx - 60), max(20, cy - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2)

        # overall status
        if want_accel:
            overall = "ACCELERATING (FIST)"
            color = FIST_COLOR
        elif want_brake:
            overall = "BRAKING (PALM)"
            color = PALM_COLOR
        else:
            overall = "IDLE"
            color = NEUTRAL_COLOR
        if want_left: overall += " | TURN LEFT"
        elif want_right: overall += " | TURN RIGHT"
        cv2.putText(frame, overall, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"Calibrated: {calibrated}  Offset: {neutral_offset:.3f}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        # ---------- Steering wheel animation (cosmetic) ----------
        # place wheel top-left (avoid covering HUD)
        wheel_center = (int(w * 0.12), int(h * 0.18))
        wheel_radius = int(min(w,h) * 0.12)
        # map smoothed_turn (-1..1) to degrees for display (-80..+80)
        wheel_angle = smoothed_turn * 80.0
        draw_steering_wheel(frame, wheel_center, wheel_radius, wheel_angle)

        cv2.imshow("Virtual Steering (medium sensitivity) - press c to calibrate", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            if left_y is not None and right_y is not None:
                neutral_offset = right_y - left_y
                calibrated = True
                print(f"Calibrated neutral_offset = {neutral_offset:.4f}")
            else:
                print("Calibration failed: show both hands level to camera then press 'c'.")

except KeyboardInterrupt:
    pass
finally:
    # release held keys
    if holding_left: release(LEFT_KEY)
    if holding_right: release(RIGHT_KEY)
    if holding_accel: release(ACCEL_KEY)
    if holding_brake: release(BRAKE_KEY)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Exited cleanly.")
