import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque

# Screen setup
SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Smoothing
pos_history = deque(maxlen=5)
smooth_x, smooth_y = SCREEN_W // 2, SCREEN_H // 2

# Click state
last_click_time = 0
CLICK_COOLDOWN = 0.4
was_pinching = False
pinch_start_time = None
pinch_held = False

# Fist activation
fist_times = deque()
FIST_WINDOW = 3.0
prev_fist = False
handsfree_active = False

# Scroll
prev_peace_y = None
scroll_accum = 0.0

# Swipe
swipe_history = deque(maxlen=12)
last_swipe_time = 0
SWIPE_THRESHOLD = 0.18

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6
)

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_finger_states(lm):
    fingers = []
    fingers.append(lm[4].x < lm[3].x)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers

def classify_gesture(lm):
    fingers = get_finger_states(lm)
    thumb, index, middle, ring, pinky = fingers

    if not any(fingers[1:]):
        return "FIST"
    if index and not middle and not ring and not pinky:
        return "POINT"
    if index and middle and not ring and not pinky:
        return "PEACE"
    if all(fingers[1:]):
        return "OPEN"
    return "NONE"

def move_cursor(lm):
    global smooth_x, smooth_y

    ix = lm[8].x
    iy = lm[8].y

    margin = 0.05
    ix = (ix - margin) / (1 - 2 * margin)
    iy = (iy - margin) / (1 - 2 * margin)
    ix = max(0, min(1, ix))
    iy = max(0, min(1, iy))

    raw_x = int(ix * SCREEN_W)
    raw_y = int(iy * SCREEN_H)

    pos_history.append((raw_x, raw_y))
    avg_x = sum(p[0] for p in pos_history) / len(pos_history)
    avg_y = sum(p[1] for p in pos_history) / len(pos_history)

    smooth_x = int(smooth_x + (avg_x - smooth_x) * 0.4)
    smooth_y = int(smooth_y + (avg_y - smooth_y) * 0.4)

    pyautogui.moveTo(smooth_x, smooth_y)

def detect_fist_activation(gesture):
    global prev_fist, handsfree_active, fist_times

    now = time.time()
    if gesture == "FIST" and not prev_fist:
        fist_times.append(now)
        # Clear old fists outside window
        while fist_times and now - fist_times[0] > FIST_WINDOW:
            fist_times.popleft()
        if len(fist_times) >= 3:
            fist_times.clear()
            handsfree_active = True
            print("✅ HandsFree ACTIVATED!")

    prev_fist = (gesture == "FIST")

def handle_scroll(lm):
    global prev_peace_y, scroll_accum

    cy = (lm[8].y + lm[12].y) / 2

    if prev_peace_y is not None:
        delta = prev_peace_y - cy
        scroll_accum += delta

        if abs(scroll_accum) > 0.005:
            pyautogui.scroll(int(scroll_accum * 1200))
            scroll_accum = 0.0

    prev_peace_y = cy

def detect_swipe(lm):
    global last_swipe_time

    cx = sum(l.x for l in lm) / len(lm)
    now = time.time()
    swipe_history.append((cx, now))

    if len(swipe_history) < 8:
        return None
    if now - last_swipe_time < 0.8:
        return None

    recent = list(swipe_history)[-8:]
    time_span = recent[-1][1] - recent[0][1]
    if time_span < 0.15 or time_span > 0.8:
        return None

    x_delta = recent[-1][0] - recent[0][0]
    if abs(x_delta) > SWIPE_THRESHOLD:
        last_swipe_time = now
        swipe_history.clear()
        return "LEFT" if x_delta < 0 else "RIGHT"

    return None

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
esc_pressed = False
prev_active = False

print("Make 3 fists to activate HandsFree! ESC+Q to deactivate.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "NONE"
    now = time.time()
    action_msg = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            gesture = classify_gesture(lm)

            if not handsfree_active:
                # Only watch for fist x3 activation
                detect_fist_activation(gesture)
                fist_count = len(fist_times)
                if fist_count > 0:
                    action_msg = f"Fist count: {fist_count}/3"

            else:
                # ── POINT: move + pinch click ──
                if gesture == "POINT":
                    move_cursor(lm)
                    prev_peace_y = None

                    thumb_tip = (lm[4].x, lm[4].y)
                    index_tip = (lm[8].x, lm[8].y)
                    pinch_dist = dist(thumb_tip, index_tip)
                    is_pinching = pinch_dist < 0.06

                    if is_pinching and not was_pinching:
                        pinch_start_time = now
                        pinch_held = False

                    if is_pinching and pinch_start_time:
                        hold_duration = now - pinch_start_time
                        progress = min(hold_duration / 0.6, 1.0)
                        bar_width = int(200 * progress)
                        cv2.rectangle(frame, (20, 110), (220, 130), (50, 50, 50), -1)
                        cv2.rectangle(frame, (20, 110), (20 + bar_width, 130), (0, 200, 255), -1)
                        if hold_duration > 0.6 and not pinch_held:
                            pinch_held = True
                            action_msg = "SELECTED!"

                    if not is_pinching and was_pinching:
                        hold_duration = now - pinch_start_time if pinch_start_time else 0
                        if hold_duration < 0.6 and (now - last_click_time) > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = now
                            action_msg = "CLICK!"
                        pinch_start_time = None
                        pinch_held = False

                    was_pinching = is_pinching

                # ── PEACE: scroll ──
                elif gesture == "PEACE":
                    handle_scroll(lm)
                    action_msg = "SCROLLING"

                # ── OPEN: swipe detection ──
                elif gesture == "OPEN":
                    prev_peace_y = None
                    swipe = detect_swipe(lm)
                    if swipe == "LEFT":
                        pyautogui.hotkey('alt', 'left')
                        action_msg = "◀ GO BACK"
                        print("◀ Swipe Left - Go Back")
                    elif swipe == "RIGHT":
                        pyautogui.hotkey('alt', 'right')
                        action_msg = "▶ GO FORWARD"
                        print("▶ Swipe Right - Go Forward")

                # ── FIST: placeholder for drag (next step) ──
                elif gesture == "FIST":
                    prev_peace_y = None
                    action_msg = "FIST - drag coming soon!"

    # ── UI OVERLAY ──
    # Mode indicator
    mode_color = (0, 200, 100) if handsfree_active else (50, 50, 200)
    mode_text = "HANDS-FREE MODE" if handsfree_active else "STANDBY - Make 3 fists!"
    cv2.rectangle(frame, (10, 10), (430, 65), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (430, 65), mode_color, 2)
    cv2.putText(frame, mode_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    cv2.putText(frame, f"Gesture: {gesture}", (20, 57),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Action message
    if action_msg:
        cv2.putText(frame, action_msg, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("HandsFree", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        esc_pressed = True
    if key == ord('q') and esc_pressed:
        if handsfree_active:
            handsfree_active = False
            print("🔴 HandsFree DEACTIVATED")
        else:
            break

cap.release()
cv2.destroyAllWindows()