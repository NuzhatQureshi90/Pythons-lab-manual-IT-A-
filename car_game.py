import cv2
import mediapipe as mp
import numpy as np
import random
import pygame

# Initialize pygame for music
pygame.init()
pygame.mixer.music.load("background_music.mp3")
pygame.mixer.music.play(-1)

# Load images
player_car = cv2.imread("assets/player_car.png", cv2.IMREAD_UNCHANGED)
obstacle_car = cv2.imread("assets/obstacle_car.png", cv2.IMREAD_UNCHANGED)
road = cv2.imread("assets/road.png")

frame_width, frame_height = 640, 480
road = cv2.resize(road, (frame_width, frame_height * 2))  # Taller image for scrolling

# Resize cars
player_car = cv2.resize(player_car, (60, 100))
obstacle_car = cv2.resize(obstacle_car, (60, 100))

# MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Game variables
car_x = 290
car_y = 360
car_speed = 20
car_moving = True
obstacles = []
obstacle_speed = 5
obstacle_spawn_delay = 40
frame_count = 0
score = 0
game_over = False
scroll_y = 0

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Overlay function
def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg
    if fg.shape[2] == 3:
        bg[y:y+h, x:x+w] = fg
    else:
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = fg[:, :, c] * alpha + bg[y:y+h, x:x+w, c] * (1 - alpha)
    return bg

# Reset game
def reset_game():
    global car_x, car_y, obstacles, score, game_over, car_moving, frame_count, obstacle_speed, obstacle_spawn_delay, scroll_y
    car_x = 290
    car_y = 360
    obstacles = []
    score = 0
    game_over = False
    car_moving = True
    frame_count = 0
    obstacle_speed = 5
    obstacle_spawn_delay = 40
    scroll_y = 0

# Main game loop
while True:
    success, cam_frame = cap.read()
    if not success:
        break
    cam_frame = cv2.flip(cam_frame, 1)
    img_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Scroll road
    scroll_y = (scroll_y + obstacle_speed) % frame_height
    game_frame = road[scroll_y:scroll_y + frame_height].copy()

    # Hand tracking
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(cam_frame, handLms, mp_hands.HAND_CONNECTIONS)
            lm_list = [(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in handLms.landmark]
            if lm_list:
                wrist_x, wrist_y = lm_list[0]
                palm_base_y = lm_list[9][1]
                car_moving = palm_base_y <= wrist_y
                if wrist_x < frame_width // 3 and car_moving:
                    car_x -= car_speed
                elif wrist_x > 2 * frame_width // 3 and car_moving:
                    car_x += car_speed

    car_x = max(0, min(car_x, frame_width - player_car.shape[1]))

    # Obstacles
    frame_count += 1
    if frame_count % obstacle_spawn_delay == 0:
        new_x = random.randint(0, frame_width - obstacle_car.shape[1])
        obstacles.append([new_x, -obstacle_car.shape[0]])

    new_obstacles = []
    for ox, oy in obstacles:
        oy += obstacle_speed
        if oy < frame_height:
            new_obstacles.append([ox, oy])
        if (ox < car_x + player_car.shape[1] and ox + obstacle_car.shape[1] > car_x and
            oy < car_y + player_car.shape[0] and oy + obstacle_car.shape[0] > car_y):
            game_over = True
        overlay_image(game_frame, obstacle_car, ox, oy)
    obstacles = new_obstacles

    overlay_image(game_frame, player_car, car_x, car_y)

    score += 1
    cv2.putText(game_frame, f"Score: {score}", (10, 40), font, 1, (255, 255, 255), 2)

    if score % 100 == 0:
        obstacle_speed += 1
        obstacle_spawn_delay = max(20, obstacle_spawn_delay - 5)

    # Game Over screen
    if game_over:
        cv2.putText(game_frame, "GAME OVER", (180, 240), font, 2, (0, 0, 255), 5)
        cv2.putText(game_frame, f"Final Score: {score}", (180, 300), font, 1, (255, 255, 255), 2)
        cv2.putText(game_frame, "Press R to Restart", (180, 350), font, 1, (255, 255, 255), 2)
        cv2.imshow("Game", game_frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            reset_game()
            continue
        else:
            break

    # Display frames
    cv2.imshow("Hand Detection", cam_frame)
    cv2.imshow("Game", game_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


