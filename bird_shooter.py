import cv2
import mediapipe as mp
import pygame
import random
import numpy as np
import time
import os
import sqlite3

class Bird:
    def __init__(self, size, speed, points, pos=None, direction=None):
        self.size = size
        self.speed = speed
        self.points = points
        self.pos = pos if pos else [0, 0]
        self.direction = direction if direction else random.choice([-1, 1])
        self.velocity_y = 0
        self.wing_up = True
        self.last_jump_time = 0

class BirdShooterGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera!")

        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Bird Shooter")
        
        self.start_time = None
        self.last_background_change = time.time()
        self.background_change_interval = 60
        self.current_time_of_day = 'morning'
        
        self.game_duration = 60
        self.game_over = False
        self.current_level = 1
        
        self.bird_types = {
            'small': {'size': 25, 'base_speed': 5, 'points': 30, 'spawn_weight': 0.2},
            'medium': {'size': 40, 'base_speed': 3, 'points': 20, 'spawn_weight': 0.5},
            'large': {'size': 55, 'base_speed': 2, 'points': 10, 'spawn_weight': 0.3}
        }
        
        self.active_birds = []
        self.max_birds = 3
        
        self.crosshair_size = 20
        self.score = 0
        self.main_font = pygame.font.Font(None, 36)
        self.large_font = pygame.font.Font(None, 72)
        self.retro_font = pygame.font.Font(None, 48)
        
        self.gravity = 0.5
        self.jump_strength = -8
        self.jump_cooldown = 1.0
        
        self.wing_delay = 0.2
        self.last_wing_time = time.time()
        
        self.explosions = []
        self.explosion_max_frames = 15
        
        self.crosshair_pos = [self.width//2, self.height//2]
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        
        self.running = True
        self.is_shooting = False
        self.last_shoot_time = 0
        self.shoot_cooldown = 0.5
        
        self.show_home_screen = True
        self.btn_color = (255, 165, 0)
        self.btn_hover_color = (255, 215, 0)
        self.btn_hover = False
        self.last_bird_spawn_time = time.time()
        self.home_screen_birds = []
        self.home_screen_background = self.create_background('morning')
        
        self.show_level_select = False
        self.next_level_btn_rect = None
        self.quit_btn_rect = None
        
        # Player name and database setup
        self.player_name = ""
        self.input_active = True
        self.setup_database()

    def setup_database(self):
        self.conn = sqlite3.connect('bird_shooter_scores.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                score INTEGER NOT NULL
            )
        ''')
        self.conn.commit()

    def save_score(self):
        self.cursor.execute('INSERT INTO leaderboard (name, score) VALUES (?, ?)', (self.player_name, self.score))
        self.conn.commit()
        # Keep only top 5 scores
        self.cursor.execute('DELETE FROM leaderboard WHERE id NOT IN (SELECT id FROM leaderboard ORDER BY score DESC LIMIT 5)')
        self.conn.commit()

    def get_leaderboard(self):
        self.cursor.execute('SELECT name, score FROM leaderboard ORDER BY score DESC LIMIT 5')
        return self.cursor.fetchall()

    def draw_cloud(self, surface, x, y):
        color = (255, 255, 255)
        pygame.draw.circle(surface, color, (x, y), 20)
        pygame.draw.circle(surface, color, (x + 15, y - 10), 15)
        pygame.draw.circle(surface, color, (x + 15, y + 10), 15)
        pygame.draw.circle(surface, color, (x + 30, y), 20)

    def process_hands(self):
        success, image = self.cap.read()
        if not success:
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        
        results = self.hands.process(image)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            index_x = int(hand_landmarks.landmark[8].x * self.width)
            index_y = int(hand_landmarks.landmark[8].y * self.height)
            
            thumb_x = int(hand_landmarks.landmark[4].x * self.width)
            thumb_y = int(hand_landmarks.landmark[4].y * self.height)
            
            self.crosshair_pos = [index_x, index_y]
            
            if self.show_home_screen and not self.input_active:
                btn_rect = pygame.Rect(self.width//2 - 100, self.height//2 + 50, 200, 70)
                if btn_rect.collidepoint(index_x, index_y):
                    self.btn_hover = True
                    distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                    if distance < 30:
                        self.show_home_screen = False
                        self.start_time = time.time()
                else:
                    self.btn_hover = False
                return
            
            if self.show_level_select:
                distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                if distance < 30 and time.time() - self.last_shoot_time >= self.shoot_cooldown:
                    if self.next_level_btn_rect.collidepoint(index_x, index_y):
                        self.current_level += 1
                        self.start_new_level()
                    elif self.quit_btn_rect.collidepoint(index_x, index_y):
                        self.running = False
                return
            
            distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            current_time = time.time()
            if distance < 30 and current_time - self.last_shoot_time >= self.shoot_cooldown:
                self.is_shooting = True
                self.last_shoot_time = current_time
            else:
                self.is_shooting = False

    def create_background(self, time_of_day):
        background = pygame.Surface((self.width, self.height))
        
        sky_colors = {
            'morning': (135, 206, 235),
            'afternoon': (255, 171, 87),
            'night': (25, 25, 112)
        }
        
        grass_colors = {
            'morning': (34, 139, 34),
            'afternoon': (85, 107, 47),
            'night': (0, 100, 0)
        }
        
        background.fill(sky_colors[time_of_day])
        grass_height = 100
        pygame.draw.rect(background, grass_colors[time_of_day],
                        (0, self.height - grass_height, self.width, grass_height))
        
        if time_of_day == 'night':
            for _ in range(50):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height - grass_height)
                pygame.draw.circle(background, (255, 255, 255), (x, y), 1)
        else:
            for _ in range(5):
                x = random.randint(0, self.width)
                y = random.randint(0, self.height//3)
                self.draw_cloud(background, x, y)
        
        return background

    def update_background(self):
        current_time = time.time()
        if current_time - self.last_background_change >= self.background_change_interval:
            times = ['morning', 'afternoon', 'night']
            current_index = times.index(self.current_time_of_day)
            self.current_time_of_day = times[(current_index + 1) % len(times)]
            self.background = self.create_background(self.current_time_of_day)
            self.last_background_change = current_time

    def get_level_speed(self, base_speed):
        speed_multiplier = 1 + (self.current_level - 1) * 1.0  
        return base_speed * speed_multiplier

    def spawn_bird(self):
        bird_type = random.choices(list(self.bird_types.keys()),
                                 weights=[self.bird_types[t]['spawn_weight'] for t in self.bird_types])[0]
        props = self.bird_types[bird_type]
        speed = self.get_level_speed(props['base_speed'])
        
        direction = random.choice([-1, 1])
        if direction == 1:
            x = -props['size']
        else:
            x = self.width + props['size']
        y = random.randint(50, self.height - 100 - props['size'])
        
        bird = Bird(props['size'], speed, props['points'], [x, y], direction)
        return bird

    def update_birds(self):
        self.active_birds = [bird for bird in self.active_birds if not (
            (bird.direction == 1 and bird.pos[0] > self.width + bird.size) or
            (bird.direction == -1 and bird.pos[0] < -bird.size)
        )]
        
        while len(self.active_birds) < self.max_birds:
            self.active_birds.append(self.spawn_bird())
        
        current_time = time.time()
        jump_probability = 0.1 + (self.current_level - 1) * 0.15
        jump_cooldown = max(0.3, 1.0 - (self.current_level - 1) * 0.2)
        
        for bird in self.active_birds:
            bird.pos[0] += bird.speed * bird.direction
            bird.velocity_y += self.gravity
            bird.pos[1] += bird.velocity_y
            
            if current_time - bird.last_jump_time > jump_cooldown:
                if random.random() < jump_probability:
                    bird.velocity_y = self.jump_strength * (1 + (self.current_level - 1) * 0.2)
                    bird.last_jump_time = current_time
            
            max_y = self.height - 100 - bird.size//2
            min_y = bird.size//2
            bird.pos[1] = max(min_y, min(max_y, bird.pos[1]))
            if bird.pos[1] >= max_y - 5:
                bird.velocity_y = self.jump_strength * (1 + (self.current_level - 1) * 0.2)
                bird.last_jump_time = current_time
            
            if current_time - self.last_wing_time > self.wing_delay:
                bird.wing_up = not bird.wing_up
                self.last_wing_time = current_time

    def update_home_screen_birds(self):
        self.home_screen_birds = [bird for bird in self.home_screen_birds if not (
            (bird.direction == 1 and bird.pos[0] > self.width + bird.size) or
            (bird.direction == -1 and bird.pos[0] < -bird.size)
        )]
        
        current_time = time.time()
        if current_time - self.last_bird_spawn_time > 1.0 and len(self.home_screen_birds) < 5:
            self.home_screen_birds.append(self.spawn_bird())
            self.last_bird_spawn_time = current_time
        
        for bird in self.home_screen_birds:
            bird.pos[0] += bird.speed * bird.direction
            bird.velocity_y += self.gravity * 0.5
            bird.pos[1] += bird.velocity_y
            
            if current_time - bird.last_jump_time > self.jump_cooldown:
                if random.random() < 0.03:
                    bird.velocity_y = self.jump_strength * 0.7
                    bird.last_jump_time = current_time
            
            max_y = self.height - 100 - bird.size//2
            min_y = bird.size//2
            bird.pos[1] = max(min_y, min(max_y, bird.pos[1]))
            if bird.pos[1] in (min_y, max_y):
                bird.velocity_y = 0
            
            if current_time - self.last_wing_time > self.wing_delay:
                bird.wing_up = not bird.wing_up
                self.last_wing_time = current_time

    def draw_bird(self, bird):
        x, y = bird.pos
        size = bird.size
        direction = bird.direction
        
        colors = {
            25: (255, 99, 71),
            40: (139, 69, 19),
            55: (47, 79, 79)
        }
        color = colors[bird.size]
        
        pygame.draw.circle(self.screen, color, (int(x), int(y)), size//2)
        head_x = x + (direction * size//2)
        pygame.draw.circle(self.screen, color, (int(head_x), int(y)), size//3)
        eye_x = head_x + (direction * size//6)
        pygame.draw.circle(self.screen, (0, 0, 0), (int(eye_x), int(y - size//6)), max(3, size//10))
        beak_start = (head_x + (direction * size//4), y)
        beak_end = (head_x + (direction * size//2), y)
        pygame.draw.line(self.screen, (255, 140, 0), beak_start, beak_end, max(2, size//10))
        wing_y = y - (size//2 if bird.wing_up else 0)
        wing_points = [
            (x - (size//3), y),
            (x, wing_y),
            (x + (size//3), y)
        ]
        pygame.draw.polygon(self.screen, color, wing_points)

    def update_explosions(self):
        self.explosions = [(pos, frame + 1) for pos, frame in self.explosions 
                          if frame < self.explosion_max_frames]

    def draw_explosions(self):
        for pos, frame in self.explosions:
            radius = int(frame * 2)
            colors = [(255, 165, 0), (255, 69, 0), (255, 0, 0)]
            for i, color in enumerate(colors):
                pygame.draw.circle(self.screen, color, 
                                 (int(pos[0]), int(pos[1])), 
                                 radius - i*5)

    def check_hits(self):
        if not self.is_shooting:
            return
            
        for bird in self.active_birds[:]:
            bird_rect = pygame.Rect(
                bird.pos[0] - bird.size//2,
                bird.pos[1] - bird.size//2,
                bird.size,
                bird.size
            )
            crosshair_rect = pygame.Rect(
                self.crosshair_pos[0] - self.crosshair_size,
                self.crosshair_pos[1] - self.crosshair_size,
                self.crosshair_size * 2,
                self.crosshair_size * 2
            )
            
            if bird_rect.colliderect(crosshair_rect):
                self.score += bird.points
                self.explosions.append((bird.pos.copy(), 0))
                self.active_birds.remove(bird)

    def draw_score_and_timer(self):
        score_box_width = 200
        score_box_height = 50
        score_box_x = self.width - score_box_width - 10
        score_box_y = 10
        
        score_surface = pygame.Surface((score_box_width, score_box_height), pygame.SRCALPHA)
        pygame.draw.rect(score_surface, (0, 0, 0, 128), 
                       (0, 0, score_box_width, score_box_height), 
                       border_radius=10)
        self.screen.blit(score_surface, (score_box_x, score_box_y))
        
        score_text = f"SCORE: {self.score} | LEVEL: {self.current_level}"
        shadow_text = self.main_font.render(score_text, True, (0, 0, 0))
        self.screen.blit(shadow_text, (score_box_x + 12, score_box_y + 13))
        text = self.main_font.render(score_text, True, (255, 255, 0))
        self.screen.blit(text, (score_box_x + 10, score_box_y + 10))
        
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.game_duration - elapsed_time)
        
        if remaining_time == 0 and not self.game_over:
            self.game_over = True
            self.save_score()  # Save score when level ends
            self.show_level_select = True
        
        timer_box_width = 200
        timer_box_height = 50
        timer_box_x = 10
        timer_box_y = 10
        
        timer_surface = pygame.Surface((timer_box_width, timer_box_height), pygame.SRCALPHA)
        pygame.draw.rect(timer_surface, (0, 0, 0, 128), 
                       (0, 0, timer_box_width, timer_box_height), 
                       border_radius=10)
        self.screen.blit(timer_surface, (timer_box_x, timer_box_y))
        
        minutes = int(remaining_time) // 60
        seconds = int(remaining_time) % 60
        time_str = f"TIME: {minutes:02d}:{seconds:02d}"
        
        if remaining_time > 20:
            color = (255, 255, 255)
        elif remaining_time > 10:
            color = (255, 255, 0)
        else:
            if int(remaining_time * 2) % 2 == 0:
                color = (255, 0, 0)
            else:
                color = (255, 255, 255)
        
        shadow_text = self.main_font.render(time_str, True, (0, 0, 0))
        self.screen.blit(shadow_text, (timer_box_x + 12, timer_box_y + 13))
        time_text = self.main_font.render(time_str, True, color)
        self.screen.blit(time_text, (timer_box_x + 10, timer_box_y + 10))

    def draw_level_select(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        completed_text = self.large_font.render(f"LEVEL {self.current_level} COMPLETED", True, (0, 255, 0))
        text_width = completed_text.get_width()
        self.screen.blit(completed_text, (self.width//2 - text_width//2, self.height//2 - 120))
        
        score_text = self.main_font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        score_width = score_text.get_width()
        self.screen.blit(score_text, (self.width//2 - score_width//2, self.height//2 - 40))
        
        hint_text = self.main_font.render("Press ENTER to continue to next level", True, (255, 255, 255))
        hint_width = hint_text.get_width()
        self.screen.blit(hint_text, (self.width//2 - hint_width//2, self.height//2 - 10))
        
        self.next_level_btn_rect = pygame.Rect(self.width//2 - 150, self.height//2 + 40, 140, 60)
        pygame.draw.rect(self.screen, self.btn_color, self.next_level_btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.next_level_btn_rect, width=2, border_radius=10)
        next_text = self.retro_font.render(f"LEVEL {self.current_level + 1}", True, (0, 0, 0))
        next_width = next_text.get_width()
        self.screen.blit(next_text, 
                        (self.width//2 - 150 + 70 - next_width//2, 
                         self.height//2 + 40 + 30 - next_text.get_height()//2))
        
        self.quit_btn_rect = pygame.Rect(self.width//2 + 10, self.height//2 + 40, 140, 60)
        pygame.draw.rect(self.screen, (255, 0, 0), self.quit_btn_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), self.quit_btn_rect, width=2, border_radius=10)
        quit_text = self.retro_font.render("QUIT", True, (0, 0, 0))
        quit_width = quit_text.get_width()
        self.screen.blit(quit_text,
                        (self.width//2 + 10 + 70 - quit_width//2,
                         self.height//2 + 40 + 30 - quit_text.get_height()//2))
        
        # Draw leaderboard
        leaderboard = self.get_leaderboard()
        lb_font = pygame.font.Font(None, 30)
        lb_title = lb_font.render("Leaderboard", True, (255, 255, 0))
        self.screen.blit(lb_title, (self.width - 220, 20))
        for i, (name, score) in enumerate(leaderboard):
            lb_text = lb_font.render(f"{i+1}. {name}: {score}", True, (255, 255, 255))
            self.screen.blit(lb_text, (self.width - 220, 50 + i * 30))

    def start_new_level(self):
        self.game_over = False
        self.show_level_select = False
        self.start_time = time.time()
        self.active_birds = []
        self.explosions = []
        self.max_birds = min(5, 3 + self.current_level - 1)
        
        print(f"Starting Level {self.current_level}:")
        print(f"  Small bird speed: {self.get_level_speed(self.bird_types['small']['base_speed']):.1f} (Base: {self.bird_types['small']['base_speed']})")
        print(f"  Medium bird speed: {self.get_level_speed(self.bird_types['medium']['base_speed']):.1f} (Base: {self.bird_types['medium']['base_speed']})")
        print(f"  Large bird speed: {self.get_level_speed(self.bird_types['large']['base_speed']):.1f} (Base: {self.bird_types['large']['base_speed']})")
        print(f"  Max birds: {self.max_birds}")

    def draw_home_screen(self):
        self.screen.blit(self.home_screen_background, (0, 0))
        
        for bird in self.home_screen_birds:
            self.draw_bird(bird)
            
        title_font = pygame.font.Font(None, 100)
        title_text = "BIRD SHOOTER"
        
        glow_colors = [
            (200, 100, 0),
            (225, 125, 25),
            (255, 165, 0),
            (255, 215, 0)
        ]
        
        pulse = (np.sin(time.time() * 3) + 1) / 2
        glow_size = int(10 + 5 * pulse)
        
        for i, color in enumerate(glow_colors):
            size_factor = glow_size - i * 2
            if size_factor <= 0:
                continue
                
            for offset_x in range(-size_factor, size_factor + 1, 2):
                for offset_y in range(-size_factor, size_factor + 1, 2):
                    if offset_x**2 + offset_y**2 <= size_factor**2:
                        text_surface = title_font.render(title_text, True, color)
                        text_width = text_surface.get_width()
                        self.screen.blit(text_surface, 
                                       (self.width//2 - text_width//2 + offset_x, 
                                        100 + offset_y))
        
        main_text_surface = title_font.render(title_text, True, (255, 255, 255))
        main_text_width = main_text_surface.get_width()
        self.screen.blit(main_text_surface, (self.width//2 - main_text_width//2, 100))
        
        if self.input_active:
            name_prompt = self.main_font.render("Enter Your Name:", True, (255, 255, 255))
            name_text = self.main_font.render(self.player_name + ("_" if int(time.time() * 2) % 2 == 0 else ""), True, (255, 255, 255))
            self.screen.blit(name_prompt, (self.width//2 - name_prompt.get_width()//2, self.height//2 - 50))
            self.screen.blit(name_text, (self.width//2 - name_text.get_width()//2, self.height//2))
        else:
            btn_rect = pygame.Rect(self.width//2 - 100, self.height//2 + 50, 200, 70)
            btn_color = self.btn_hover_color if self.btn_hover else self.btn_color
            
            shadow_rect = btn_rect.copy()
            shadow_rect.x += 5
            shadow_rect.y += 5
            pygame.draw.rect(self.screen, (0, 0, 0, 128), shadow_rect, border_radius=10)
            
            pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=10)
            pygame.draw.rect(self.screen, (255, 255, 255), btn_rect, width=3, border_radius=10)
            
            btn_text = self.retro_font.render("PLAY NOW", True, (0, 0, 0))
            btn_text_width = btn_text.get_width()
            self.screen.blit(btn_text, 
                           (self.width//2 - btn_text_width//2, 
                            self.height//2 + 50 + 70//2 - btn_text.get_height()//2))
        
        instr_font = pygame.font.Font(None, 24)
        instr_text = "Use your hand to aim. Pinch index finger and thumb to shoot!" if not self.input_active else "Press ENTER to confirm name"
        instr_surface = instr_font.render(instr_text, True, (255, 255, 255))
        instr_width = instr_surface.get_width()
        self.screen.blit(instr_surface, 
                       (self.width//2 - instr_width//2, 
                        self.height - 50))
        
        keyboard_text = instr_font.render("Press ENTER to start or ESC to quit" if not self.input_active else "", True, (255, 255, 255))
        keyboard_width = keyboard_text.get_width()
        self.screen.blit(keyboard_text, 
                       (self.width//2 - keyboard_width//2, 
                        self.height - 25))
        
        hand_icon_size = 100
        hand_center_x = self.width//2
        hand_center_y = self.height//2 - 50 if not self.input_active else self.height//2 + 100
        
        pygame.draw.circle(self.screen, (255, 219, 172), 
                         (hand_center_x, hand_center_y),
                         hand_icon_size//3)
        
        for angle in [20, 35, 50, 65, 80]:
            rad_angle = np.radians(angle)
            finger_length = hand_icon_size//2
            end_x = hand_center_x + int(np.cos(rad_angle) * finger_length)
            end_y = hand_center_y - int(np.sin(rad_angle) * finger_length)
            pygame.draw.line(self.screen, (255, 219, 172),
                           (hand_center_x, hand_center_y),
                           (end_x, end_y),
                           hand_icon_size//8)
            pygame.draw.circle(self.screen, (255, 219, 172),
                             (end_x, end_y),
                             hand_icon_size//16)

        if results := self.hands.process(cv2.cvtColor(cv2.flip(self.cap.read()[1], 1), cv2.COLOR_BGR2RGB)):
            if results.multi_hand_landmarks:
                pygame.draw.circle(self.screen, (255, 0, 0), 
                                (int(self.crosshair_pos[0]), int(self.crosshair_pos[1])), 
                                self.crosshair_size//2, 2)
                pygame.draw.line(self.screen, (255, 0, 0),
                              (self.crosshair_pos[0] - self.crosshair_size, self.crosshair_pos[1]),
                              (self.crosshair_pos[0] + self.crosshair_size, self.crosshair_pos[1]))
                pygame.draw.line(self.screen, (255, 0, 0),
                              (self.crosshair_pos[0], self.crosshair_pos[1] - self.crosshair_size),
                              (self.crosshair_pos[0], self.crosshair_pos[1] + self.crosshair_size))
        
        # Draw leaderboard
        leaderboard = self.get_leaderboard()
        lb_font = pygame.font.Font(None, 30)
        lb_title = lb_font.render("Leaderboard", True, (255, 255, 0))
        self.screen.blit(lb_title, (self.width - 220, 20))
        for i, (name, score) in enumerate(leaderboard):
            lb_text = lb_font.render(f"{i+1}. {name}: {score}", True, (255, 255, 255))
            self.screen.blit(lb_text, (self.width - 220, 50 + i * 30))

    def run(self):
        clock = pygame.time.Clock()
        self.background = self.create_background(self.current_time_of_day)
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.show_level_select or self.game_over:
                            self.running = False
                        elif self.show_home_screen:
                            self.running = False
                        else:
                            self.show_home_screen = True
                            self.game_over = False
                            self.score = 0
                            self.current_level = 1
                            self.active_birds = []
                            self.explosions = []
                    elif event.key == pygame.K_RETURN:
                        if self.show_home_screen:
                            if self.input_active and self.player_name:
                                self.input_active = False
                            elif not self.input_active:
                                self.show_home_screen = False
                                self.start_time = time.time()
                        elif self.show_level_select:
                            self.current_level += 1
                            self.start_new_level()
                    elif self.input_active and self.show_home_screen:
                        if event.key == pygame.K_BACKSPACE:
                            self.player_name = self.player_name[:-1]
                        elif event.key in range(pygame.K_a, pygame.K_z + 1) or event.key in range(pygame.K_0, pygame.K_9 + 1):
                            self.player_name += event.unicode.upper()
                            if len(self.player_name) > 10:  # Limit name length
                                self.player_name = self.player_name[:10]
            
            self.process_hands()
            
            if self.show_home_screen:
                self.update_home_screen_birds()
                self.draw_home_screen()
            else:
                self.screen.blit(self.background, (0, 0))
                
                if not self.game_over:
                    self.update_background()
                    self.update_birds()
                    self.update_explosions()
                    self.check_hits()
                    
                    for bird in self.active_birds:
                        self.draw_bird(bird)
                    
                    self.draw_explosions()
                    
                    pygame.draw.circle(self.screen, (255, 0, 0), 
                                    (int(self.crosshair_pos[0]), int(self.crosshair_pos[1])), 
                                    self.crosshair_size//2, 2)
                    pygame.draw.line(self.screen, (255, 0, 0),
                                  (self.crosshair_pos[0] - self.crosshair_size, self.crosshair_pos[1]),
                                  (self.crosshair_pos[0] + self.crosshair_size, self.crosshair_pos[1]))
                    pygame.draw.line(self.screen, (255, 0, 0),
                                  (self.crosshair_pos[0], self.crosshair_pos[1] - self.crosshair_size),
                                  (self.crosshair_pos[0], self.crosshair_pos[1] + self.crosshair_size))
                    
                    self.draw_score_and_timer()
                elif self.show_level_select:
                    self.draw_level_select()
            
            pygame.display.flip()
            clock.tick(60)
        
        self.conn.close()
        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    game = BirdShooterGame()
    game.run()
