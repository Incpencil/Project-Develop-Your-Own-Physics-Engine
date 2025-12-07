import pygame
import sys
import math
import numpy as np

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Sight Checking")
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255,255,255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

# 장애물 추가 - 직사각형과 원 분리
rect_obstacles = [
    pygame.Rect(200, 200, 120, 80)   # 직사각형 장애물
]

circle_obstacles = [
    {'center': [500, 300], 'radius': 60}  # 원형 장애물
]

targetColor = WHITE
playerPosition = [300,300]
targetPosition = [130, 100]

move_speed = 5  # 이동 속도
theta = 0
playerViewVector = [0, -100]
norm_playerViewVector = [1,1]
dir1 = [-100, -100*math.sqrt(3)]
dir2 = [100, -100*math.sqrt(3)]
playerToTarget = [1,1] #v
norm_playerToTarget = [1,1]


def calculate_playerToTarget(playerToTarget,norm_playerToTarget):
    ptt = [targetPosition[0]-playerPosition[0],targetPosition[1]-playerPosition[1]]
    norm_ptt = np.array(playerToTarget/np.linalg.norm(playerToTarget)).tolist()
    
    return ptt, norm_ptt

# ========== new =============
def angle_between_vectors(norm_v, norm_f):

    dot_product = np.dot(norm_v, norm_f)  # 두 벡터의 내적
    angle = np.arccos(dot_product)  # 라디안으로 각도 계산
    angle = angle * 180 / math.pi
    return angle

def is_line_blocked(start, end, rect_obstacles, circle_obstacles):
    """start에서 end까지 직선이 장애물에 막히는지 확인"""
    # 직사각형 장애물 체크
    for obstacle in rect_obstacles:
        if obstacle.clipline(start, end):
            return True
    
    # 원형 장애물 체크
    for circle in circle_obstacles:
        if is_line_blocked_by_circle(start, end, circle):
            return True
    
    return False

def is_line_blocked_by_circle(start, end, circle):
    """직선이 원에 막히는지 확인 (점-선분 거리 계산)"""
    cx, cy = circle['center']
    r = circle['radius']
    x1, y1 = start
    x2, y2 = end
    
    # 선분의 방향 벡터
    dx = x2 - x1
    dy = y2 - y1
    
    # 선분의 길이의 제곱
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        # 시작점과 끝점이 같은 경우
        dist = math.sqrt((x1-cx)**2 + (y1-cy)**2)
        return dist < r
    
    # 원의 중심에서 선분까지의 최단거리 계산
    t = max(0, min(1, ((cx-x1)*dx + (cy-y1)*dy) / length_sq))
    
    # 선분 위의 가장 가까운 점
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # 거리 계산
    dist = math.sqrt((closest_x-cx)**2 + (closest_y-cy)**2)
    
    return dist < r

def can_move(new_position, rect_obstacles, circle_obstacles):
    """캐릭터가 새 위치로 이동 가능한지 확인 (충돌 체크)"""
    player_radius = 15
    px, py = new_position
    
    # 직사각형 충돌 체크
    player_rect = pygame.Rect(px - player_radius, py - player_radius, 
                               player_radius * 2, player_radius * 2)
    for obstacle in rect_obstacles:
        if player_rect.colliderect(obstacle):
            return False
    
    # 원 충돌 체크
    for circle in circle_obstacles:
        cx, cy = circle['center']
        distance = math.sqrt((px - cx)**2 + (py - cy)**2)
        if distance < (player_radius + circle['radius']):
            return False
    
    return True
# ========== new =============

def draw_circles():
    pygame.draw.circle(screen, RED, playerPosition, 15)  # player
    pygame.draw.circle(screen, targetColor, targetPosition, 5)  # target

def draw_obstacles():
    """장애물 그리기 - 직사각형과 원"""
    # 직사각형 그리기
    for obstacle in rect_obstacles:
        pygame.draw.rect(screen, GRAY, obstacle)
    
    # 원 그리기
    for circle in circle_obstacles:
        pygame.draw.circle(screen, GRAY, circle['center'], circle['radius'])

def draw_text():
    font = pygame.font.Font(None, 36)
    text_player = font.render(f"Player Pos: {playerPosition}", True, WHITE)
    text_target = font.render(f"Target Pos: {targetPosition}", True, WHITE)
    screen.blit(text_player, (screen.get_width() - text_player.get_width() - 10,10))
    screen.blit(text_target, (screen.get_width() - text_target.get_width() - 10,50))

    text_v = font.render(f"v: {norm_playerToTarget[0]:.3f},{norm_playerToTarget[1]:.3f}", True, BLUE)
    screen.blit(text_v, (screen.get_width() - text_v.get_width() - 10,90))
    # ========== new =============
    text_alpha = font.render(f"alpha: {alpha:.3f}", True, BLUE)
    screen.blit(text_alpha, (screen.get_width() - text_alpha.get_width() - 10,130))
    # ========== new =============
    
def rotation(x,y,theta):
    x_ = x*math.cos(theta) - y*math.sin(theta)
    y_ = x*math.sin(theta) + y*math.cos(theta)
    return x_,y_

def draw_line():
    # 중앙선 제거, 시야각 경계선만 표시
    pygame.draw.line(screen, WHITE, playerPosition, (np.array(playerPosition) + np.array(dir1)).tolist(),2)
    pygame.draw.line(screen, WHITE, playerPosition, (np.array(playerPosition) + np.array(dir2)).tolist(),2)

def draw_player_to_target_line():
    """플레이어와 타겟을 연결하는 선 그리기"""
    if show_line:
        # 타겟 색상에 따라 선 색상 결정
        if targetColor == BLUE:
            line_color = BLUE  # 볼 수 있음
        elif targetColor == RED:
            line_color = RED   # 시야각 내지만 막힘
        else:
            line_color = WHITE # 시야각 밖
        
        pygame.draw.line(screen, line_color, playerPosition, targetPosition, 2)

# ========== new =============
alpha = 0
# ========== new =============
done = False
left_p = False
right_p = False
up_p = False
down_p = False
space_p = False
a_p = False  # 왼쪽 회전
d_p = False  # 오른쪽 회전
show_line = True  # 플레이어-타겟 연결선 표시 여부

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True       
        if event.type == pygame.KEYDOWN:
            if event.key==pygame.K_SPACE:
                space_p = True
            if event.key == pygame.K_LEFT:
                left_p = True
            if event.key == pygame.K_RIGHT:
                right_p = True
            if event.key == pygame.K_UP:
                up_p = True
            if event.key == pygame.K_DOWN:
                down_p = True
            if event.key == pygame.K_a:  # A 키 추가
                a_p = True
            if event.key == pygame.K_d:  # D 키 추가
                d_p = True
            if event.key == pygame.K_t:  # T 키로 선 토글
                show_line = not show_line
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                space_p = False
            if event.key == pygame.K_LEFT:
                left_p = False
            if event.key == pygame.K_RIGHT:
                right_p = False
            if event.key == pygame.K_UP:
                up_p = False
            if event.key == pygame.K_DOWN:
                down_p = False
            if event.key == pygame.K_a:  # A 키 추가
                a_p = False
            if event.key == pygame.K_d:  # D 키 추가
                d_p = False
    
    if space_p:
        theta = 0.05
        playerViewVector = rotation(playerViewVector[0],playerViewVector[1],theta)
        dir1 = rotation(dir1[0],dir1[1],theta)
        dir2 = rotation(dir2[0],dir2[1],theta)
    
    # A 키: 왼쪽(반시계 방향) 회전
    if a_p:
        theta = 0.05
        playerViewVector = rotation(playerViewVector[0],playerViewVector[1],theta)
        dir1 = rotation(dir1[0],dir1[1],theta)
        dir2 = rotation(dir2[0],dir2[1],theta)
    
    # D 키: 오른쪽(시계 방향) 회전
    if d_p:
        theta = -0.05
        playerViewVector = rotation(playerViewVector[0],playerViewVector[1],theta)
        dir1 = rotation(dir1[0],dir1[1],theta)
        dir2 = rotation(dir2[0],dir2[1],theta)
    
    # 이동 시 충돌 체크
    if left_p:
        new_pos = [playerPosition[0] - move_speed, playerPosition[1]]
        if can_move(new_pos, rect_obstacles, circle_obstacles):
            playerPosition[0] -= move_speed
    if right_p:
        new_pos = [playerPosition[0] + move_speed, playerPosition[1]]
        if can_move(new_pos, rect_obstacles, circle_obstacles):
            playerPosition[0] += move_speed
    if up_p:
        new_pos = [playerPosition[0], playerPosition[1] - move_speed]
        if can_move(new_pos, rect_obstacles, circle_obstacles):
            playerPosition[1] -= move_speed
    if down_p:
        new_pos = [playerPosition[0], playerPosition[1] + move_speed]
        if can_move(new_pos, rect_obstacles, circle_obstacles):
            playerPosition[1] += move_speed
        
    # update
    playerToTarget,norm_playerToTarget = calculate_playerToTarget(playerToTarget,norm_playerToTarget)
    
    # ========== new =============

    norm_playerViewVector = np.array(playerViewVector/np.linalg.norm(playerViewVector))
    alpha = angle_between_vectors(norm_playerToTarget, norm_playerViewVector)
    
    # 시야각 내에 있는지 확인
    beta = 30
    if alpha < beta:
        # 시야각 내에 있으면 장애물 체크
        if not is_line_blocked(playerPosition, targetPosition, rect_obstacles, circle_obstacles):
            targetColor = BLUE  # 볼 수 있음
        else:
            targetColor = RED   # 시야각 내지만 장애물에 막힘
    else:
        targetColor = WHITE  # 시야각 밖
    # ========== new =============
    
    screen.fill(BLACK)
    draw_obstacles()  # 장애물 먼저 그리기
    draw_player_to_target_line()  # 플레이어-타겟 연결선
    draw_circles()
    draw_text()
    draw_line()
    pygame.display.update()
    clock.tick(30)


pygame.quit()
