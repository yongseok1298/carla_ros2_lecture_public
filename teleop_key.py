#!/usr/bin/env python3
import time, math, pygame, carla

HOST, PORT = "127.0.0.1", 2000
# speed limit < 20km/h
SPEED_LIMIT_KMH = 20.0
SPEED_LIMIT_MS  = SPEED_LIMIT_KMH / 3.6
GOV_MARGIN_MS   = 1.2   
GOV_TH_MIN      = 0.05  
GOV_TH_CUT      = 0.35  
GOV_KP_BRAKE    = 0.18  
GOV_BIAS_BRAKE  = 0.04  

# control tuning
THR_SLEW_UP     = 1.2  
THR_SLEW_DOWN   = 1.5  
BRK_SLEW_UP     = 3.5   
BRK_SLEW_DOWN   = 3.0  

STEER_RATE      = 2.8  
STEER_RETURN    = 4.0  

SOFTSTART_THR   = 0.28 
SOFTSTART_V_MAX = 4.0 

NONLIN_WEIGHT   = 0.55  


def find_vehicle(world, timeout=10.0, interval=0.2):
    deadline = time.time() + timeout
    last_cnt = -1
    while time.time() < deadline:
        try:
            world.wait_for_tick(timeout=1.0)
        except Exception:
            pass
        actors = world.get_actors().filter("vehicle.*")
        if last_cnt != len(actors):

            print(f"[carla] vehicles detected: {len(actors)}")
            last_cnt = len(actors)
        for v in actors:
            if v.attributes.get("role_name", "") == "hero":
                return v
        time.sleep(interval)
    return None

def main():
    client = carla.Client(HOST, PORT); client.set_timeout(5.0)
    world  = client.get_world()

    veh = find_vehicle(world, timeout=15.0)
    if not veh:
        print("No vehicle found. Make sure ros2_native.py spawned the hero.")
        return

    veh.set_autopilot(False)

    ctrl = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0,
                                hand_brake=False, reverse=False)
    pygame.init()
    pygame.display.set_caption("CARLA Teleop (WASD)")
    screen = pygame.display.set_mode((400, 120))
    clock = pygame.time.Clock()
    
    th_cmd = 0.0
    br_cmd = 0.0
    st_cmd = 0.0

    def _speed_ms(v):  # m/s
        return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

    def _slew(curr, target, up, down, dt):
        rate = up if target > curr else down
        step = rate * dt
        if abs(target - curr) <= step:
            return target
        return curr + step * (1 if target > curr else -1)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: 
                        return
                    if event.key == pygame.K_q: ctrl.reverse = True
                    if event.key == pygame.K_e: ctrl.reverse = False
                    if event.key == pygame.K_r: st_cmd = 0.0

            keys = pygame.key.get_pressed()

            dt = max(1e-3, clock.tick(60) / 1000.0)

            accel_pressed  = keys[pygame.K_w] or keys[pygame.K_UP]
            brake_pressed  = keys[pygame.K_s] or keys[pygame.K_DOWN]
            left_pressed   = keys[pygame.K_a] or keys[pygame.K_LEFT]
            right_pressed  = keys[pygame.K_d] or keys[pygame.K_RIGHT]
            ctrl.hand_brake = keys[pygame.K_SPACE]

            try:
                v_ms = _speed_ms(veh.get_velocity())
            except Exception:
                v_ms = 0.0

            want_thr = 0.0
            if accel_pressed and not brake_pressed:
                want_thr = 1.0
                if v_ms < SOFTSTART_V_MAX:
                    k = 1.0 - (v_ms / SOFTSTART_V_MAX)
                    want_thr = max(want_thr * (1.0 - 0.25*k), SOFTSTART_THR)

            want_brk = 1.0 if brake_pressed else 0.0

            th_cmd = _slew(th_cmd, want_thr, THR_SLEW_UP, THR_SLEW_DOWN, dt)
            br_cmd = _slew(br_cmd, want_brk, BRK_SLEW_UP, BRK_SLEW_DOWN, dt)

            if br_cmd > 0.05:
                th_cmd = max(0.0, th_cmd - 3.0*dt)

            if left_pressed and not right_pressed:
                target_st = -1.0
            elif right_pressed and not left_pressed:
                target_st = 1.0
            else:
                if abs(st_cmd) < STEER_RETURN*dt:
                    target_st = 0.0
                else:
                    target_st = st_cmd - STEER_RETURN*dt * (1 if st_cmd > 0 else -1)

            st_cmd = _slew(st_cmd, target_st, STEER_RATE, STEER_RATE, dt)

            if v_ms < 15.0:
                st_limit = 1.0 - 0.65*(v_ms/15.0)  
            elif v_ms < 25.0:
                st_limit = 0.35 - 0.10*((v_ms-15.0)/10.0) 
            else:
                st_limit = 0.25
            st_cmd = max(-st_limit, min(st_limit, st_cmd))

            steer_out = (1.0 - NONLIN_WEIGHT)*st_cmd + NONLIN_WEIGHT*(st_cmd**3)

            if v_ms >= (SPEED_LIMIT_MS - GOV_MARGIN_MS):
                approach = min(1.0, max(0.0, (v_ms - (SPEED_LIMIT_MS - GOV_MARGIN_MS)) / GOV_MARGIN_MS))
                th_cmd = max(GOV_TH_MIN, th_cmd * (GOV_TH_CUT + (1.0 - GOV_TH_CUT) * (1.0 - approach)))

            if v_ms > SPEED_LIMIT_MS:
                overshoot = v_ms - SPEED_LIMIT_MS
                th_cmd = 0.0
                br_cmd = max(br_cmd, min(1.0, GOV_KP_BRAKE * overshoot + GOV_BIAS_BRAKE))

            ctrl.throttle = max(0.0, min(1.0, th_cmd))
            ctrl.brake    = max(0.0, min(1.0, br_cmd))
            ctrl.steer    = max(-1.0, min(1.0, steer_out))
            veh.apply_control(ctrl)

            screen.fill((20,20,20))
            txt = f"v:{v_ms:4.1f} T:{ctrl.throttle:.2f} B:{ctrl.brake:.2f} S:{ctrl.steer:.2f} Rev:{int(ctrl.reverse)} HB:{int(ctrl.hand_brake)}"
            pygame.display.set_caption(txt)
            pygame.display.flip()

            clock.tick(30)
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
