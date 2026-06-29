import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from config import CELL_SIZE, FPS
from env.guard import GuardState

# Colors
C_BG        = (15,  17,  26)
C_EMPTY     = (30,  33,  48)
C_WALL      = (70,  75, 100)
C_AGENT     = (80, 220, 160)
C_EXIT      = (255, 200,  50)
C_GUARD     = (220,  80,  80)
C_GUARD_ALT = (255, 150,  50)
C_FOV       = (220,  80,  80,  40)
C_GRID      = (40,  44,  60)
C_TEXT      = (200, 210, 230)


class Renderer:
    def __init__(self, grid_size):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")

        pygame.init()
        self.grid_size   = grid_size
        self.cell        = CELL_SIZE
        self.width       = grid_size * CELL_SIZE
        self.height      = grid_size * CELL_SIZE + 40
        self.screen      = pygame.display.set_mode((self.width, self.height))
        self.clock       = pygame.time.Clock()
        self.fov_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.display.set_caption("Stealth Escape — RL Agent")
        self.font = pygame.font.SysFont("monospace", 14)

    def draw(self, grid, agent_pos, exit_pos, guards, step):
        """Render one frame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        self.screen.fill(C_BG)
        self.fov_surface.fill((0, 0, 0, 0))

        n  = self.grid_size
        cs = self.cell

        # Draw grid cells
        for r in range(n):
            for c in range(n):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                val  = grid[r][c]
                if val == 1:
                    pygame.draw.rect(self.screen, C_WALL, rect)
                else:
                    pygame.draw.rect(self.screen, C_EMPTY, rect)
                pygame.draw.rect(self.screen, C_GRID, rect, 1)

        # Draw FOV cones for guards
        for guard in guards:
            self._draw_fov(guard, grid)
        self.screen.blit(self.fov_surface, (0, 0))

        # Draw exit
        er, ec = exit_pos
        exit_rect = pygame.Rect(ec * cs + 4, er * cs + 4, cs - 8, cs - 8)
        pygame.draw.rect(self.screen, C_EXIT, exit_rect, border_radius=4)
        label = self.font.render("EXIT", True, C_BG)
        self.screen.blit(label, (ec * cs + 4, er * cs + cs // 3))

        # Draw guards
        for guard in guards:
            gr, gc = guard.pos
            color  = C_GUARD_ALT if guard.state == GuardState.ALERT else C_GUARD
            cx     = gc * cs + cs // 2
            cy     = gr * cs + cs // 2
            pygame.draw.circle(self.screen, color, (cx, cy), cs // 3)
            import math
            rad = math.radians(guard.facing)
            ex  = int(cx + (cs // 2.5) * math.cos(rad))
            ey  = int(cy + (cs // 2.5) * math.sin(rad))
            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (ex, ey), 2)

        # Draw agent
        ar, ac = agent_pos
        acx = ac * cs + cs // 2
        acy = ar * cs + cs // 2
        pygame.draw.circle(self.screen, C_AGENT, (acx, acy), cs // 3)

        # Info bar
        info_y = n * cs + 8
        step_label = self.font.render(f"Step: {step}", True, C_TEXT)
        self.screen.blit(step_label, (8, info_y))

        alert_guards = sum(1 for g in guards if g.state == GuardState.ALERT)
        if alert_guards:
            alert_label = self.font.render(f"! {alert_guards} guard(s) ALERT", True, C_GUARD_ALT)
            self.screen.blit(alert_label, (120, info_y))

        pygame.display.flip()
        self.clock.tick(FPS)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def _draw_fov(self, guard, grid):
        """Draw semi-transparent FOV cone for a guard."""
        import math
        from config import GUARD_FOV_ANGLE, GUARD_FOV_RANGE
        from env.fov import has_line_of_sight

        cs     = self.cell
        gr, gc = guard.pos
        cx     = gc * cs + cs // 2
        cy     = gr * cs + cs // 2

        half  = GUARD_FOV_ANGLE / 2
        color = (255, 100, 50, 40) if guard.state == GuardState.ALERT else (220, 80, 80, 35)

        points = [(cx, cy)]
        steps  = 20
        for i in range(steps + 1):
            angle = guard.facing - half + (GUARD_FOV_ANGLE * i / steps)
            rad   = math.radians(angle)
            for dist in range(1, GUARD_FOV_RANGE + 1):
                tr = int(round(guard.pos[0] + dist * math.sin(rad)))
                tc = int(round(guard.pos[1] + dist * math.cos(rad)))
                n  = self.grid_size
                if tr < 0 or tr >= n or tc < 0 or tc >= n or grid[tr][tc] == 1:
                    tr = int(guard.pos[0] + (dist - 1) * math.sin(rad))
                    tc = int(guard.pos[1] + (dist - 1) * math.cos(rad))
                    break
            px = tc * cs + cs // 2
            py = tr * cs + cs // 2
            points.append((px, py))

        if len(points) >= 3:
            pygame.draw.polygon(self.fov_surface, color, points)

    def close(self):
        pygame.quit()