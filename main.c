
#include <SDL2/SDL.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define GRID_HEIGHT 50
#define GRID_WIDTH 50
#define DX 0.1f
#define DY 0.1f
#define DT 0.01f
#define VISCOSITY 0.001f
#define GRAVITY -9.8f

float p[GRID_HEIGHT][GRID_WIDTH];          // Pressure at cell centers
float u[GRID_HEIGHT][GRID_WIDTH + 1];      // X-velocity at vertical edges
float v[GRID_HEIGHT + 1][GRID_WIDTH];      // Y-velocity at horizontal faces
float u_prev[GRID_HEIGHT][GRID_WIDTH + 1]; // Previous u for diffusion
float v_prev[GRID_HEIGHT + 1][GRID_WIDTH]; // Previous v for diffusion
float divergence[GRID_HEIGHT][GRID_WIDTH]; // Divergence field

// ===================================
// SDL Window Configuration
// ===================================

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 800
#define CELL_SIZE (SCREEN_WIDTH / GRID_WIDTH)

SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;

void visualize() {
  // Clear screen
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
  SDL_RenderClear(renderer);

  // Find max pressure for normalization
  float max_pressure = 0.001f;
  for (int i = 0; i < GRID_HEIGHT; i++) {
    for (int j = 0; j < GRID_WIDTH; j++) {
      if (fabs(p[i][j]) > max_pressure)
        max_pressure = fabs(p[i][j]);
    }
  }

  // Draw pressure field
  for (int i = 0; i < GRID_HEIGHT; i++) {
    for (int j = 0; j < GRID_WIDTH; j++) {
      // Normalize pressure to [0,1]
      float normalized_p = fabs(p[i][j]) / max_pressure;

      // Color mapping: blue for positive, red for negative pressure
      if (p[i][j] > 0) {
        SDL_SetRenderDrawColor(renderer, 0, 0, (int)(255 * normalized_p), 255);
      } else {
        SDL_SetRenderDrawColor(renderer, (int)(255 * normalized_p), 0, 0, 255);
      }

      // Draw cell
      SDL_Rect rect = {j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE};
      SDL_RenderFillRect(renderer, &rect);

      // Draw velocity vectors (white)
      float u_avg = 0.5f * (u[i][j] + u[i][j + 1]);
      float v_avg = 0.5f * (v[i][j] + v[i + 1][j]);
      int x1 = j * CELL_SIZE + CELL_SIZE / 2;
      int y1 = i * CELL_SIZE + CELL_SIZE / 2;
      int x2 = x1 + (int)(u_avg * 20); // Scale factor for visibility
      int y2 = y1 + (int)(v_avg * 20);

      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
      SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
    }
  }

  SDL_RenderPresent(renderer);
}

// ===================================
//       HELPER FUNCTIONS
// ===================================

/*
  LERP by beloved <3
  computes a value between a and b based on weight t
*/
float lerp(float a, float b, float t) {
  //
  return a + t * (b - a);
}

float sampleU(float x, float y) {
  x = fmax(0, fmin(GRID_WIDTH, x));
  y = fmax(0, fmin(GRID_HEIGHT - 1, y));

  int i = (int)y;
  int j = (int)x;
  float s = x - j;
  float t = y - i;

  // lerp of both edges
  return lerp(lerp(u[i][j], u[i][j + 1], s),
              lerp(u[i + 1][j], u[i + 1][j + 1], s), t);
}

float sampleV(float x, float y) {
  x = fmax(0, fmin(GRID_WIDTH - 1, x));
  y = fmax(0, fmin(GRID_HEIGHT, y));

  int i = (int)y;
  int j = (int)x;
  float s = x - j;
  float t = y - i;

  // lerp of both edges
  return lerp(lerp(v[i][j], v[i][j + 1], s),
              lerp(v[i + 1][j], v[i + 1][j + 1], s), t);
}

// ===================================
//       UPDATE FUNCTIONS
// ===================================

void apply_Gravity(float dt) {
  for (int i = 0; i <= GRID_HEIGHT; i++) {
    for (int j = 0; j < GRID_WIDTH; j++) {
      v[i][j] += GRAVITY * dt;
    }
  }
}
/*
  Advect velocities U and V
  we want to find out:
  "Where did the current fluid at this edge[i][j] come from
  in the directly previous time step"

  1. Compute backtraces to find out where fluids came from
      x = j - (prev * dt / DX)
      distance traveled horizontally in time, converted to grid coordinate
  2. Interpolate using LERP (sampleU and sampleV), velocity is a weighted value
    between past location and new
*/
void advect_Velocity(float dt) {
  memcpy(u_prev, u, sizeof(u));
  memcpy(v_prev, v, sizeof(v));

  for (int i = 0; i < GRID_HEIGHT; i++) {
    for (int j = 0; j <= GRID_WIDTH; j++) {
      float x = j - (u_prev[i][j] * dt / DX);
      float y = (i + 0.5f) - 0.5f * (v_prev[i][j] + v_prev[i + 1][j]) * dt / DY;
      u[i][j] = sampleU(x, y);
    }
  }

  for (int i = 0; i <= GRID_HEIGHT; i++) {
    for (int j = 0; j < GRID_WIDTH; j++) {
      float x = (j + 0.5f) - 0.5f * (u_prev[i][j] + u_prev[i][j + 1]) * dt / DX;
      float y = i - (v_prev[i][j] * dt / DY);
      v[i][j] = sampleV(x, y);
    }
  }
}

/*
  Project pressure, fluid is treated as incompressible
  ▽ . u = 0
  MODIFIED for numerical stability

  1. Calculate Divergence:
    sum of inflows to cell (inflow for x is the difference of both x edges for
    example, same for y)
  2. Solve Poisson equation using Jacobi-esque
    -> +ve divergence means must flow out
    -> -ve divergence means must flow in

*/
void project_Pressure(float dt) {
  // Compute divergence
  float total_divergence = 0.0f;
  for (int i = 0; i < GRID_HEIGHT; i++) {
    for (int j = 0; j < GRID_WIDTH; j++) {
      divergence[i][j] =
          (u[i][j + 1] - u[i][j]) / DX + (v[i + 1][j] - v[i][j]) / DY;
      total_divergence += fabs(divergence[i][j]);
    }
  }

  // Normalize divergence to prevent explosion
  float divergence_scale =
      total_divergence > 0 ? 1.0f / (total_divergence + 1e-6f) : 0;

  // Solve pressure with improved boundary handling
  memset(p, 0, sizeof(p));
  for (int iter = 0; iter < 100; iter++) {
    for (int i = 1; i < GRID_HEIGHT - 1; i++) {
      for (int j = 1; j < GRID_WIDTH - 1; j++) {
        p[i][j] = (divergence[i][j] * divergence_scale + p[i - 1][j] +
                   p[i + 1][j] + p[i][j - 1] + p[i][j + 1]) *
                  0.25f;
      }
    }

    // Neumann boundary conditions (pressure gradient = 0 at boundaries)
    for (int i = 0; i < GRID_HEIGHT; i++) {
      p[i][0] = p[i][1];
      p[i][GRID_WIDTH - 1] = p[i][GRID_WIDTH - 2];
    }
    for (int j = 0; j < GRID_WIDTH; j++) {
      p[0][j] = p[1][j];
      p[GRID_HEIGHT - 1][j] = p[GRID_HEIGHT - 2][j];
    }
  }

  // Apply pressure gradient with damping
  float pressure_scale = 0.8f * dt; // Damping factor
  for (int i = 0; i < GRID_HEIGHT; i++) {
    for (int j = 1; j < GRID_WIDTH; j++) {
      u[i][j] -= pressure_scale * (p[i][j] - p[i][j - 1]) / DX;
    }
  }
  for (int i = 1; i < GRID_HEIGHT; i++) {
    for (int j = 0; j < GRID_WIDTH; j++) {
      v[i][j] -= pressure_scale * (p[i][j] - p[i - 1][j]) / DY;
    }
  }
}

/*
  Set boundry velocities
  from edge cells to outside = 0
*/
void enforce_Boundaries() {
  for (int i = 0; i < GRID_HEIGHT; i++) {
    u[i][0] = 0;
    u[i][GRID_WIDTH] = 0;
  }
  for (int i = 0; i < GRID_WIDTH; i++) {
    v[GRID_HEIGHT][i] = 0;
    v[0][i] = 0;
  }
}

void simulate_Step(float dt) {
  // Limit maximum velocity to prevent instability
  float max_vel = 0.0f;
  for (int i = 0; i <= GRID_HEIGHT; i++) {
    for (int j = 0; j <= GRID_WIDTH; j++) {
      if (i < GRID_HEIGHT)
        max_vel = fmax(max_vel, fabs(u[i][j]));
      if (j < GRID_WIDTH)
        max_vel = fmax(max_vel, fabs(v[i][j]));
    }
  }

  // Adaptive time stepping
  float safe_dt = fmin(DT, 0.5f * DX / max_vel);

  apply_Gravity(safe_dt);
  enforce_Boundaries();
  advect_Velocity(safe_dt);
  enforce_Boundaries();
  project_Pressure(safe_dt);
  enforce_Boundaries();
}

void init_Sim() {
  memset(u, 0, sizeof(u));
  memset(v, 0, sizeof(v));
  memset(p, 0, sizeof(p));

  // Create a gentle circular vortex
  for (int i = 0; i < GRID_HEIGHT; i++) {
    for (int j = 0; j < GRID_WIDTH; j++) {
      float dx = j - GRID_WIDTH / 2;
      float dy = i - GRID_HEIGHT / 2;
      float dist = sqrtf(dx * dx + dy * dy);
      if (dist < 10.0f && dist > 0.1f) {
        u[i][j] = -dy / dist * 0.5f;
        v[i][j] = dx / dist * 0.5f;
      }
    }
  }
}

int main() {

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! Error: %s\n", SDL_GetError());
    return 1;
  }

  window = SDL_CreateWindow("Fluid Simulation", SDL_WINDOWPOS_CENTERED,
                            SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT,
                            SDL_WINDOW_SHOWN);

  if (!window) {
    printf("Window could not be created! Error: %s\n", SDL_GetError());
    return 1;
  }

  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (!renderer) {
    printf("Renderer could not be created! Error: %s\n", SDL_GetError());
    return 1;
  }

  init_Sim();

  int quit = 0;
  SDL_Event e;

  for (int frame = 0; frame < 1000 && !quit; frame++) {
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) {
        quit = 1;
      }
    }

    simulate_Step(DT);
    visualize();
    SDL_Delay(16); // ~60 FPS
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
  return 0;
}
