# ─────────────────────────────────────────────
#  Stealth Escape RL — Configuration  (FIXED v3)
# ─────────────────────────────────────────────

# Environment
GRID_SIZE       = 15
NUM_GUARDS      = 3
MAX_STEPS       = 300
NUM_WALLS       = 20

# Guard settings
GUARD_FOV_ANGLE = 90
GUARD_FOV_RANGE = 4
GUARD_SPEED     = 1
ALERT_DURATION  = 10

# Rewards
REWARD_REACH_EXIT   =  20.0
REWARD_CAUGHT       = -15.0   # was -1.0 → 15x stronger: agent must fear guards
REWARD_STEP         =  -0.001
REWARD_TOWARD_EXIT  =   0.3
REWARD_AWAY_EXIT    =  -0.1
REWARD_SURVIVE_STEP =   0.02

# PPO Hyperparameters  — tuned for sparse-reward stealth task
PPO_TIMESTEPS       = 800_000   # was 200k → needs more to learn evasion + pathing
PPO_LEARNING_RATE   = 2e-4      # slightly lower = more stable
PPO_N_STEPS         = 2048      # larger rollout = better credit assignment
PPO_BATCH_SIZE      = 128       # bigger batch = smoother gradients
PPO_N_EPOCHS        = 10
PPO_GAMMA           = 0.995     # higher = agent thinks longer-term (needed for escape)
PPO_CLIP_RANGE      = 0.2
PPO_ENT_COEF        = 0.01      # lower entropy = exploit once agent finds escape

# Paths
MODEL_SAVE_PATH = "models/stealth_ppo"
LOG_PATH        = "logs/"

# Rendering
CELL_SIZE       = 48
FPS             = 10