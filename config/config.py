# ================ CONFIG ================

# ----- SIMULATION -----
SEED = 7
T = 600                 # ticks
tatonnement_speed = 0.8        # price adjustment speed
price_alpha = 0.3       # price smoothing factor (to avoid extremely jagged prices)

# ----- FIRMS -----
N_FIRMS = 60
RECORD_FIRM_HISTORY = False  # keep a per-tick row for every firm (slow, memory-heavy; only for firm-level analysis)

# Draws for firm heterogeneity
# FC ~ lognormal, MC ~ normal clipped, capacity ~ uniform
FC_LOGMEAN, FC_LOGSD = 2.0, 0.6      # exp draw, scale FC (multiplied by 20)
MC_MEAN, MC_SD = 1, 0.1              # mean MC near the price region
CAP_LOW, CAP_HIGH = 20, 150          # capacity range per firm, hi val used to be 100

# ----- WAGES / LABOR MARKET -----
WAGE = 1.0                   # initial wage per worker per tick (each province's wage moves from here)
TARGET_EMPLOYMENT = 0.95     # employment rate at which a province's wage holds steady
WAGE_ADJ_SPEED = 0.05        # max fractional wage change per tick (reached at full employment)
WAGE_MIN = 0.05              # wage floor

# ----- ENTRY -----
ENTRY_ALPHA = 0.002          # controls steepness of Pr(entry)
ENTRY_WINDOW = 8             # lookback window (ticks) for avg profit
ENTRY_MAX_PER_TICK = 0.1       # cap entrants per tick
ENTRANT_RIGHTS_LOW, ENTRANT_RIGHTS_HIGH = 0.02, 0.08  # resource-rights share drawn by a single RGO entrant

# ----- TREASURY / CAPITAL BUFFER -----
START_CAPITAL = 6000.0             # initial cash buffer for each firm
TREASURY_GRACE_TICKS = 2           # shutdown if treasury < 0 for this many consecutive ticks