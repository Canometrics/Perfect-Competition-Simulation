# ================ CONFIG ================

# ----- SIMULATION -----
SEED = 7
T = 600                 # ticks
tatonnement_speed = 0.8        # price adjustment speed
price_alpha = 0.3       # price smoothing factor (to avoid extremely jagged prices)

# ----- FIRMS -----
N_FIRMS = 60

# Draws for firm heterogeneity
# FC ~ lognormal, MC ~ normal clipped, capacity ~ uniform
FC_LOGMEAN, FC_LOGSD = 2.0, 0.6      # exp draw, scale FC (multiplied by 20)
MC_MEAN, MC_SD = 1, 0.1              # mean MC near the price region
CAP_LOW, CAP_HIGH = 20, 150          # capacity range per firm, hi val used to be 100

# ----- WAGES -----
WAGE = 1.0  # wage per worker per tick

# ----- ENTRY -----
ENTRY_ALPHA = 0.002          # controls steepness of Pr(entry)
ENTRY_WINDOW = 8             # lookback window (ticks) for avg profit
ENTRY_MAX_PER_TICK = 0.1       # cap entrants per tick

# ----- TREASURY / CAPITAL BUFFER -----
START_CAPITAL = 6000.0             # initial cash buffer for each firm
TREASURY_GRACE_TICKS = 2           # shutdown if treasury < 0 for this many consecutive ticks