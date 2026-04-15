import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("🧠 Social Curiosity Agent (Tunable)")

# =========================
# UI
# =========================
left, right = st.columns([1, 2])

with left:
    st.markdown("## 🎛 Agent Mind")

    temperature = st.slider("🧭 Exploration vs Routine", 0.1, 5.0, 1.0)
    beta = st.slider("🌍 Space Curiosity", 0.5, 5.0, 2.0)
    gamma = st.slider("❤️ Social Curiosity", 0.5, 5.0, 2.0)
    decay = st.slider("💤 Boredom Speed", 0.1, 2.0, 0.5)

    lr = st.slider("🧠 Emotion Learning Rate", 0.05, 1.0, 0.3)

    st.markdown("## 🎭 Social Environment")

    volatility = st.slider("🎭 Emotional Volatility", 0.05, 1.0, 0.4)
    perception_range = st.slider("👀 Perception Range", 0.5, 5.0, 2.0)
    social_drive = st.slider("🧲 Social Seeking Drive", 0.0, 0.2, 0.05)

    grid_size = st.slider("🌍 World Size", 3, 10, 5)

    start = st.button("▶ Start Exploring")
    reset = st.button("🔄 Reset")

with right:
    grid_placeholder = st.empty()
    info_placeholder = st.empty()
    plot_placeholder = st.empty()

# =========================
# INIT
# =========================
if "visited" not in st.session_state or st.session_state.visited.shape[0] != grid_size:
    st.session_state.visited = np.zeros((grid_size, grid_size))
    st.session_state.visit_count = np.zeros((grid_size, grid_size))
    st.session_state.pos = [grid_size // 2, grid_size // 2]

    # 👉 社交对象放在中心区域
    center = grid_size // 2
    st.session_state.social_pos = [
        np.clip(center + np.random.randint(-1, 2), 1, grid_size - 2),
        np.clip(center + np.random.randint(-1, 2), 1, grid_size - 2),
    ]

    st.session_state.true_emotion = 0.5
    st.session_state.pred_emotion = 0.5

    st.session_state.t = 0
    st.session_state.time_since_seen = 0

    st.session_state.pe_space_hist = []
    st.session_state.pe_social_hist = []

# =========================
# RESET
# =========================
if reset:
    st.session_state.visited *= 0
    st.session_state.visit_count *= 0
    st.session_state.pos = [grid_size // 2, grid_size // 2]

    center = grid_size // 2
    st.session_state.social_pos = [
        np.clip(center + np.random.randint(-1, 2), 1, grid_size - 2),
        np.clip(center + np.random.randint(-1, 2), 1, grid_size - 2),
    ]

    st.session_state.pred_emotion = 0.5
    st.session_state.t = 0
    st.session_state.time_since_seen = 0

    st.session_state.pe_space_hist = []
    st.session_state.pe_social_hist = []

# =========================
# FUNCTIONS
# =========================
def get_moves(pos):
    x, y = pos
    moves = []
    if x > 0: moves.append([x-1, y])
    if x < grid_size-1: moves.append([x+1, y])
    if y > 0: moves.append([x, y-1])
    if y < grid_size-1: moves.append([x, y+1])
    return moves

def pe_space(x, y):
    v = st.session_state.visit_count[x, y]
    return np.exp(-decay * v)

def update_true_emotion():
    st.session_state.t += 1
    st.session_state.true_emotion = 0.5 + volatility * np.sin(0.2 * st.session_state.t)

def choose(moves):

    qs = []
    for x, y in moves:

        s_pe = pe_space(x, y)

        dist = abs(x - st.session_state.social_pos[0]) + abs(y - st.session_state.social_pos[1])

        # 👉 可调感知距离
        visibility = np.exp(-dist / perception_range)

        e_pe = abs(st.session_state.pred_emotion - st.session_state.true_emotion) * visibility

        # 👉 可调主动社交动机
        uncertainty_bonus = social_drive * st.session_state.time_since_seen

        q = beta * s_pe + gamma * (e_pe + uncertainty_bonus)
        qs.append(q)

    qs = np.array(qs)
    qs = qs - np.max(qs)

    probs = np.exp(qs / temperature)
    probs /= np.sum(probs)

    return moves[np.random.choice(len(moves), p=probs)]

def step():

    pos = st.session_state.pos
    moves = get_moves(pos)
    new = choose(moves)

    x, y = new

    st.session_state.visit_count[x, y] += 1
    st.session_state.visited[x, y] = 1

    update_true_emotion()

    dist = abs(x - st.session_state.social_pos[0]) + abs(y - st.session_state.social_pos[1])

    if dist == 0:
        obs = st.session_state.true_emotion
        pred = st.session_state.pred_emotion

        pe_e = abs(pred - obs)

        st.session_state.pred_emotion += lr * (obs - pred)

        st.session_state.time_since_seen = 0

    else:
        visibility = np.exp(-dist / perception_range)
        pe_e = abs(st.session_state.pred_emotion - st.session_state.true_emotion) * visibility

        st.session_state.time_since_seen += 1

    pe_s = pe_space(x, y)

    st.session_state.pe_space_hist.append(pe_s)
    st.session_state.pe_social_hist.append(pe_e)

    st.session_state.pos = new

# =========================
# RUN
# =========================
if start:

    for _ in range(grid_size * grid_size * 3):

        step()

        grid_display = ""
        for i in range(grid_size):
            for j in range(grid_size):

                if [i, j] == st.session_state.pos:
                    grid_display += "🤖 "
                elif [i, j] == st.session_state.social_pos:
                    grid_display += "🙂 "
                elif st.session_state.visited[i, j] > 0:
                    grid_display += "🟥 "
                else:
                    grid_display += "⬜ "

            grid_display += "\n"

        grid_placeholder.code(grid_display)

        info_placeholder.write(
            f"Space PE: {st.session_state.pe_space_hist[-1]:.2f} | "
            f"Social PE: {st.session_state.pe_social_hist[-1]:.2f} | "
            f"Pred: {st.session_state.pred_emotion:.2f} | "
            f"True: {st.session_state.true_emotion:.2f}"
        )

        fig, ax = plt.subplots()

        ax.plot(st.session_state.pe_space_hist, linewidth=0.8, label="space")
        ax.plot(st.session_state.pe_social_hist, linewidth=0.8, label="social")

        ax.set_title("PE", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

        fig.set_size_inches(3.8, 1.2)

        plot_placeholder.pyplot(fig)

        time.sleep(0.15)
