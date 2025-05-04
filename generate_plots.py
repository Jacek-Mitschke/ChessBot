import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIG ===
#core_results_dir = "experiment_results_stochastic"
#targeted_results_dir = "targeted_experiments_stochastic"
#output_dir = "figures_stochastic"

core_results_dir = "experiment_results_non_stochastic"
targeted_results_dir = "targeted_experiments_non_stochastic"
output_dir = "figures_non_stochastic"

os.makedirs(output_dir, exist_ok=True)

# 1. Match Outcome Bar Chart

core_files = {
    "DQN vs Minimax": "evaluation_dqn_vs_minimax.csv",
    "DQN vs Minimax Midgame": "evaluation_dqn_vs_minimax_midgame.csv",
    "DQN vs MCTS": "evaluation_dqn_vs_mcts.csv",
    "MCTS vs Minimax": "evaluation_mcts_vs_minimax.csv"
}

match_summary = []
for label, filename in core_files.items():
    path = os.path.join(core_results_dir, filename)
    df = pd.read_csv(path)
    match_summary.append({
        "Matchup": label,
        "Agent1 Wins": (df["Winner"] == "dqn").sum() + (df["Winner"] == "mcts").sum(),
        "Agent2 Wins": (df["Winner"] == "minimax").sum(),
        "Draws": (df["Winner"] == "draw").sum()
    })

match_df = pd.DataFrame(match_summary)
ax = match_df.set_index("Matchup")[["Agent1 Wins", "Agent2 Wins", "Draws"]].plot(
    kind="bar", stacked=True, colormap="Set2", figsize=(10, 6)
)
ax.set_ylabel("Games")
ax.set_title("Match Outcomes per Agent Pairing")
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "match_results_plot.png"))
plt.close()

# 2. Q-Value Confidence Line Plot: dqn vs minimax

qval_file = os.path.join(targeted_results_dir, "dqn_vs_minimax_qval.csv")
df_qval = pd.read_csv(qval_file)

plt.figure(figsize=(8, 5))

# Plot each Q-value line with distinct styles
plt.plot(df_qval["Game"], df_qval["QChosen_Mean"], label="QChosen", color="blue", linestyle='-', marker='o')
plt.plot(df_qval["Game"], df_qval["QMax_Mean"], label="QMax", color="orange", linestyle='--', marker='s')
plt.plot(df_qval["Game"], df_qval["QMean_Mean"], label="QMean", color="green", linestyle='-.', marker='^')

plt.title("Q-Value Confidence per Game (DQN vs Minimax)", fontsize=14)
plt.xlabel("Game", fontsize=12)
plt.ylabel("Average Q-Value", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.ylim(bottom=0)

plt.savefig(os.path.join(output_dir, "dqn_vs_minimax_q_value_plot.png"))
plt.close()


# 3. Q-Value Confidence Line Plot: dqn selfplay

qval_file = os.path.join(targeted_results_dir, "dqn_vs_dqn_selfplay.csv")
df_qval = pd.read_csv(qval_file)

plt.figure(figsize=(8, 5))

# Plot each Q-value line with distinct styles
plt.plot(df_qval["Game"], df_qval["QChosen_Mean"], label="QChosen", color="blue", linestyle='-', marker='o')
plt.plot(df_qval["Game"], df_qval["QMax_Mean"], label="QMax", color="orange", linestyle='--', marker='s')
plt.plot(df_qval["Game"], df_qval["QMean_Mean"], label="QMean", color="green", linestyle='-.', marker='^')

plt.title("Q-Value Confidence per Game (DQN vs DQN)", fontsize=14)
plt.xlabel("Game", fontsize=12)
plt.ylabel("Average Q-Value", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.ylim(bottom=0)

plt.savefig(os.path.join(output_dir, "dqn_selfplay_q_value_plot.png"))
plt.close()

# 4. Q-Value Confidence Line Plot: dqn vs mcts

qval_file = os.path.join(targeted_results_dir, "dqn_vs_mcts_midgame.csv")
df_qval = pd.read_csv(qval_file)

plt.figure(figsize=(8, 5))

# Plot each Q-value line with distinct styles
plt.plot(df_qval["Game"], df_qval["QChosen_Mean"], label="QChosen", color="blue", linestyle='-', marker='o')
plt.plot(df_qval["Game"], df_qval["QMax_Mean"], label="QMax", color="orange", linestyle='--', marker='s')
plt.plot(df_qval["Game"], df_qval["QMean_Mean"], label="QMean", color="green", linestyle='-.', marker='^')

plt.title("Q-Value Confidence per Game (DQN vs MCTS)", fontsize=14)
plt.xlabel("Game", fontsize=12)
plt.ylabel("Average Q-Value", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.ylim(bottom=0)

plt.savefig(os.path.join(output_dir, "dqn_vs_mcts_q_value_plot.png"))
plt.close()

# 5. Q-Value Confidence Line Plot: dqn vs scripted human

qval_file = os.path.join(targeted_results_dir, "dqn_vs_scripted_human.csv")
df_qval = pd.read_csv(qval_file)

plt.figure(figsize=(8, 5))

# Plot each Q-value line with distinct styles
plt.plot(df_qval["Game"], df_qval["QChosen_Mean"], label="QChosen", color="blue", linestyle='-', marker='o')
plt.plot(df_qval["Game"], df_qval["QMax_Mean"], label="QMax", color="orange", linestyle='--', marker='s')
plt.plot(df_qval["Game"], df_qval["QMean_Mean"], label="QMean", color="green", linestyle='-.', marker='^')

plt.title("Q-Value Confidence per Game (DQN vs Human)", fontsize=14)
plt.xlabel("Game", fontsize=12)
plt.ylabel("Average Q-Value", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.ylim(bottom=0)

plt.savefig(os.path.join(output_dir, "dqn_vs_human_q_value_plot.png"))
plt.close()

