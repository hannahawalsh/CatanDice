from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import mean_squared_error as mse
import altair as alt



class StreamlitStyle:

    def get_number_text(n):
        """ Given a number, return the html for displaying it"""
        color = "red" if n in [6, 8] else "black"
        style = ("<h1 style='text-align: center; font-size: 12.0em; "
                 f"font-family: Arial Black; padding: 0px; color: {color};'> "
                 f"{n} </h1>")
        return style

    def get_name_text(name):
        """ Given a player's name, return the html for displaying it """
        style = ("<h1 style='text-align: center; font-size: 4.0em; font-family:"
                 f" Arial; padding: 0px'> {name} </h1>")
        return style

    def style_frequencies(df):
        color_results = pd.DataFrame(columns=df.columns, index=df.index)
        norm = colors.Normalize(0.0, 2.5)
        for nm, col in df.items():
            if nm in ["Roll", "Theoretical"]:
                color_results[nm] = ""
            else:
                chg = (pd.DataFrame.from_dict({"x": df["Theoretical"],
                                               "xhat": col}, orient="index")
                                   .pct_change().iloc[-1].abs().values)
                all_colors = [colors.rgb2hex(x) for x in
                              plt.cm.get_cmap("Reds")(norm(chg))]
                color_results[nm] = pd.Series([f"background-color: {clr}" for clr
                                               in all_colors])

        to_color = lambda col, color_map_df: color_map_df[col.name]
        return df.style.apply(to_color, args=(color_results,),
                              axis=0).set_precision(3)


class Dice:
    def __init__(self):
        self.fair_weights = {**{i: (i-1)/36 for i in range(2, 8)},
                             **{i: (13-i)/36 for i in range(8, 13)}}


    def calculate_frequencies(self, roll_history, player_names):
        """
        Calculate the roll frequencies for the game and each player.
        """
        player_rolls = {p: roll_history[i::len(player_names)] for
                        i, p in enumerate(player_names)}
        player_freqs = {p: {k: v/len(rs) for k, v in Counter(rs).items()}
                        for p, rs in player_rolls.items()}
        player_freqs = pd.DataFrame(player_freqs).sort_index()
        fair_series = pd.Series(self.fair_weights, name="Theoretical")
        all_roll_freq = pd.Series(self.normalize_dict(Counter(
                                  roll_history)), name="All")
        frequencies = pd.concat([fair_series, all_roll_freq, player_freqs],
                                 axis=1).reset_index()
        frequencies = frequencies.rename(columns={"index": "Roll"}).fillna(0.0)
        return frequencies, player_rolls


    def normalize_dict(self, D):
        """
        Return a dictionary with values normalized and missing values at 0
        """
        new = {k: v / sum(D.values()) for k, v in D.items()}
        missing_vals = [x for x in range(2, 13) if x not in new.keys()]
        new = {**new, **{m: 0.0 for m in missing_vals}}
        return {k: new[k] for k in range(2, 13)}


    def gambler_weights(self, rate, current_weights):
        """ """
        new = {roll: 2 ** (rate * (self.fair_weights[roll] -
               current_weights[roll])) for roll in range(2, 13)}
        return self.normalize_dict(new)


    def roll(self, roll_history, first, random_rate, convergence_rate):
        """
        """
        if len(roll_history) <= first or np.random.random() <= random_rate:
            roll = np.random.choice(list(self.fair_weights.keys()),
                                    p=list(self.fair_weights.values()))
        else:
            past_frequency = self.normalize_dict(Counter(roll_history))
            new_weights = self.gambler_weights(convergence_rate,
                                                past_frequency)
            roll = np.random.choice(list(new_weights.keys()),
                                    p=list(new_weights.values()))
        return roll


    def new_roll(self, roll_history, player_names, current_player, first,
                 random_rate, convergence_rate):
        """
        """
        # Randomly roll with fair weights
        if len(roll_history) <= first or np.random.random() <= random_rate:
            roll = np.random.choice(list(self.fair_weights.keys()),
                                    p=list(self.fair_weights.values()))
        else:
            # Get all player + current player frequencies
            freq_df, _ = self.calculate_frequencies(roll_history, player_names)
            player_freq = freq_df.set_index("Roll")[current_player].to_dict()
            all_freq = freq_df.set_index("Roll")["All"].to_dict()

            # Get the new roll weights and roll
            new_all_weights = self.gambler_weights(convergence_rate, all_freq)
            new_player_weights = self.gambler_weights(convergence_rate,
                                                      player_freq)
            p_roll = np.random.choice(list(new_player_weights.keys()),
                                      p=list(new_player_weights.values()))
            a_roll = np.random.choice(list(new_all_weights.keys()),
                                      p=list(new_all_weights.values()))

            # Randomly select one of those rolls, weighting the probability by
            # the current roll numnber
            # p_weight = min(len(roll_history), 45) / 50
            # a_weight = 1 - p_weight
            # roll = np.random.choice([p_roll, a_roll], p=[p_weight, a_weight])
            roll = np.random.choice([p_roll, a_roll])
        return roll





    def game_stats(self, roll_history, player_names):
        """
        """
        def plot_history(frequencies):
            """
            Create a plot showing the true and theoretical breakdown of the
            rolls. The plot has a bar for each player.
            """
            freq_plot, freq_ax = plt.subplots(figsize=(20, 10))
            freq_ax.set_xlim(1, 13)
            frequencies.drop(columns=["Theoretical", "All"]).plot(x="Roll",
                             kind="bar", color=["red", "blue", "green",
                             "orange"], ax=freq_ax)
            frequencies["Theoretical"].plot(x="Roll", style="*-", ms=25, lw=3,
                                            color="Fuchsia", ax=freq_ax)
            frequencies["All"].plot(x="Roll", style=".-k", ms=25, lw=3,
                                    ax=freq_ax)

            freq_ax.set_title("Roll Frequencies By Player", fontsize=48)
            freq_ax.set_xlabel("Roll", fontsize=36, labelpad=20)
            freq_ax.tick_params(axis="x", pad=20, labelsize=28)
            freq_ax.tick_params(axis="y", pad=10, labelsize=24)
            leg_labels = freq_ax.get_legend_handles_labels()[1]
            freq_ax.legend(leg_labels, fontsize=20, loc="upper right")
            return freq_plot


        freqs, player_rolls = self.calculate_frequencies(roll_history,
                                                         player_names)
        freq_plot = plot_history(freqs)
        roll_breakdown = StreamlitStyle.style_frequencies(freqs.fillna(0.0))
        turn_count = pd.DataFrame({**{k: len(v) for k, v in player_rolls.items()
                                  }, "All": len(roll_history)}, index=["Rolls"])
        return freq_plot, roll_breakdown, turn_count, freqs
