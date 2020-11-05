import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import colors, cm
import altair as alt
import pyautogui
from sklearn.metrics import mean_squared_error as mse



def adjust_colors(color_list):
    """ If a color is too light, make it a bit darker"""
    for i, clr in enumerate(color_list):
        rgb_col = colors.to_rgb(clr)
        if sum(rgb_col) >= 2.5:
            darkened = [min(1.0, x * 0.925) for x in rgb_col]
            color_list[i] = colors.to_hex(darkened)
    return color_list


class VAR:
    fair_wts = {**{i: (i-1)/36 for i in range(2, 8)},
                **{i: (13-i)/36 for i in range(8, 13)}}


class StreamlitStyle:

    def get_number_text(n):
        """ Given a number, return the html for displaying it"""
        color = "red" if n in [6, 8] else "black"
        style = ("<h1 style='text-align: center; font-size: 12.0em; "
                 f"font-family: Arial Black; padding: 0px; color: {color};'> "
                 f"{n} </h1>")
        return style


    def get_name_text(name, color="#000000"):
        """ Given a player's name, return the html for displaying it """
        color = adjust_colors([color])[0]
        font_color = "black" if sum(colors.to_rgb(color)) >= 2.0 else "white"

        style = ("<h1 style='text-align: center; font-size: 4.0em; font-family:"
                 f" Arial; padding: 10px; color: {font_color}; background: "
                 f"{color};'> <span> {name} </span> </h1>")
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
    def calculate_frequencies(self, roll_history, player_names):
        """
        Calculate the roll frequencies for the game and each player.
        """
        player_rolls = {p: roll_history[i::len(player_names)] for
                        i, p in enumerate(player_names)}
        player_freqs = {p: {k: v/len(rs) for k, v in Counter(rs).items()}
                        for p, rs in player_rolls.items()}
        player_freqs = pd.DataFrame(player_freqs).sort_index()
        fair_series = pd.Series(VAR.fair_wts, name="Theoretical")
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
        new = {roll: 2 ** (rate * (VAR.fair_wts[roll] -
               current_weights[roll])) for roll in range(2, 13)}
        return self.normalize_dict(new)


    def roll(self, roll_history, first, random_rate, convergence_rate):
        """
        """
        if len(roll_history) <= first or np.random.random() <= random_rate:
            roll = np.random.choice(list(VAR.fair_wts.keys()),
                                    p=list(VAR.fair_wts.values()))
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
            roll = np.random.choice(list(VAR.fair_wts.keys()),
                                    p=list(VAR.fair_wts.values()))
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


    def plot_difference(roll_history, player_names):
        freqs, player_rolls = self.calculate_frequencies(roll_history,
                                                         player_names)




class PlotResults:
    def __init__(self, roll_history, player_names, player_colors):
        """   """
        self.roll_history = roll_history
        self.player_names = player_names
        self.player_colors = adjust_colors(list(player_colors.values()))
        self.screen_width  = pyautogui.size().width

        # Data frame of all turns with customers
        n = len(roll_history)
        self.turns = pd.DataFrame({"Player": (player_names * n)[:n],
                                   "Roll": roll_history})

        # All game roll counts
        h_cnt = self.turns["Roll"].value_counts()
        h_cnt = pd.concat([h_cnt, pd.Series(VAR.fair_wts) * n],
                           axis=1).reset_index()
        h_cnt.columns = ["Roll", "Count", "Expected"]
        h_cnt["Difference"] = h_cnt["Count"] - h_cnt["Expected"]
        self.history_count = h_cnt.fillna(0)

        # Per-player roll counts
        rl_grp = self.turns.groupby("Player")["Roll"]
        get_expected = lambda grp: pd.Series(VAR.fair_wts) * len(grp)
        p_cnt = pd.concat([rl_grp.value_counts().rename("Count"),
                           rl_grp.apply(get_expected)], axis=1).reset_index()
        p_cnt.columns = ["Player", "Roll", "Count", "Expected"]
        p_cnt["Difference"] = p_cnt["Count"] - p_cnt["Expected"]
        self.player_count = p_cnt.fillna(0)


    def get_turn_count(self):
        cnts = pd.DataFrame(self.turns.groupby("Player").count().to_dict("index"))
        cnts["All"] = len(self.turns)
        cnts.index = ["Turn Count"]
        return cnts


    def get_divergence_chart(self):
        """   """
        # Get colors for bars
        normalize = lambda s: (s - s.min()) / (s.max() - s.min())
        rdiffs = normalize(self.history_count["Difference"])
        reds =  [cm.get_cmap('Reds_r', 51)(i) for i in range(51)]
        reds += reds[::-1]
        roll_colors = [colors.to_hex(reds[int(100 * d)]) for d in rdiffs]

        # Make Altair horizontal bar chart
        plt_df = self.history_count.round()
        diff_chart = alt.Chart(plt_df).mark_bar(size=30).encode(
            y='Roll:O',
            x='Difference:Q',
            color=alt.Color('Roll:Q', legend=None, scale=alt.Scale(
                domain=self.history_count.Roll.to_list(), range=roll_colors)),
            tooltip=list(plt_df.columns)
        ).properties(
            title="Roll Differential from Expected Count",
            width=self.screen_width / 4, height=alt.Step(32)
        ).configure_title(
            fontSize=32, dy=-50, limit=600, font="Arial", align="center",
            anchor="middle"
        ).configure_axis(
            labelFontSize=14, titleFontSize=16
        )

        roll_count = self.history_count[["Roll", "Count"]].T
        roll_count.columns = [""] * len(roll_count.columns)

        return diff_chart, roll_count


    def player_roll_chart(self):
        """ """
        # Make Altair bark chart
        plt_df = self.player_count.round(2)
        roll_chart = alt.Chart(plt_df).mark_bar().encode(
            x=alt.X("Player:O", axis=alt.Axis(title=None, labels=False,
                    ticks=False)),
            y='Count:Q',
            color=alt.Color('Player:N', scale=alt.Scale(
                            domain=self.player_names, range=self.player_colors),
                            legend=alt.Legend()),
            column=alt.Column("Roll:N", header=alt.Header(title=None,
                              labelOrient="bottom", labelFontSize=22)),
            tooltip=list(self.player_count.columns)
        ).configure_view(
            strokeWidth=0
        ).configure_title(
            fontSize=32, limit=800, dx=45, dy=-50,
            font="Arial", align="center", anchor="middle"
        ).configure_legend(
            strokeColor="black", padding=10, orient="bottom", cornerRadius=10,
            direction="horizontal", labelFontSize=10
        ).properties(
            title="Roll Count by Player",
            width=self.screen_width / 45
        ).configure_axis(
            grid=False, labelFontSize=14, titleFontSize=16
        )


        return roll_chart


    def player_diff_chart(self):
        """ Create an Altair chart showing difference from expected rolls """
        # Create chart
        plt_df = self.player_count.round(2)
        p_diff_chart = alt.Chart(plt_df).mark_bar().encode(
            x=alt.X("Player:O", axis=alt.Axis(title=None, labels=False,
                    ticks=False)),
            y="Difference:Q",
            color=alt.Color("Player:N", scale=alt.Scale(
                            domain=self.player_names, range=self.player_colors),
                            legend=alt.Legend()),
            column=alt.Column("Roll:N", header=alt.Header(title=None,
                              labelOrient="bottom", labelFontSize=22)),
            tooltip=list(plt_df.columns)
        ).configure_view(
            strokeWidth=0
        ).configure_title(
            fontSize=32, limit=800, dx=45, dy=-50,
            font="Arial", align="center", anchor="middle"
        ).configure_legend(
            strokeColor="black", padding=10, orient="bottom", cornerRadius=10,
            direction="horizontal", labelFontSize=10
        ).properties(
            title="Roll Differential by Player",
            width=self.screen_width / 45
        ).configure_axis(
            grid=False, labelFontSize=14, titleFontSize=16
        )


        return p_diff_chart

### Need to fill
