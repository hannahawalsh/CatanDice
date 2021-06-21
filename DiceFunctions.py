"""
This script contains the functions and style definitions for CatanDice.py
"""


import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import colors, cm
import altair as alt
from sklearn.metrics import mean_squared_error as mse


def adjust_colors(color_list):
    """ If a color is too light, make it a bit darker. """
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
    # Colors
    CatanRed = "#C62028"
    CatanGold = "#FACC0B"

    player_colors = {
        "red": "#F12627",
        "blue": "#044F9B",
        "orange": "#F37824",
        "white": "#FFFFFF",
        "green": "#50833C",
        "brown": "#914B1D"
        }
    color_names = list(player_colors.keys())


    def get_number_text(n):
        """ Given a number, return the html for displaying it"""
        color = "red" if n in [6, 8] else "black"
        style = ("<h1 style='text-align: center; font-size: 20.0em; "
                 f"font-family: Arial Black; padding: 0px; color: {color};'> "
                 f"{n} </h1>")
        return style


    def get_name_text(name, color="#000000"):
        """ Given a player's name, return the html for displaying it """
        outline = "black" if color == "#FFFFFF" else "transparent"

        style_dict = {
            "text-align": "center",
            "font-size": "4.0em",
            "font-family": "Arial",
            "padding": "10px",
            "color": color,
            "background": "transparent",
            "-webkit-text-stroke-width": "2px",
            "-webkit-text-stroke-color": outline,
        }
        style_string = "; ".join([f"{k}: {v}" for k, v in style_dict.items()])
        style = (f"<h1 style='{style_string}'> <span> {name} </span> </h1>")
        return style


    # def style_frequencies(df):
    #     """ Style the frequency plot. """
    #     color_results = pd.DataFrame(columns=df.columns, index=df.index)
    #     norm = colors.Normalize(0.0, 2.5)
    #     for nm, col in df.items():
    #         if nm in ["Roll", "Theoretical"]:
    #             color_results[nm] = ""
    #         else:
    #             chg = (pd.DataFrame.from_dict({"x": df["Theoretical"],
    #                                            "xhat": col}, orient="index")
    #                                .pct_change().iloc[-1].abs().values)
    #             all_colors = [colors.rgb2hex(x) for x in
    #                           plt.cm.get_cmap("Reds")(norm(chg))]
    #             color_results[nm] = pd.Series([f"background-color: {clr}" for clr
    #                                            in all_colors])
    #
    #     to_color = lambda col, color_map_df: color_map_df[col.name]
    #     return df.style.apply(to_color, args=(color_results,),
    #                           axis=0).set_precision(3)



class Dice:
    def normalize_dict(self, D):
        """
        Return a dictionary with values normalized and missing values at 0
        """
        new = {k: v / sum(D.values()) for k, v in D.items()}
        missing_vals = [x for x in range(2, 13) if x not in new.keys()]
        new = {**new, **{m: 0.0 for m in missing_vals}}
        return {k: new[k] for k in range(2, 13)}


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


    def gambler_weights(self, rate, history):
        """
        Return new weights according to the gambler's fallacy such that the
        distribution of rolls actively converges to the expected distribution.
        """
        current_dist = (pd.Series(history).value_counts(normalize=True)
                          .reindex(range(2, 13)).fillna(0))
        fair_dist = pd.Series(VAR.fair_wts)
        diff =  fair_dist - current_dist
        f_rate = 200 + rate * 250
        step = {idx: 2 ** (val * f_rate) for idx, val in diff.items()}
        return self.normalize_dict(step)


    def roll(self, roll_history, first, random_rate, convergence_rate):
        """
        Perform a dice roll:
            - If the roll is within the first <first> rolls, roll randomly
            - According to the random rate, choose whether to roll randomly
            - Otherwise, calculate the new weights and roll according to the
              gambler's fallacy weights
        """
        if len(roll_history) < first or np.random.random() <= random_rate:
            roll = np.random.choice(list(VAR.fair_wts.keys()),
                                    p=list(VAR.fair_wts.values()))
        else:
            new_weights = self.gambler_weights(convergence_rate, roll_history)
            roll = np.random.choice(list(new_weights.keys()),
                                    p=list(new_weights.values()))
        return roll


    def roll_balanced(self, roll_history, n_players, first, random_rate,
                      convergence_rate, player_weight):
        """
        Perform a dice roll according to each player's history:
            - If the roll is within the first <first> rolls, roll randomly
            - According to the random rate, choose whether to roll randomly
            - Otherwise, calculate the new weights and roll according to the
              gambler's fallacy weights
            - Player weights get more important as game progresses
        """
        # Roll history for current player
        player_roll_history = roll_history[(-1 * n_players)::(-1 * n_players)]

        # Roll normally if not enough turns have passed
        if len(roll_history) < (first * n_players) // 2:
            return self.roll(roll_history, first, random_rate, convergence_rate)

        # Randomly roll with fair weights
        elif np.random.random() <= random_rate:
            roll = np.random.choice(list(VAR.fair_wts.keys()),
                                    p=list(VAR.fair_wts.values()))
            return roll

        # Balance player and game distributions
        else:
            g_weights_all = self.gambler_weights(convergence_rate, roll_history)
            g_weights_player = self.gambler_weights(convergence_rate,
                                                    player_roll_history)
            p_weight = min(0.95, len(roll_history) / 45)
            new_weights = {r: g_weights_player[r] * p_weight + p * (1-p_weight)
                           for r, p in g_weights_all.items()}
            new_weights = self.normalize_dict(new_weights)
            roll = np.random.choice(list(new_weights.keys()),
                                    p=list(new_weights.values()))
            return roll


    def roll_balance_7s(self, roll_history, n_players, first, random_rate,
                        convergence_rate, player_weight):
        """
        Try to balance the sevens: NOT CURRENTLY IN USE OR WORKING
        """
        # Roll history for current player
        player_roll_history = roll_history[(-1 * n_players)::(-1 * n_players)]

        if len(roll_history) < first:
            return self.roll(roll_history, first, random_rate, convergence_rate)

        elif np.random.random() <= random_rate:
            roll = np.random.choice(list(VAR.fair_wts.keys()),
                                    p=list(VAR.fair_wts.values()))
            return roll

        else:

            g_weights_1 = self.gambler_weights(convergence_rate, roll_history)
            g_weights_2 = self.gambler_weights(convergence_rate,
                                               player_roll_history)

            def reduce_7(W, red):
                W[7] /= red
                return self.normalize_dict(W)

            if player_roll_history[-1] == player_roll_history[-2] == 7:
                reduce = 5
            elif player_roll_history[-1] == 7:
                reduce = 2
            elif player_roll_history[-2] == 7:
                reduce = 1.5
            else:
                reduce = 1
            g_weights_1 = reduce_7(g_weights_1, reduce)
            g_weights_2 = reduce_7(g_weights_2, reduce)
            p_weight = min(len(roll_history) / 25 * player_weight,
                           player_weight) # increases as game goes on
            new_weights = {r: g_weights_1[r] * p_weight + p * (1-p_weight)
                           for r, p in g_weights_2.items()}
            new_weights = self.normalize_dict(new_weights)
            roll = np.random.choice(list(new_weights.keys()),
                                    p=list(new_weights.values()))
            return roll



class PlotResults:
    def __init__(self, roll_history, player_names, player_colors):

        self.roll_history = roll_history
        self.player_names = player_names
        self.player_colors = adjust_colors(list(player_colors.values()))
        self.screen_width = 1440

        # Data frame of all turns with customers
        n = len(roll_history)
        dice = range(2, 13)
        self.turns = pd.DataFrame({"Player": (player_names * n)[:n],
                                   "Roll": roll_history})

        # All game roll counts
        h_cnt = self.turns["Roll"].value_counts()#.reindex(dice).fillna(0.0)
        h_cnt = pd.concat([h_cnt, pd.Series(VAR.fair_wts) * n],
                           axis=1).reset_index().fillna(0.0)
        h_cnt.columns = ["Roll", "Count", "Expected"]
        h_cnt["Difference"] = h_cnt["Count"] - h_cnt["Expected"]
        self.history_count = h_cnt

        # Per-player roll counts
        rl_grp = self.turns.groupby("Player")["Roll"]
        get_expected = lambda grp: pd.Series(VAR.fair_wts) * len(grp)
        p_cnt = pd.concat([rl_grp.value_counts().rename("Count"),
                           rl_grp.apply(get_expected)], axis=1
                          ).reset_index().fillna(0.0)
        p_cnt.columns = ["Player", "Roll", "Count", "Expected"]
        p_cnt["Difference"] = p_cnt["Count"] - p_cnt["Expected"]
        self.player_count = p_cnt


    def get_turn_count(self):
        """ Count total number of turns per player and overall. """
        cnts = pd.DataFrame(
            self.turns.groupby("Player").count().to_dict("index")
        )
        cnts["All"] = len(self.turns)
        cnts.index = ["Turn Count"]
        return cnts


    def get_divergence_chart(self):
        """
        Calculate and style the chart of how actual rolls diverged from the
        expected distribution.
        """
        # Get colors for bars
        normalize = lambda s: (s - s.min()) / (s.max() - s.min())
        rdiffs = normalize(self.history_count["Difference"])
        rdiffs = [d if not np.isnan(d) else 0 for d in rdiffs]
        reds =  [cm.get_cmap('Reds_r', 51)(i) for i in range(51)]
        reds += reds[::-1]
        roll_colors = [colors.to_hex(reds[int(100 * d)]) for d in rdiffs]

        # Make Altair horizontal bar chart
        plt_df = self.history_count.round(2)
        diff_chart = alt.Chart(plt_df).mark_bar(size=30, strokeWidth=3,
                                                stroke="black").encode(
            y='Roll:O',
            x=alt.X('Difference:Q', scale=alt.Scale(padding=25)),
            color=alt.Color('Roll:O', legend=None, scale=alt.Scale(
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



    def player_diff_chart(self):
        """
        Create an Altair chart showing difference from expected rolls for each
        player.
        """
        # Create chart
        plt_df = self.player_count.round(2)
        p_diff_chart = alt.Chart(plt_df).mark_bar(strokeWidth=0.5,
                                                  stroke="black").encode(
            x=alt.X("Player:O", axis=alt.Axis(title=None, labels=False,
                    ticks=False)),
            y="Difference:Q",
            color=alt.Color("Player:N", scale=alt.Scale(
                            domain=self.player_names, range=self.player_colors),
                            legend=alt.Legend()),
            column=alt.Column("Roll:O", header=alt.Header(title=None,
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


    def all_roll_chart(self):
        """
        """
        # Make Altair bar chart
        w = self.screen_width / 2.5
        plt_df = self.history_count.round(2)
        bars = alt.Chart(plt_df).mark_bar(color=StreamlitStyle.CatanGold,
                                          strokeWidth=0).encode(
            x="Roll:O",
            y="Count:Q",
            tooltip=list(plt_df.columns)
        )

        # Add lines at Theoretical values
        marks = alt.Chart(plt_df).mark_tick(color=StreamlitStyle.CatanRed,
                                             width=50, thickness=3).encode(
            x="Roll:O",
            y='Expected:Q',
            tooltip=list(plt_df.columns)
        )

        # Put them together
        roll_chart = (bars + marks).properties(
            title="Game Roll Count", width=w, height=w * 0.75
        ).configure_title(
            fontSize=32, limit=800, dx=45, dy=-50,
            font="Arial", align="center", anchor="middle"
        ).configure_axisBottom(
            grid=False, labelFontSize=20, titleFontSize=16, labelAngle=0
        ).configure_axisLeft(
            grid=False, labelFontSize=14, titleFontSize=16
        )
        return roll_chart



    def player_roll_chart(self):
        """ """
        # Make Altair bar chart
        plt_df = self.player_count.round(2)
        roll_chart = alt.Chart(plt_df).mark_bar(strokeWidth=0.5,
                                                stroke="black").encode(
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
