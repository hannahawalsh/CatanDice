### Imports
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors

from DiceFunctions import StreamlitStyle as SS, Dice, PlotResults



def main():
    """ This is the main body of the dice app. """
    ### Set up sidebar
    max_players = 6
    st.sidebar.title("Game Options")
    num_players = st.sidebar.selectbox("Number of Players",
                                       range(3, max_players+1), index=1)
    players = {}
    player_colors = {}
    for i in range(max_players):
        if num_players > i:
            plr = st.sidebar.beta_container()
            # col1, _, col2 = plr.beta_columns([7, 1, 2])
            col1, col2 = plr.beta_columns([7,4])
            player = col1.text_input(f"Player {i+1}", f"Player {i+1}")
            color = col2.selectbox("", SS.color_names, index=i,
                                   key=f"c{i+1}",)
            players[i] = player
            player_colors[i] = SS.player_colors[color]

    convergence_rate_slider = st.sidebar.slider("Convergence Rate",
                                                0.0, 1.0, 0.75)
    player_rate_slider = st.sidebar.slider("Player Weight", 0., 1., 0.75)
    random_rate_slider = st.sidebar.slider("Randomness Parameter", 0., 1., 0.15)
    random_turns_slider = st.sidebar.number_input("Starting Turns",
                                                  min_value=num_players,
                                                  value=num_players * 2)


    ### Set up main page
    title_text = ("<h1 style='text-align: center; font-size: 5.0em; "
                   F"color: {SS.CatanGold}; background-color: {SS.CatanRed}; "
                   "font-family: Georgia;'> CATAN DICE </h1>")
    st.markdown(title_text, unsafe_allow_html=True)
    number_text = st.empty()
    _, b1 = st.beta_columns([21, 30])
    roll_button = b1.button("ROLL DICE")
    player_name_text = st.empty()
    _, b2, b3 = st.beta_columns([11, 15, 15])
    reset_button = b2.button("Reset")
    undo_button = b3.button("Undo")
    stats_cont = st.beta_expander("Game Statistics", False)


    ### Use session state
    if "roll_history" not in st.session_state:
        st.session_state.roll_history = []
    if "player_history" not in st.session_state:
        st.session_state.player_history = []


    ### Testing Button:
    n_tests = 200
    testing_button = stats_cont.button(f"Roll {n_tests} Times")
    if testing_button:
        for _ in range(n_tests):
            if not st.session_state.player_history:
                current_player = 0
            else:
                current_player = int((st.session_state.player_history[-1] + 1) % len(players))
            player_name = players[current_player]
            st.session_state.player_history.append(current_player)
            # Roll the dice
            next_roll = Dice().roll_balanced(
                st.session_state.roll_history.copy(),
                num_players,
                random_turns_slider,
                random_rate_slider,
                convergence_rate_slider,
                player_rate_slider)
            st.session_state.roll_history.append(next_roll)

    ### Actions
    if roll_button:
        # Update the player
        if not st.session_state.player_history:
            current_player = 0
        else:
            current_player = int((st.session_state.player_history[-1] + 1) % len(players))
        player_name = players[current_player]
        st.session_state.player_history.append(current_player)

        # Roll the dice
        next_roll = Dice().roll_balanced(
            st.session_state.roll_history.copy(),
            num_players,
            random_turns_slider,
            random_rate_slider,
            convergence_rate_slider,
            player_rate_slider)
        st.session_state.roll_history.append(next_roll)


    # "Undo" removes last turn from history
    elif undo_button:
        st.session_state.roll_history.pop()
        st.session_state.player_history.pop()

    # "Reset" clears the cache and the history
    elif reset_button:
        st.caching.clear_cache()
        st.session_state.roll_history = []
        st.session_state.player_history = []


    ### Display name and number (or starting text and image)
    if not st.session_state.roll_history:
        number_text.image(dice_image, use_column_width=True)
        player_name_text.markdown(SS.get_name_text("Roll to start game!"),
                                  unsafe_allow_html=True)
    else:
        number_text.markdown(
            SS.get_number_text(st.session_state.roll_history[-1]),
            unsafe_allow_html=True)
        player_name = players[st.session_state.player_history[-1]]
        player_color = player_colors[st.session_state.player_history[-1]]
        player_name_text.markdown(SS.get_name_text(player_name, player_color),
                                  unsafe_allow_html=True)

        ### Game Statistics
        if len(st.session_state.roll_history) > 1:
            player_names = [players[k] for k in sorted(players)]
            plotter = PlotResults(st.session_state.roll_history,
                                  player_names, player_colors)


            stats_cont.markdown("<h2 style='text-align: center; "
                                "font-size: 1.5em; font-family: Arial;'>"
                                "Turn Count</h2>",
                                unsafe_allow_html=True)
            stats_cont.table(plotter.get_turn_count())
            stats_cont.altair_chart(plotter.all_roll_chart())
            div_cht, roll_cnt = plotter.get_divergence_chart()
            stats_cont.altair_chart(div_cht, use_container_width=True)
            stats_cont.table(roll_cnt)
            stats_cont.altair_chart(plotter.player_diff_chart())
            stats_cont.altair_chart(plotter.player_roll_chart())




if __name__ == "__main__":
    st.set_page_config(page_title="Gambler's Fallacy Dice", page_icon="🎲",
                       layout="centered")
    dice_image = Image.open("DicePic.png")
    main()
