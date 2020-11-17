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
    """ """
    ### Set up sidebar
    st.sidebar.title("Game Options")
    players_radio = st.sidebar.radio("Number of Players", [3, 4], index=1)

    pl1 = st.sidebar.beta_container()
    p1c1, _, p1c2 = pl1.beta_columns([7, 1, 2])
    player1 = p1c1.text_input("Player 1", "Player 1")
    color1 = p1c2.color_picker(f"{player1}'s Color", value="#D70404", key="c1")

    pl2 = st.sidebar.beta_container()
    p2c1, _, p2c2 = pl2.beta_columns([7, 1, 2])
    player2 = p2c1.text_input("Player 2", "Player 2")
    color2 = p2c2.color_picker(f"{player2}'s Color", value="#0434E5", key="c2")

    pl3 = st.sidebar.beta_container()
    p3c1, _, p3c2 = pl2.beta_columns([7, 1, 2])
    player3 = p3c1.text_input("Player 3", "Player 3")
    color3 = p3c2.color_picker(f"{player3}'s Color", value="#F76E02", key="c3")

    players = {0: player1, 1: player2, 2: player3}
    player_colors = {0: color1, 1: color2, 2: color3}
    if players_radio == 4:
        pl4 = st.sidebar.beta_container()
        p4c1, _, p4c2 = pl2.beta_columns([7, 1, 2])
        player4 = p4c1.text_input("Player 4", "Player 4")
        color4 = p4c2.color_picker(f"{player4}'s Color", value="#FFFFFF",
                                   key="c4")
        players[3] = player4
        player_colors[3] = color4

    convergence_rate_slider = st.sidebar.slider("Convergence Rate", 0., 1., 0.5)
    player_rate_slider = st.sidebar.slider("Player Weight", 0., 1., 0.75)
    random_rate_slider = st.sidebar.slider("Randomness Parameter", 0., 1., 0.15)
    random_turns_slider = st.sidebar.number_input("Starting Turns",
                                                  players_radio, value=8)


    ### Set up main page
    title_text = ("<h1 style='text-align: center; font-size: 4.0em; "
                   "color: gold; background-color: maroon; "
                   "font-family: Georgia;'> CATAN DICE (experimental) </h1>")
    st.markdown(title_text, unsafe_allow_html=True)
    number_text = st.empty()
    player_name_text = st.empty()
    buttons = st.beta_container()
    stats_cont = st.beta_expander("Game Statistics", False)

    b1, b2, b3 = buttons.beta_columns(3)
    reset_button = b1.button("Reset")
    roll_button = b2.button("Roll!")
    undo_button = b3.button("Undo")
    ###
    n_trials = 100
    trial_button = b2.button(f"**Roll {n_trials} times**")
    ###

    ### Get cached variables
    roll_history = get_roll_history()
    player_history = get_player_history()
    stats_history = get_statistics_history()


    ### Actions
    if roll_button:
        # Update the player
        if not player_history:
            current_player = 0
        else:
            current_player = int((player_history[-1] + 1) % len(players))
        player_name = players[current_player]
        player_history.append(current_player)

        # Roll the dice
        next_roll = Dice().roll_balanced_2(roll_history.copy(), players_radio,
                                           random_turns_slider,
                                           random_rate_slider,
                                           convergence_rate_slider,
                                           player_rate_slider)
        roll_history.append(next_roll)



    # "Undo" removes last turn from history
    elif undo_button:
        roll_history.pop()
        player_history.pop()

    # "Reset" clears the cache and the history
    elif reset_button:
        st.caching.clear_cache()
        roll_history = player_history = None

    ### Temporary:
    # roll a bunch of times
    elif trial_button:
        for _ in range(n_trials):
            if not player_history:
                current_player = 0
            else:
                current_player = int((player_history[-1] + 1) % len(players))

            R  = Dice().roll_balance_7s(roll_history.copy(), players_radio,
                                        random_turns_slider, random_rate_slider,
                                        convergence_rate_slider,
                                        player_rate_slider)
            roll_history.append(R)
            player_history.append(current_player)



    ### Display name and number (or starting text and image)
    if not roll_history:
        number_text.image(dice_image, use_column_width=True)
        player_name_text.markdown(SS.get_name_text("Roll to start game!"),
                                  unsafe_allow_html=True)
    else:
        number_text.markdown(SS.get_number_text(roll_history[-1]),
                             unsafe_allow_html=True)
        player_name = players[player_history[-1]]
        player_color = player_colors[player_history[-1]]
        player_name_text.markdown(SS.get_name_text(player_name, player_color),
                                  unsafe_allow_html=True)

        ### Game Statistics
        player_names = [players[k] for k in sorted(players)]
        plotter = PlotResults(roll_history, player_names, player_colors)


        stats_cont.markdown("<h2 style='text-align: center; font-size: 1.5em;"
                            "font-family: Arial;'> Turn Count </h2>",
                            unsafe_allow_html=True)
        stats_cont.table(plotter.get_turn_count())
        div_cht, roll_cnt = plotter.get_divergence_chart()
        stats_cont.altair_chart(div_cht, use_container_width=True)
        stats_cont.table(roll_cnt)
        stats_cont.altair_chart(plotter.player_diff_chart())
        stats_cont.altair_chart(plotter.player_roll_chart())
        stats_cont.altair_chart(plotter.all_roll_chart())



### Cached Functions
@st.cache(allow_output_mutation=True)
def get_roll_history():
    return []

@st.cache(allow_output_mutation=True)
def get_player_history():
    return []

@st.cache(allow_output_mutation=True)
def get_statistics_history():
    return {}




if __name__ == "__main__":
    st.set_page_config(page_title="Gambler's Fallacy Dice", page_icon="🎲",
                       layout="centered")
    dice_image = Image.open("DicePic.png")
    main()
