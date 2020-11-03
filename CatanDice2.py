### Imports
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors

from DiceFunctions import StreamlitStyle as SS, Dice


def main():
    """ """
    ### Set up sidebar
    st.sidebar.title("Game Options")
    players_radio = st.sidebar.radio("Number of Players", [3, 4], index=1)
    player1 = st.sidebar.text_input("Player 1", "Player 1")
    player2 = st.sidebar.text_input("Player 2", "Player 2")
    player3 = st.sidebar.text_input("Player 3", "Player 3")
    players = {0: player1, 1: player2, 2: player3}
    if players_radio == 4:
        player4 = st.sidebar.text_input("Player 4", "Player 4")
        players[3] = player4
    random_rate_slider = st.sidebar.slider("Randomness Parameter", 0., 1., 0.15)
    convergence_rate_slider = st.sidebar.slider("Convergence Rate", 0., 1., 0.5)
    convergence_rate_slider = convergence_rate_slider * 250 + 200
    random_turns_slider = st.sidebar.slider("Starting Random Turns", 1, 32, 8)


    ### Set up main page
    title_text = ("<h1 style='text-align: center; font-size: 4.0em; "
                   "color: gold; background-color: maroon; "
                   "font-family: Georgia;'> CATAN DICE (2) </h1>")
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
    trial_button = b2.button("**Roll 50 times**")
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
        next_roll = Dice().new_roll(roll_history, list(players.values()),
                                    players[current_player],
                                    random_turns_slider,
                                    random_rate_slider, convergence_rate_slider)
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
        n = 50
        if not player_history:
            current_player = 0
        else:
            current_player = int((player_history[-1] + 1) % len(players))
        next_rolls = [Dice().new_roll(roll_history, list(players.values()),
                      players[current_player], random_turns_slider,
                      random_rate_slider, convergence_rate_slider) for _ in
                      range(n)]
        roll_history.extend(next_rolls)
        next_p = (player_history[-1] + 1) % 4 if player_history else 0
        player_history.extend([x % 4 for x in range(next_p, n + next_p)])
    ###


    ### Statistics section
    update_stats = stats_cont.button("Get Current Statistics")
    # if "updated_turn" in stats_history:
    #     stats_history["updated_turn"] += 1
    # else:
    #     stats_history["updated_turn"] = 0

    if update_stats:
        player_names = [players[k] for k in sorted(players)]
        fig, stats, turns = Dice().game_stats(roll_history, player_names)
        stats_history["updated_turn"] = 0
        stats_history["fig"] = fig
        stats_history["stats"] = stats
        stats_history["turns"] = turns
        stats_cont.markdown("## Current Stats:")
    elif "stats" in stats_history:
        stats_cont.markdown(f"## Stats from {stats_history['updated_turn']} "
                            "turns ago:")

    if "stats" in stats_history:
        stats_cont.write(stats_history["fig"])
        stats_cont.table(stats_history["stats"])
        stats_cont.table(stats_history["turns"])

    ### Display name and number (or starting text and image)
    if not roll_history:
        number_text.image(dice_image, use_column_width=True)
        player_name_text.markdown(SS.get_name_text("Roll to start game!"),
                                  unsafe_allow_html=True)
    else:
        number_text.markdown(SS.get_number_text(roll_history[-1]),
                             unsafe_allow_html=True)
        player_name = players[player_history[-1]]
        player_name_text.markdown(SS.get_name_text(player_name),
                                  unsafe_allow_html=True)


    stats_cont.markdown(roll_history)


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
    dice_image = Image.open("DicePic.png")
    st.set_page_config(page_title="Gambler's Fallacy Dice",
                       page_icon="🎲", layout="centered")
    main()


### Future possible features:
# Freeze statistics and load on button (not blank spac)
# Players get colors: change player's names and plot color
