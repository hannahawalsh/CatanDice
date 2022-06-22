### Imports
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from matplotlib import colors

from DiceFunctions import StreamlitStyle as SS



def main():
    """ Home Page of Catan dice app. """
    ### Set up session state
    state = st.session_state
    if "roll_history" not in state:
        state.roll_history = []
    if "player_history" not in state:
        state.player_history = []
    if "players" not in state:
        state.players = {}
    if "player_colors" not in state:
        state.player_colors = {}
    if "num_players" not in state:
        state.num_players = 4
    if "game_was_setup" not in state:
        state.game_was_setup = False

    ### Set up main page
    title_text = ("<h1 style='text-align: center; font-size: 5.0em; "
                   F"color: {SS.CatanGold}; background-color: {SS.CatanRed}; "
                   "font-family: Georgia;'> CATAN DICE </h1>")
    st.markdown(title_text, unsafe_allow_html=True)

    _, dice_spot, _ = st.columns([1, 4, 1])
    dice_spot.image(state.dice_image, use_column_width=True)
    

    ### Reset game option
    def new_game():
        for key in st.session_state.keys():
            del st.session_state[key]
    if state.roll_history:
        st.markdown("#")
        _, b, _ = st.columns([0.45, 0.15, 0.4])
        b.button("New Game", on_click=new_game)

    ### Game options
    st.markdown("##")
    st.header("Game Options")

    with st.form("setup"):
        # Layout
        col1, col2a = st.columns(2)
        col1a, col1b, col2b = st.columns([7, 4, 11])

        # Players
        min_players = 3
        max_players = 6
        col1.subheader("Players")
        state.num_players = col1.selectbox("Number of Players",
            range(min_players, max_players+1), index=1)
        for i in range(max_players):
            if state.num_players > i:
                player = col1a.text_input(f"Player {i+1}", f"Player {i+1}")
                color = col1b.selectbox("", SS.color_names, index=i, key=f"c{i+1}",)
                state.players[i] = player
                state.player_colors[i] = SS.player_colors[color]

        # Dice parameters
        col2a.subheader("Game Parameters")
        state.random_turns_slider = col2a.number_input("Starting Turns",
            min_value=state.num_players, value=state.num_players * 2)
        state.convergence_rate_slider = col2b.slider(
            "Convergence Rate", 0.0, 1.0, 0.75)
        state.player_rate_slider = col2b.slider("Player Weight", 0., 1., 0.75)
        state.random_rate_slider = col2b.slider("Randomness Parameter", 0., 1., 0.1)


        if st.form_submit_button("OK"):
            # Indicate that defaults were changed
            st.success("Options Set")
            state.game_was_setup = True


if __name__ == "__main__":
    st.set_page_config(page_title="Gambler's Fallacy Dice", page_icon="🎲",
                       layout="centered")
    if "dice_image" not in st.session_state:
        st.session_state.dice_image = Image.open("DicePic.png")
    main()
