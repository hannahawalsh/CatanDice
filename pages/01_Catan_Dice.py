### Imports
import streamlit as st
from DiceFunctions import StreamlitStyle as SS, Dice

def undo():
    """ Undo the last roll """
    st.session_state.roll_history.pop()
    st.session_state.player_history.pop()


def roll_dice():
    """ Roll the dice """
    state = st.session_state
    # Update the player
    if not state.player_history:
        current_player = 0
    else:
        current_player = int((state.player_history[-1] + 1) %
                             len(state.players))
    player_name = state.players[current_player]
    state.player_history.append(current_player)

    # Roll the dice
    next_roll = Dice().roll_balanced(
        state.roll_history.copy(),
        state.num_players,
        state.random_turns_slider,
        state.random_rate_slider,
        state.convergence_rate_slider,
        state.player_rate_slider)
    state.roll_history.append(next_roll)



def main():
    ### Set up main page
    title_text = ("<h1 style='text-align: center; font-size: 5.0em; "
                   f"color: {SS.CatanGold}; background-color: {SS.CatanRed}; "
                   "font-family: Georgia;'> CATAN DICE </h1>")
    st.markdown(title_text, unsafe_allow_html=True)
    st.markdown("#")
    state = st.session_state

    _, b1, _ = st.columns([0.45, 0.15, 0.4])
    number_text = st.empty()
    player_name_text = st.empty()
    _, b2, _ = st.columns([0.45, 0.15, 0.4])


    # Display
    roll_text = "  \nROLL\n    "
    undo_text = "  \nUNDO\n    "
    if not state.roll_history:
        number_text = st.image(state.dice_image)
        if not state.game_was_setup:
            text = ("<h2 style='text-align: center;'> "
                    "Please set game options! </h2>")
            st.markdown(text, unsafe_allow_html=True)
        else:
            roll_button = b1.button(roll_text, on_click=roll_dice)
    else:
        # Buttons
        roll_button = b1.button(roll_text, on_click=roll_dice)
        undo_button = b2.button(undo_text, on_click=undo)

        if roll_button or undo_button:
            player_name = state.players[state.player_history[-1]]
            player_color = state.player_colors[state.player_history[-1]]
            number_text.markdown(SS.get_number_text(state.roll_history[-1]),
                                 unsafe_allow_html=True)
            player_name_text.markdown(
                SS.get_name_text(player_name, player_color),
                unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Gambler's Fallacy Dice", page_icon="🎲",
                       layout="centered")
    main()
