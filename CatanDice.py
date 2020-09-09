### Imports
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.beta_set_page_config(page_title="Gambler's Dice")


def main():
    # Page Title
    title_style = ("style='text-align: center; font-size: 4.0em; "
                   "color: gold; background-color: maroon; "
                   "font-family: Georgia;'")
    st.markdown(f"<h1 {title_style}> CATAN DICE </h1>",
                unsafe_allow_html=True)

    # Get cached variables
    roll_history = get_roll_history()
    player_history = get_player_history()

    # Set up sidebar layout
    st.sidebar.title("Game Options")
    apply_placeholder = st.sidebar.empty()
    radio = st.sidebar.empty()
    p1 = st.sidebar.empty()
    p2 = st.sidebar.empty()
    p3 = st.sidebar.empty()
    p4 = st.sidebar.empty()
    rand_rate = st.sidebar.empty()
    conv_rate = st.sidebar.empty()
    rand_turns = st.sidebar.empty()

    # Set up main page layout and buttons
    number_text = st.empty()
    player_name_text = st.empty()
    spot_1 = st.empty()
    spot_2 = st.empty()
    spot_3 = st.empty()
    roll_button = st.button("Roll!")
    undo_button = st.button("Undo Roll")
    reset_button = st.button("Reset Game")
    stats_button = st.button("Game Statistics")
    ###
    # trial_button = st.button("Roll 100 times")
    ###

    # Add in widgets
    players_radio = radio.radio("Number of Players", [3, 4],
                                        index=1, format_func=int)
    player1 = p1.text_input("Player 1", "Player 1")
    player2 = p2.text_input("Player 2", "Player 2")
    player3 = p3.text_input("Player 3", "Player 3")
    players = {0: player1, 1: player2, 2: player3}
    if players_radio == 4:
        player4 = p4.text_input("Player 4", "Player 4")
        players[3] = player4
    random_rate_slider = rand_rate.slider("Randomness Parameter",
                                            0.0, 1.0, 0.15)
    convergence_rate_slider = conv_rate.slider("Convergence Rate",
                                                0.0, 1.0, 0.5)
    convergence_rate_slider = convergence_rate_slider * 250 + 200
    random_turns_slider = rand_turns.slider("Starting Random Turns", 1, 32, 8)


    # define styles
    name_style = ("style='text-align: center; font-size: 4.0em; "
                  "font-family: Arial; padding: 0px'")
    num_style = lambda n: ("style='text-align: center; font-size: 12.0em; "
                           "font-family: Arial Black; padding: 0px; "
                           f"color: {'red' if n in [6, 8] else 'black'};'")

    # Do action based on button pressed
    if not roll_history:
        number_text.image(dice_image, use_column_width=True)
        player_name_text.markdown(f"<h1 {name_style}> Roll to start game! "
                                  "</h1>", unsafe_allow_html=True)
    edit_params_button = apply_placeholder.button("Apply Changes")

    if roll_button:
        # ROll and add to history
        next_roll = dice_roll(roll_history, random_turns_slider,
                              random_rate_slider, convergence_rate_slider)
        roll_history.append(next_roll)

        # Add player to the history
        if not player_history:
            current_player = 0
        else:
            current_player = int((player_history[-1] + 1) % len(players))
        player_name = players[current_player]
        player_history.append(current_player)

        # Display the results
        number_text.markdown(f"<h1 {num_style(next_roll)}> {next_roll} "
                             "</h1>", unsafe_allow_html=True)
        player_name_text.markdown(f"<h1 {name_style}> {player_name} </h1>",
                                  unsafe_allow_html=True)

    elif undo_button:
        roll_history.pop()
        player_history.pop()

        # Display the results
        if roll_history:
            prev_roll = roll_history[-1]
            prev_player = players[player_history[-1]]
            number_text.markdown(f"<h1 {num_style(prev_roll)}> {prev_roll} "
                                 "</h1>", unsafe_allow_html=True)
            player_name_text.markdown(f"<h1 {name_style}> {prev_player} </h1>",
                                      unsafe_allow_html=True)
        else:
            number_text.image(dice_image, use_column_width=True)
            player_name_text.markdown(f"<h1 {name_style}> Roll to start game! "
                                      "</h1>", unsafe_allow_html=True)

    elif reset_button:
        st.caching.clear_cache()
        number_text.image(dice_image, use_column_width=True)
        player_name_text.markdown(f"<h1 {name_style}> Roll to start game! "
                                  "</h1>", unsafe_allow_html=True)
        apply_placeholder = apply_placeholder.empty()

    elif edit_params_button:
        if player_history:
            current_player = players[player_history[-1]]
            current_roll = roll_history[-1]
            number_text.markdown(f"<h1 {num_style(current_roll)}> {current_roll} "
                             "</h1>", unsafe_allow_html=True)
            player_name_text.markdown(f"<h1 {name_style}> {current_player} </h1>",
                                      unsafe_allow_html=True)
        else:
            number_text.image(dice_image, use_column_width=True)
            player_name_text.markdown(f"<h1 {name_style}> Roll to start game! "
                                      "</h1>", unsafe_allow_html=True)

    elif stats_button:
        player_name_text.markdown(f"<h1 {name_style}> Good Game! "
                                  "</h1>", unsafe_allow_html=True)
        player_names = [players[k] for k in sorted(players)]
        result_fig, roll_stats, turn_count = analyze_results(roll_history,
                                                              player_names)
        spot_1.pyplot(fig=result_fig)
        spot_2.write(roll_stats)
        spot_3.write(turn_count)

    # elif trial_button:
    #     for _ in range(100):
    #         next_roll = dice_roll(roll_history, random_turns_slider,
    #                               random_rate_slider, convergence_rate_slider)
    #         roll_history.append(next_roll)
    #     number_text.markdown(f"<h1 {num_style(next_roll)}> "
    #                          f"{len(roll_history)} Rolls in Total"
    #                          f"</h1>", unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def get_roll_history():
    return []

@st.cache(allow_output_mutation=True)
def get_player_history():
    return []

def normalize_dict(D):
    new = {k: v / sum(D.values()) for k, v in D.items()}
    missing_vals = [x for x in range(2, 13) if x not in new.keys()]
    new = {**new, **{m: 0.0 for m in missing_vals}}
    return {k: new[k] for k in range(2, 13)}

def dice_roll(roll_history, first, random_rate, convergence_rate):
    """
    """
    def gambler_weights(rate, current_weights):
        """ """
        new = {roll: 2 ** (rate * (fair_weights[roll] - current_weights[roll]))
               for roll in range(2, 13)}
        return normalize_dict(new)

    if len(roll_history) <= first or np.random.random() <= random_rate:
        roll = np.random.choice(list(fair_weights.keys()),
                                p=list(fair_weights.values()))
    else:
        past_frequency = normalize_dict(Counter(roll_history))
        new_weights = gambler_weights(convergence_rate, past_frequency)
        roll = np.random.choice(list(new_weights.keys()),
                                p=list(new_weights.values()))
    return roll


def analyze_results(roll_history, player_names):
    # Get rolls per player
    n_plrs = len(player_names)
    player_rolls = {p: roll_history[i::n_plrs] for i, p in
                    enumerate(player_names)}
    player_freqs = {p: {k: v/len(rs) for k, v in Counter(rs).items()}
                    for p, rs in player_rolls.items()}
    player_freqs = pd.DataFrame(player_freqs).sort_index()
    fair_series = pd.Series(fair_weights, name="Theoretical")
    all_roll_freq = pd.Series(normalize_dict(Counter(roll_history)),
                              name="All")
    frequencies = pd.concat([fair_series, all_roll_freq, player_freqs], axis=1)
    frequencies.reset_index(inplace=True)
    frequencies.rename(columns={"index": "Roll"}, inplace=True)
    plot_x = range(2, 13)
    # Create frequency plot
    freq_plot, freq_ax = plt.subplots(figsize=(20, 10))
    freq_ax.set_xlim(1, 13)
    frequencies.drop(columns=["Theoretical", "All"]).plot(x="Roll",
                     kind="bar", ax=freq_ax,
                      color=["red", "blue", "green", "orange"])
    frequencies["Theoretical"].plot(x="Roll", style="*-", color="Fuchsia",
                                    ax=freq_ax, markersize=25, linewidth=3)
    frequencies["All"].plot(x="Roll", style=".-k", ax=freq_ax,
                            markersize=25, linewidth=3)
    # freq_ax.set_xlim(0, 14)
    freq_ax.set_title("Roll Frequencies By Player", fontsize=36)
    freq_ax.set_xlabel("Roll", fontsize=20)
    freq_ax.legend()

    # Create game statistics
    roll_breakdown = frequencies.copy().fillna(0.0)
    turn_count = pd.DataFrame({**{k: len(v) for k, v in player_rolls.items()},
                               "All": len(roll_history)}, index=["Rolls"])

    return freq_plot, roll_breakdown, turn_count

if __name__ == "__main__":
    fair_weights = {**{i: (i-1)/36 for i in range(2, 8)}, **{i: (13-i)/36 for
                    i in range(8, 13)}}
    dice_image = Image.open("DicePic.png")
    main()
