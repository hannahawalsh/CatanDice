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

    # Set up sidebar with widgets and gaps
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.selectbox("Page", ["Dice", "Game Stats"],
                                          index=0)
    st.sidebar.write("---")

    st.sidebar.title("Game Options")
    edit_params_button = st.sidebar.button("Apply Options")

    players_radio = st.sidebar.radio("Number of Players", [3, 4], index=1)
    player1 = st.sidebar.text_input("Player 1", "Player 1")
    player2 = st.sidebar.text_input("Player 2", "Player 2")
    player3 = st.sidebar.text_input("Player 3", "Player 3")
    players = {0: player1, 1: player2, 2: player3}
    p4 = st.sidebar.empty()
    if players_radio == 4:
        player4 = p4.text_input("Player 4", "Player 4")
        players[3] = player4
    st.sidebar.write("---")
    random_rate_slider = st.sidebar.slider("Randomness Parameter",
                                           0.0, 1.0, 0.15)
    convergence_rate_slider = st.sidebar.slider("Convergence Rate",
                                                0.0, 1.0, 0.5)
    convergence_rate_slider = convergence_rate_slider * 250 + 200
    random_turns_slider = st.sidebar.slider("Starting Random Turns",
                                            len(players), len(players)*10, 10)


    # Set up main page placeholders
    number_text = st.empty()
    player_name_text = st.empty()
    stats_spot_1 = st.empty()
    stats_spot_2 = st.empty()
    stats_spot_3 = st.empty()
    # roll_button = st.button("Roll!")
    # undo_button = st.button("Undo Roll")
    # reset_button = st.button("Reset Game")
    # stats_button = st.button("Game Statistics")
    roll_button_holder = st.empty()
    undo_button_holder = st.empty()
    reset_button_holder = st.empty()
    ###
    trial_button = st.button("Roll 500 times")
    ###

    # Get cached variables
    roll_history = get_roll_history()
    player_history = get_player_history()

    ### Main Page
    if page_selection == "Dice":
        roll_button = roll_button_holder.button("Roll!")
        undo_button = undo_button_holder.button("Undo Roll")
        reset_button = reset_button_holder.button("Reset Game")

        # Define button action
        if not roll_history[0]:
            number_text.image(dice_image, use_column_width=True)
            player_name_text.markdown(name_html("Roll to start game!"),
                                      unsafe_allow_html=True)
        if roll_button:
            # Add player to the history
            if not player_history:
                current_player = 0
            else:
                current_player = int((player_history[-1] + 1) % len(players))
            player_name = players[current_player]
            player_history.append(current_player)

            # ROll and add to history
            next_roll = dice_roll(roll_history, random_turns_slider,
                                  random_rate_slider,
                                  convergence_rate_slider)
            roll_history[current_player].append(next_roll)
            # roll_history.append(next_roll)

            # Display the results
            number_text.markdown(number_html(next_roll), unsafe_allow_html=True)
            player_name_text.markdown(name_html(player_name),
                                      unsafe_allow_html=True)

        elif undo_button:
            if roll_history:
                player = player_history[-1]
                player_history.pop()
                roll_history[player].pop()

                prev_roll = roll_history[player][-1]
                player_name = players[player_history[-1]]
                number_text.markdown(number_html(prev_roll),
                                     unsafe_allow_html=True)
                player_name_text.markdown(name_html(player_name),
                                          unsafe_allow_html=True)
            else:
                number_text.image(dice_image, use_column_width=True)
                player_name_text.markdown(name_html("Roll to start game!"),
                                          unsafe_allow_html=True)

        elif reset_button:
            st.caching.clear_cache()
            number_text.image(dice_image, use_column_width=True)
            player_name_text.markdown(name_html("Roll to start game!"),
                                      unsafe_allow_html=True)

        elif trial_button:
            for _ in range(500):
                if not player_history:
                    current_player = 0
                else:
                    current_player = int((player_history[-1] + 1) % len(players))
                player_history.append(current_player)
                next_roll = dice_roll(roll_history, random_turns_slider,
                                      random_rate_slider,
                                      convergence_rate_slider)
                roll_history[current_player].append(next_roll)
            number_text.markdown(f"## {len(roll_history)} Rolls in Total")

        if edit_params_button:
            if player_history:
                current_player = player_history[-1]
                current_roll = roll_history[current_player][-1]
                number_text.markdown(number_html(current_roll),
                                     unsafe_allow_html=True)
                player_name_text.markdown(name_html(players[current_player]),
                                          unsafe_allow_html=True)
            else:
                number_text.image(dice_image, use_column_width=True)
                player_name_text.markdown(name_html("Roll to start game!"),
                                          unsafe_allow_html=True)
        st.write(roll_history)

    elif page_selection == "Game Stats":
        player_name_text.markdown(name_html("Good Game!"),
                                  unsafe_allow_html=True)
        player_names = [players[k] for k in sorted(players)]
        result_fig, roll_stats, turn_count = analyze_results(roll_history,
                                                              player_names)
        spot_1.pyplot(fig=result_fig)
        spot_2.write(roll_stats)
        spot_3.write(turn_count)

@st.cache(allow_output_mutation=True)
def get_roll_history():
    return {i: [] for i in range(4)}

@st.cache(allow_output_mutation=True)
def get_player_history():
    return []

def name_html(name):
    style = ("style='text-align: center; font-size: 4.0em; "
             "font-family: Arial; padding: 0px'")
    return f"<h1 {style}>{name}</h1>"

def number_html(number):
    style = ("style='text-align: center; font-size: 12.0em; "
             "font-family: Arial Black; padding: 0px; "
             f"color: {'red' if number in [6, 8] else 'black'};'")
    return f"<h1 {style}>{number}</h1>"

def normalize_dict(D):
    new = {k: v / sum(D.values()) for k, v in D.items()}
    missing_vals = [x for x in range(2, 13) if x not in new.keys()]
    new = {**new, **{m: 0.0 for m in missing_vals}}
    return {k: new[k] for k in range(2, 13)}

def dice_roll(roll_history, first, random_rate, convergence_rate):
    """
    """
    current_player = max([p for p, r in roll_history.items() if len(r) >=
                          len(roll_history[0])]) if roll_history[0] else 0
    player_rolls = roll_history[current_player]
    all_rolls = [roll for player in roll_history.values() for roll in player]

    def gambler_weights(rate, current_weights):
        """ """
        new = {roll: 2 ** (rate * (fair_weights[roll] - current_weights[roll]))
               for roll in range(2, 13)}
        return normalize_dict(new)

    if len(all_rolls) <= first or np.random.random() <= random_rate:
        roll = np.random.choice(list(fair_weights.keys()),
                                p=list(fair_weights.values()))
    else:
        all_past_frequency = normalize_dict(Counter(all_rolls))
        player_past_frequency = normalize_dict(Counter(player_rolls))

        all_new_weights = gambler_weights(convergence_rate, all_past_frequency)
        player_new_weights = gambler_weights(convergence_rate,
                                             player_past_frequency)
        n = len(all_rolls)
        new_weights = {roll: (prob * n + all_new_weights[roll]) / (n + 1)
                       for roll, prob in player_new_weights.items()}
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
