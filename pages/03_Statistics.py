import streamlit as st
from DiceFunctions import PlotResults, Dice


def show_statistics():
    # Layout
    st.header("Game Options")
    st.markdown("---")
    state = st.session_state

    # Statistics
    if len(st.session_state.roll_history) > 1:
        player_names = [state.players[k] for k in sorted(state.players)]
        plotter = PlotResults(state.roll_history, player_names,
                              state.player_colors)
        div_cht, roll_cnt = plotter.get_divergence_chart()

        st.altair_chart(plotter.all_roll_chart())
        st.table(roll_cnt)
        st.markdown("<h2 style='text-align: center; "
                            "font-size: 1.5em; font-family: Arial;'>"
                            "Turn Count</h2>",
                            unsafe_allow_html=True)
        st.table(plotter.get_turn_count())
        st.altair_chart(div_cht, use_container_width=True)
        st.altair_chart(plotter.player_diff_chart())
        st.altair_chart(plotter.player_roll_chart())

    else:
        st.subheader("No rolls recorded")

    st.button("Test 200 Rolls", on_click=test_lotsa_rolls)


def test_lotsa_rolls(n_tests=200):
    """ Roll a bunch of time for testing purposes """
    state = st.session_state
    for _ in range(n_tests):
        if not state.player_history:
            current_player = 0
        else:
            current_player = int((state.player_history[-1] + 1) % len(state.players))
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



if __name__ == "__main__":
    st.set_page_config(page_title="Gambler's Fallacy Dice", page_icon="🎲",
                       layout="centered")
    show_statistics()
