import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import time
import os


@st.cache
def load_data(directory="./archive"):
    path_dc = os.path.join(directory, 'dc-wikia-data.csv')
    path_marvel = os.path.join(directory, 'marvel-wikia-data.csv')
    return pd.read_csv(path_dc), pd.read_csv(path_marvel)


def show_raw_data(dc, marvel):
    if st.checkbox("Show raw data"):
        st.write("DC", dc)
        st.write("Marvel", marvel)
        if st.checkbox("debug"):
            st.write('dc align:', str(dc['ALIGN'].drop_duplicates().tolist()))
            st.write('dc id:', str(dc['ID'].drop_duplicates().tolist()))
            st.write('dc char:', str(dc['']))
            st.write('marvel align:', str(
                marvel['ALIGN'].drop_duplicates().tolist()))
            st.write('marvel id:', str(
                marvel['ID'].drop_duplicates().tolist()))


if __name__ == '__main__':
    st.title('Characteristic in DC/Marvel')
    dc, marvel = load_data()
    show_raw_data(dc, marvel)


# for reference below
# progress_bar = st.progress(0)
# status_text = st.empty()
# chart = st.line_chart(np.random.randn(10, 2))
# df = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
# st.dataframe(df.style.highlight_max(axis=0))
# status_text.text('Done!')
# options = st.multiselect(
#     'What are your favorite colors',
#     ['Green', 'Yellow', 'Red', 'Blue'],
#     ['Yellow', 'Red'])
# st.write('You selected:', options)
# st.write(type(options))
# genre = st.radio(
#     "What's your favorite movie genre",
#     ('Comedy', 'Drama', 'Documentary'))
# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn't select comedy.")

# age = st.slider('How old are you?', 0, 130, 25)
# st.write("I'm ", age, 'years old')


# values = st.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0))
# st.write('Values:', values)
# from datetime import time
# appointment = st.slider(
#     "Schedule your appointment:",
#     value=(time(11, 30), time(12, 45)))
# st.write("You're scheduled for:", appointment)
# from datetime import datetime
# start_time = st.slider(
#     "When do you start?",
#     value=datetime(2020, 1, 1, 9, 30),
#     format="MM/DD/YY - hh:mm")
# st.write("Start time:", start_time)

# color = st.select_slider(
#     'Select a color of the rainbow',
#     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
# st.write('My favorite color is', color)
# start_color, end_color = st.select_slider(
#     'Select a range of color wavelength',
#     options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
#     value=('red', 'blue'))
# st.write('You selected wavelengths between', start_color, 'and', end_color)

# title = st.text_input('Movie title', 'Life of Brian')
# st.write('The current movie title is', title)
# txt = st.text_area('Text to analyze', '''
#     It was the best of times, it was the worst of times, it was
#     the age of wisdom, it was the age of foolishness, it was
#     the epoch of belief, it was the epoch of incredulity, it
#     was the season of Light, it was the season of Darkness, it
#     was the spring of hope, it was the winter of despair, (...)
#     ''')
# st.write('Sentiment:', txt)


# def get_user_name():
#     return 'John'


# with st.echo():
#     # Everything inside this block will be both printed to the screen
#     # and executed.

#     def get_punctuation():
#         return '!!!'

#     greeting = "Hi there, "
#     value = get_user_name()
#     punctuation = get_punctuation()

#     st.write(greeting, value, punctuation)

# # And now we're back to _not_ printing to the screen
# foo = 'bar'
# st.write('Done!')
# import time
# with st.spinner('Wait for it...'):
#     time.sleep(5)
# st.success('Done!')

# with st.empty():
#     for seconds in range(60):
#         st.write(f"⏳ {seconds} seconds have passed")
#         st.write(f"⏳ {seconds} seconds have passed")
#         time.sleep(1)
#     st.write("✔️ 1 minute over!")
