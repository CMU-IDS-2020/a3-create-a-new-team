from collections import Counter
from datetime import datetime
import altair as alt
import numpy as np
import os
import pandas as pd
import streamlit as st
import time


@st.cache(allow_output_mutation=True)
def load_data(directory="./archive"):
    path_dc = os.path.join(directory, 'dc-wikia-data.csv')
    path_marvel = os.path.join(directory, 'marvel-wikia-data.csv')
    dc = pd.read_csv(path_dc)
    marvel = pd.read_csv(path_marvel)
    dc.loc[dc['EYE'] == 'Auburn Hair', 'HAIR'] = 'Auburn Hair'
    dc.loc[dc['EYE'] == 'Auburn Hair', 'EYE'] = np.nan
    return dc, marvel


def show_raw_data(dc, marvel):
    if st.checkbox("Show raw data"):
        col1, col2 = st.beta_columns([2, 1])
        with col1:
            st.write("DC", dc)
        with col2:
            option = st.selectbox("DC keys", sorted(dc.keys()))
            st.write(Counter(dc[option].tolist()))
        col1, col2 = st.beta_columns([2, 1])
        with col1:
            st.write("Marvel", marvel)
        with col2:
            option = st.selectbox("Marvel keys", sorted(marvel.keys()))
            st.write(Counter(marvel[option].tolist()))


def merge(dc, marvel):
    dc['TYPE'] = 'DC'
    marvel['TYPE'] = 'Marvel'
    data = pd.concat([dc, marvel])
    data['count'] = 1
    return data


def show_most_appear_name(data):
    pass


def show_character_distribution(data):
    st.markdown('---')
    st.text('Brief description: TODO')
    # collect user input
    plot = st.empty()
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        x = st.selectbox("Base feature", ('ALIGN', 'ID'))
    with col2:
        y = st.selectbox("Target feature", ('EYE', 'HAIR', 'SEX', 'GSM'))
    with col3:
        dataset = st.multiselect("Dataset", ["DC", "Marvel"], ["DC"])
    # process data
    if len(dataset) == 0:
        plot.write('At least one dataset need to be selected.')
        return
    elif len(dataset) == 1:
        data = data[data['TYPE'] == dataset[0]]
    data = data.dropna(subset=[y])
    year = st.slider(
        "Year range",
        min_value=data['Year'].min(),
        max_value=data['Year'].max(),
        value=(data['Year'].min(), data['Year'].max())
    )
    data = data[data['Year'] >= year[0]]
    data = data[data['Year'] <= year[1]]
    plot.write(alt.Chart(data).mark_bar().encode(
        x='count(count)',
        y=x,
        color=y,
        tooltip=y,
    ))


def show_heatmap(data):
    pass


if __name__ == '__main__':
    st.title('Characteristic in DC/Marvel')
    dc, marvel = load_data()
    show_raw_data(dc, marvel)
    data = merge(dc, marvel)
    # 第一个bubble是visualize符合某一特征的角主要角色（的名字）
    show_most_appear_name(data)
    # 第二个是不管出现次数的 是看角色数量关于特征的分布
    show_character_distribution(data)
    # 第三个是以出现次数作为weights 看对于每种align来说比较常见的feature组合

    #
    show_heatmap(data)

# for reference below
# progress_bar = st.progress(0)
# status_text = st.empty()
# chart = st.line_chart(np.random.randn(10, 2))
# df = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
# st.dataframe(df.style.highlight_max(axis=0))
# status_text.text('Done!')
# option = st.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone'))
# st.write('You selected:', option)
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
