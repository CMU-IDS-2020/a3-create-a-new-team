from collections import Counter
from scipy.stats import chisquare
from wordcloud import WordCloud
import altair as alt
import numpy as np
import os
import pandas as pd
import streamlit as st


MAX_WIDTH = 700


@st.cache(allow_output_mutation=True)
def load_data(directory="./archive"):
    path_dc = os.path.join(directory, 'dc-wikia-data.csv')
    path_marvel = os.path.join(directory, 'marvel-wikia-data.csv')
    # the Auburn Hair and Year issues have resolved in the csv file
    dc, marvel = pd.read_csv(path_dc), pd.read_csv(path_marvel)
    return merge(dc, marvel), dc, marvel


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
    marvel.replace({"SEX": {"Genderfluid Characters": "Genderless Characters",
                            "Agender Characters": "Genderless Characters"}},
                   inplace=True)
    dc.replace({"SEX": {"Genderfluid Characters": "Genderless Characters",
                        "Agender Characters": "Genderless Characters"}},
               inplace=True)
    dc.loc[dc['SEX'] == "Transgender Characters",
           "GSM"] = "Transgender Characters"
    dc.loc[dc['SEX'] == "Transgender Characters", "SEX"] = np.nan
    data = pd.concat([dc, marvel])
    data['count'] = 1
    return data


def filter_year(data, key="Year range"):
    min_year, max_year = int(data['YEAR'].min()), int(data['YEAR'].max())
    year = st.slider(
        key,
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    data = data[data['YEAR'] >= year[0]]
    data = data[data['YEAR'] <= year[1]]
    return data


def show_most_appear_name(data):
    """符合某一特征的角主要角色（的名字）"""
    st.markdown('---')
    st.text('Brief description for show_most_appear_name: TODO')
    plot = st.empty()
    col = st.beta_columns(3)
    choice = {}
    layout_id = [0, 0, 1, 1, 2, 2]
    layout_index = ['Align', 'ID', 'Eye', 'Hair', 'Sex', 'GSM']
    for col_id, index in zip(layout_id, layout_index):
        with col[col_id]:
            choice[index] = st.selectbox(
                index, ["ALL"] + list(set(data[index.upper()])))
    col1, col2 = st.beta_columns(2)
    with col1:
        threshold = st.slider(
            "Appearance threshold", 0, int(data['APPEARANCES'].max()) // 2, 50)
    with col2:
        dataset = st.multiselect(
            "Dataset for most appear", ["DC", "Marvel"], ["DC"])
    if len(dataset) == 0:
        plot.write('At least one dataset need to be selected.')
        return
    elif len(dataset) == 1:
        data = data[data['TYPE'] == dataset[0]]
    data = filter_year(data, "Year range for most appear")
    data = data[data['APPEARANCES'] >= threshold]
    for key, value in choice.items():
        if not isinstance(value, str) and np.isnan(value):
            data = data[data[key.upper()].isnull()]
        elif value != 'ALL':
            data = data[data[key.upper()] == value]
    freq = {k.replace(r'\"', ''): v for k, v in zip(
        data['name'], data['APPEARANCES'])}
    if len(freq) > 0:
        wc = WordCloud(background_color="white", width=MAX_WIDTH * 2)
        plot.image(wc.generate_from_frequencies(freq).to_image(),
                   use_column_width=True)
    else:
        plot.write('No such person :(')


def show_character_distribution(data):
    """不管出现次数的 角色数量关于特征的分布"""
    st.markdown('---')
    st.text('Brief description for show_character_distribution: TODO')
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
    data = filter_year(data)
    plot.write(alt.Chart(data).mark_bar().encode(
        x=alt.X(
            'count(count)',
            axis=alt.Axis(title="Count of different " + y.lower())
        ),
        y=x,
        color=y,
        tooltip=y,
    ).properties(width=MAX_WIDTH))


def show_combination(data):
    """以出现次数作为weights 看对于每种align来说比较常见的feature组合"""
    st.markdown('---')
    st.text('Brief description for show_combination: TODO')
    # collect user input
    plot2 = st.empty()
    plot = st.empty()
    col1, col2 = st.beta_columns(2)
    with col1:
        align = st.selectbox("Which ALIGN", ['ALL'] + list(set(data['ALIGN'])))
        y = st.multiselect("Target feature for combination",
                           ('EYE', 'HAIR', 'SEX', 'GSM'), ['EYE'])
    with col2:
        id = st.selectbox("Which ID", ['ALL'] + list(set(data['ID'])))
        dataset = st.multiselect("Dataset for combination",
                                 ["DC", "Marvel"], ["DC"])
    # process data
    if len(dataset) == 0:
        plot.write('At least one dataset need to be selected.')
        return
    elif len(dataset) == 1:
        data = data[data['TYPE'] == dataset[0]]
    data = filter_year(data, 'Year for combination')
    data = data.dropna(subset=y + ['APPEARANCES'])
    data['APPEARANCES'] = np.log(data['APPEARANCES'] + 1)
    if align != 'ALL':
        if isinstance(align, str):
            data = data[data['ALIGN'] == align]
        else:  # nan
            data = data[data['ALIGN'].isnull()]
    if id != 'ALL':
        if isinstance(data, str):
            data = data[data['ID'] == id]
        else:  # nan
            data = data[data['ID'].isnull()]
    try:
        data = data[y + ['APPEARANCES', 'TYPE', 'name']]
        data = data.groupby(y).agg({'APPEARANCES': 'sum'})
        freq_dict = {
            k if isinstance(k, str) else ', '.join(k): v
            for k, v in zip(data.index, data['APPEARANCES'])
        }
        wc = WordCloud(background_color="white", width=MAX_WIDTH * 2)
        plot.image(wc.generate_from_frequencies(freq_dict).to_image(),
                   use_column_width=True)
        data['NAME'] = list(freq_dict.keys())
        plot2.write(alt.Chart(data).mark_bar().encode(
            x='APPEARANCES',
            y='NAME',
            tooltip='NAME',
        ).properties(width=MAX_WIDTH))
    except Exception:
        plot.write('No such combination :(')


def show_heatmap(data):
    st.markdown('---')
    st.text('Brief description: TODO')
    plot = st.empty()
    x = ['ALIGN', 'ID']
    y = ['EYE', 'HAIR', 'SEX', 'GSM']
    dataset = st.multiselect("Dataset for heatmap", ["DC", "Marvel"], ["DC"])
    if len(dataset) == 0:
        plot.write('At least one dataset need to be selected.')
        return
    elif len(dataset) == 1:
        data = data[data['TYPE'] == dataset[0]]
    data = filter_year(data, "Year range for heatmap")
    data = data.apply(lambda x: pd.factorize(x)[0]) + 1
    for target_feature in y:
        result = data.dropna(subset=[target_feature] + x)[[target_feature] + x]
        result
        result = pd.DataFrame(
            [chisquare(result[b].values, f_exp=result.values.T, axis=1)[0]
             for b in result],
            columns=result.keys(),
            index=result.keys())
        st.write(result, "why is it still NaN???")


if __name__ == '__main__':
    st.title('Characteristic in DC/Marvel')
    data, dc, marvel = load_data()
    show_raw_data(dc, marvel)
    show_most_appear_name(data)
    show_character_distribution(data)
    show_combination(data)
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
