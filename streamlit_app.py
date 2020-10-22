from collections import Counter
from scipy.stats import chisquare, chi2_contingency
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


def show_info():
    st.write('## Dataset Description')
    st.write('''
        This dataset comes from [Marvel Wikia]
        (http://marvel.wikia.com/Main_Page) and [DC Wikia]
        (http://dc.wikia.com/wiki/Main_Page). It has over 22,000 comic
        characters.
    ''')
    desc = '''
        - `page_id`: The unique identifier for that characters page within the wikia
        - `name`: The name of the character
        - `urlslug`: The unique url within the wikia that takes you to the character
        - `ID`: The identity status of the character (Secret Identity, Public identity, [on marvel only: No Dual Identity])
        - `ALIGN`: If the character is Good, Bad or Neutral
        - `EYE`: Eye color of the character
        - `HAIR`: Hair color of the character
        - `SEX`: Sex of the character (e.g. Male, Female, etc.)
        - `GSM`: If the character is a gender or sexual minority (e.g. Homosexual characters, bisexual characters)
        - `ALIVE`: If the character is alive or deceased
        - `APPEARANCES`: The number of appareances of the character in comic books (as of Sep. 2, 2014. Number will become increasingly out of date as time goes on.)
        - `FIRST APPEARANCE`: The month and year of the character's first appearance in a comic book, if available
        - `YEAR`: The year of the character's first appearance in a comic book, if available
    '''.splitlines()
    if st.checkbox('Show dataset fields description'):
        for d in desc:
            st.write(d)


def show_raw_data(dc, marvel):
    show_info()
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


def filter_year(data, key="Year range", call_fn=st.sidebar.slider):
    min_year, max_year = int(data['YEAR'].min()), int(data['YEAR'].max())
    year = call_fn(
        key,
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    data = data[data['YEAR'] >= year[0]]
    data = data[data['YEAR'] <= year[1]]
    return data


def show_most_appear_name(data):
    st.write('## The most popular character')
    desc = st.empty()
    plot = st.empty()
    col = st.sidebar.beta_columns(2)
    choice = {}
    layout_id = [0, 1, 0, 1, 0, 1]
    layout_index = ['Align', 'ID', 'Eye', 'Hair', 'Sex', 'GSM']
    for col_id, index in zip(layout_id, layout_index):
        with col[col_id]:
            choice[index] = st.selectbox(
                index, ["ALL"] + list(set(data[index.upper()])))
    col1, col2 = st.sidebar.beta_columns(2)
    with col1:
        threshold = st.slider(
            "Appearance threshold", 0, int(data['APPEARANCES'].max()) // 2, 50)
    with col2:
        dataset = st.multiselect(
            "In which company", ["DC", "Marvel"], ["DC"])
    if len(dataset) == 0:
        plot.write('At least one dataset need to be selected.')
        return
    elif len(dataset) == 1:
        data = data[data['TYPE'] == dataset[0]]
    data = filter_year(data, "Year range")
    data = data[data['APPEARANCES'] >= threshold]
    desc_str = []
    for key, value in choice.items():
        if not isinstance(value, str) and np.isnan(value):
            data = data[data[key.upper()].isnull()]
        elif value != 'ALL':
            data = data[data[key.upper()] == value]
            desc_str.append(f'`{value}`')
    if desc_str:
        desc_str = ', '.join(desc_str)
        desc.write(
            f'Who is the super star with {desc_str}?')
    else:
        desc.write('Who is the super star?')
    freq = {k.replace(r'\"', ''): v for k, v in zip(
        data['name'], data['APPEARANCES'])}
    if len(freq) > 0:
        wc = WordCloud(
            background_color="white",
            width=MAX_WIDTH * 2,
            height=400,
        )
        plot.image(wc.generate_from_frequencies(freq).to_image(),
                   use_column_width=True)
    else:
        plot.write('No such person :(')


def show_company(data):
    st.write('## Appearance vs Company')
    st.write('Which is the most popular company, DC or Marvel?')
    data = data.dropna(subset=['APPEARANCES'])
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['YEAR'], empty='none')
    line = alt.Chart(data).mark_line().encode(
        x=alt.X('YEAR', axis=alt.Axis(title='Year')),
        y=alt.Y(
            'sum(APPEARANCES)',
            axis=alt.Axis(title='Sum of appearance of all characteristic')),
        color='TYPE',
        tooltip=['TYPE', 'YEAR', 'sum(APPEARANCES)'],
    )
    selectors = alt.Chart(data).mark_point().encode(
        x='YEAR',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'sum(APPEARANCES)', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(data).mark_rule(color='gray').encode(
        x='YEAR',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    st.altair_chart(alt.layer(
        line, selectors, points, rules, text
    ).interactive(), use_container_width=True)


def show_character_distribution(data):
    """不管出现次数的 角色数量关于特征的分布"""
    st.write('## Feature proportion')
    desc = st.empty()
    # collect user input
    plot = st.empty()
    col1, col2 = st.sidebar.beta_columns(2)
    with col1:
        align = st.selectbox(
            'Which align ', ["ALL"] + list(set(data['ALIGN'])))
        y = st.selectbox("Target feature",
                         ('EYE', 'HAIR', 'SEX', 'GSM'))
    with col2:
        id = st.selectbox('Which ID ', ['ALL'] + list(set(data['ID'])))
        dataset = st.multiselect("In which company ",
                                 ["DC", "Marvel"], ["DC"])
    # process data
    if len(dataset) == 0:
        plot.write('At least one dataset need to be selected.')
        return
    elif len(dataset) == 1:
        data = data[data['TYPE'] == dataset[0]]
    data = data.dropna(subset=[y])
    desc_str = f'What\'s the proportion of {y.lower()}'
    if align != 'ALL' or id != 'ALL':
        desc_str += ' with '
        desc_list = []
        if align != 'ALL':
            if isinstance(align, str):
                desc_list.append(f'`{align}`')
                data = data[data['ALIGN'] == align]
            else:
                data = data[data['ALIGN'].isnull()]
        if id != 'ALL':
            if isinstance(id, str):
                desc_list.append(f'`{id}`')
                data = data[data['ID'] == id]
            else:
                data = data[data['ID'].isnull()]
        desc_str += 'and '.join(desc_list)
    desc.write(desc_str + '?')
    brush = alt.selection_interval(encodings=['x'])
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('YEAR', axis=alt.Axis(title='Year')),
        y=alt.Y(
            'count(count)',
            axis=alt.Axis(title="Count of different " + y.lower())
        ),
        color=y,
        tooltip=[y, 'count(count)'],
    ).add_selection(brush)
    bar = alt.Chart(data).mark_bar().encode(
        x=alt.X(
            'count(count)',
            axis=alt.Axis(title="Count of different " + y.lower())
        ),
        y=y,
    ).transform_filter(brush)
    plot.altair_chart(chart & bar, use_container_width=True)


def show_combination(data):
    """以出现次数作为weights 看对于每种align来说比较常见的feature组合"""
    st.write('## Most common combination of features')
    desc = st.empty()
    # collect user input
    plot2 = st.empty()
    plot = st.empty()
    col1, col2 = st.sidebar.beta_columns(2)
    with col1:
        align = st.selectbox("Which align", ['ALL'] + list(set(data['ALIGN'])))
    with col2:
        id = st.selectbox("Which id", ['ALL'] + list(set(data['ID'])))
    y = st.sidebar.multiselect("Target feature",
                               ('EYE', 'HAIR', 'SEX', 'GSM'), ['EYE'])
    data = filter_year(data, 'Year range')
    # process data
    data = data.dropna(subset=y + ['APPEARANCES'])
    data['POPULARITY'] = np.log(data['APPEARANCES'] + 1)
    y_ = [s.lower() for s in y]
    desc_str = f'What\'s the most common type of {"/".join(y_)}'
    desc_list = []
    if align != 'ALL':
        if isinstance(align, str):
            data = data[data['ALIGN'] == align]
            desc_list.append(f'`{align}`')
        else:  # nan
            data = data[data['ALIGN'].isnull()]
    if id != 'ALL':
        if isinstance(id, str):
            data = data[data['ID'] == id]
            desc_list.append(f'`{id}`')
        else:  # nan
            data = data[data['ID'].isnull()]
    if desc_list:
        desc_str += ' with ' + ' and '.join(desc_list)
    desc_str += '?'
    desc.write(desc_str)
    try:
        data['FEATURE'] = data[y].apply(
            lambda row: ', '.join(row.values), axis=1)
        d = data.groupby('FEATURE').agg({'POPULARITY': 'sum'})
        freq_dict = {
            k: v for k, v in zip(d.index, d['POPULARITY'])
        }
        wc = WordCloud(background_color="white", width=MAX_WIDTH * 2)
        plot2.image(wc.generate_from_frequencies(freq_dict).to_image(),
                    use_column_width=True)
        plot.altair_chart(alt.Chart(data).mark_bar().encode(
            x=alt.X('sum(POPULARITY)', axis=alt.Axis(title='Popularity')),
            y=alt.Y('FEATURE', sort='-x', axis=alt.Axis(title='Feature')),
            color='TYPE',
            tooltip=['TYPE', 'FEATURE', 'sum(POPULARITY)'],
        ).interactive(), use_container_width=True)
    except Exception:
        plot.write('No such combination :(')


def show_heatmap(data):
    st.write('## Relationship of features')
    st.write('What\'s the correlation between different set of features?')
    # https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
    col1, col2 = st.beta_columns(2)
    plot = []
    with col1:
        plot.append(st.empty())
        st.write('<center>DC</center>', unsafe_allow_html=True)
    with col2:
        plot.append(st.empty())
        st.write('<center>Marvel</center>', unsafe_allow_html=True)
    orig_data = filter_year(data, "Year range  ")
    for i, dataset in enumerate(["DC", "Marvel"]):
        x = ['ALIGN', 'ID']
        y = ['EYE', 'HAIR', 'SEX', 'GSM']
        data = orig_data[orig_data['TYPE'] == dataset]

        # treat NaNs in `GSM` as the majority group
        data.replace({"GSM": {np.nan: "N/A"}}, inplace=True)

        keys = data.dropna(subset=y + x)[y + x].columns.values
        factors_paired = [(i, j) for i in y + x for j in y + x]

        cramer = []
        for f in factors_paired:
            features_in_interest = list(set(x) | set([f[0]]) | set([f[1]]))
            result = data.dropna(subset=features_in_interest)[
                features_in_interest]
            confusion_matrix = pd.crosstab(result[f[0]], result[f[1]])
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            cramer.append(np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))))
        cramer_arr = np.array(cramer).reshape(
            (len(x) + len(y), len(x) + len(y)))  # shape it as a matrix
        intra_corr = pd.DataFrame(cramer_arr, index=keys, columns=keys).loc[
            y, y].copy().reset_index().melt('index')
        inter_corr = pd.DataFrame(cramer_arr, index=keys, columns=keys)[
            x].copy().reset_index().melt('index')
        intra_corr.columns = ['var1', 'var2', 'correlation']
        inter_corr.columns = ['var1', 'var2', 'correlation']
        chart_intra = alt.Chart(intra_corr).mark_rect().encode(
            x=alt.X('var2', title=None),
            y=alt.Y('var1', title=None),
            color=alt.Color('correlation'),
        ).properties(width=MAX_WIDTH // 3, height=150)
        chart_intra += chart_intra.mark_text().encode(
            text=alt.Text('correlation', format='.2f'),
            color=alt.condition(
                'datum.correlation >= 0.5',
                alt.value('white'),
                alt.value('black'),
            )
        )

        chart_inter = alt.Chart(inter_corr).mark_rect().encode(
            x=alt.X('var2', title=None),
            y=alt.Y('var1', title=None),
            color=alt.Color('correlation'),
        ).properties(width=MAX_WIDTH // 3, height=150)
        chart_inter += chart_inter.mark_text().encode(
            text=alt.Text('correlation', format='.2f'),
            color=alt.condition(
                'datum.correlation >= 0.5',
                alt.value('white'),
                alt.value('black'),
            )
        )
        plot[i].altair_chart(chart_intra & chart_inter,
                             use_container_width=True)


def show_prediction(data):
    st.write('## Let\'s make prediction')
    st.write(
        'Can we predict the characteristic with respect to a set of features?')
    # TODO


if __name__ == '__main__':
    st.write('# Comic Dataset Analysis')
    st.markdown('''
        > GitHub project page: https://github.com/CMU-IDS-2020/a3-create-a-new-team

        > Dataset credit: [Kaggle](https://www.kaggle.com/fivethirtyeight/fivethirtyeight-comic-characters-dataset)
    ''')
    data, dc, marvel = load_data()
    function_mapping = {
        'Dataset description': lambda: show_raw_data(dc, marvel),
        'Appearance vs Company': lambda: show_company(data),
        'The most popular character': lambda: show_most_appear_name(data),
        'Feature proportion': lambda: show_character_distribution(data),
        'Most common combination of features': lambda: show_combination(data),
        'Relationship of features': lambda: show_heatmap(data),
        'Let\'s make prediction': lambda: show_prediction(data),
    }
    st.sidebar.write('Choose options 👇🏻 to play with this comic dataset!')
    option = st.sidebar.selectbox(
        "Option", list(function_mapping.keys())
    )
    st.sidebar.markdown('---')
    st.markdown('---')
    function_mapping[option]()

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
