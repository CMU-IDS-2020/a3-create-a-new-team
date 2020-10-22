from collections import Counter
from scipy.stats import chisquare, chi2_contingency
from wordcloud import WordCloud
import altair as alt
import numpy as np
import os
import pandas as pd
import streamlit as st

from models import calc_feature_importances

MAX_WIDTH = 700


@st.cache(allow_output_mutation=True)
def load_data(directory="./archive"):
    path_dc = os.path.join(directory, 'dc-wikia-data.csv')
    path_marvel = os.path.join(directory, 'marvel-wikia-data.csv')
    # the Auburn Hair and Year issues have resolved in the csv file
    dc, marvel = pd.read_csv(path_dc), pd.read_csv(path_marvel)
    feature_importances = calc_feature_importances()
    dc['WORLD'] = 'DC'
    marvel['WORLD'] = 'Marvel'
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
    data['name'] = data['name'].apply(lambda x: x.replace(r'\"', ''))
    return data, dc, marvel, feature_importances


def show_info():
    st.write('## Dataset description')
    st.write('''
        This dataset comes from [Marvel Wikia]
        (http://marvel.wikia.com/Main_Page) and [DC Wikia]
        (http://dc.wikia.com/wiki/Main_Page). It has over 22,000 comic
        characters.
    ''')
    st.markdown('''
    ## Definitions
    - We group `EYE`, `HAIR`, `SEX`, `GSM` (refer to data field descriptions) as genetic features for the sake of convenience, though we are aware that it is controversial whether sexual orientation is genetic.\
    We believe these genetic features are good indicators of one's inborn cultural identities.
    - We group `ID` and `ALIGN` as acquired identities.
    ## Goals
    - Visualize the distribution of different features of the characters in two comic worlds, DC and Marvel.
    - Analyze the correlations among genetic features.
    - Analyze the correlations between genetic features of the characters and their acquired identities (identity status and alignment).
    ''')
    desc = '''
        #### Fields
        - `page_id`: The unique identifier for that characters page within the wikia
        - `name`: The name of the character
        - `urlslug`: The unique url within the wikia that takes you to the character
        - `ID`: The identity status of the character (Secret Identity, Public identity, [on marvel only: No Dual Identity])
        - `ALIGN`: If the character is Good, Bad or Neutral
        - `EYE`: Eye color of the character
        - `HAIR`: Hair color of the character
        - `SEX`: Sex of the character (e.g. Male, Female, etc.)
        - `GSM`: If the character is a gender or sexual minority (e.g. homosexual characters, bisexual characters)
        - `ALIVE`: If the character is alive or deceased
        - `APPEARANCES`: The number of appareances of the character in comic books (as of Sep. 2, 2014. Number will become increasingly out of date as time goes on.)
        - `FIRST APPEARANCE`: The month and year of the character's first appearance in a comic book, if available
        - `YEAR`: The year of the character's first appearance in a comic book, if available
    '''.splitlines()
    if st.checkbox('Show dataset field descriptions.'):
        for d in desc:
            st.write(d)


def show_desc(dc, marvel):
    show_info()
    if st.checkbox("Show raw data."):
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
    st.write('## The most exposed characters')
    st.markdown('''
    We use the number of appearances as a metric for the level of popularity in a character's world.
    
    The number of appearances is accumulated from the year the character debuted to 2013.
    ''')
    desc = st.empty()
    plot = st.empty()
    plot2 = st.empty()
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
        dataset = st.multiselect(
            "In which world", ["DC", "Marvel"], ["DC", "Marvel"])
    with col2:
        data = filter_year(data, "Year range", st.slider)
    data = data.dropna(subset=['APPEARANCES'])
    if len(dataset) == 0:
        plot.write('At least one dataset need to be selected.')
        return
    elif len(dataset) == 1:
        data = data[data['WORLD'] == dataset[0]]
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
    freq = {k: v for k, v in zip(data['name'], data['APPEARANCES'])}
    if len(freq) > 0:
        wc = WordCloud(
            background_color="white",
            width=MAX_WIDTH * 2,
            height=400,
        )
        plot.image(wc.generate_from_frequencies(freq).to_image(),
                   use_column_width=True)
        data = data.sort_values(by=['APPEARANCES', 'name'], ascending=False)[:20]
        plot2.altair_chart(alt.Chart(data).mark_bar().encode(
            x=alt.X('APPEARANCES', axis=alt.Axis(title='Appearance')),
            y=alt.Y('name', sort='-x', axis=alt.Axis(title=f'Top {len(data)} Big Names')),
            color='WORLD',
            tooltip=['name', 'APPEARANCES'],
        ), use_container_width=True)
    else:
        plot.write('No such person :(')


def show_company(data):
    st.write('## Level of activity of DC/Marvel')
    st.write('Which is the most active world, DC or Marvel?')
    st.write("We estimate the level of activity of a world in a certain year by the sum of appearances of all that world's characters that debuted in that year.")
    data = data.dropna(subset=['APPEARANCES'])
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['YEAR'], empty='none')
    line = alt.Chart(data).mark_line().encode(
        x=alt.X('YEAR', axis=alt.Axis(title='Debut Year')),
        y=alt.Y(
            'sum(APPEARANCES)',
            axis=alt.Axis(title='Sum of Appearances of All Characters')),
        color='WORLD',
        tooltip=['WORLD', 'YEAR', 'sum(APPEARANCES)'],
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
    st.markdown('''
    Overall, the Marvel world is more active than the DC world. And there are peaks for both worlds.
    For example, the level of activity of the Marvel world peaks at around year 1963. We could suspect that there must be influential stories or characters around that period.
    To verify, go on to the next playground and set the year range to `1963-1963` and `world` to `Marvel`.
    ''')


def show_character_distribution(data):
    """不管出现次数的 角色数量关于特征的分布"""
    st.write('## Distribution of genetic features')
    st.write('Genetic features represent inborn cultural identity of a character. Characters are overall more diverse over the years.')
    desc = st.empty()
    # collect user input
    plot = st.empty()
    col1, col2 = st.sidebar.beta_columns(2)
    with col1:
        align = st.selectbox(
            'Which align ', ["ALL"] + list(set(data['ALIGN'])))
        y = st.selectbox("Target genetic feature",
                         ('EYE', 'HAIR', 'SEX', 'GSM'))
    with col2:
        id = st.selectbox('Which ID ', ['ALL'] + list(set(data['ID'])))
        dataset = st.multiselect("In which world ",
                                 ["DC", "Marvel"], ["DC"])
    # process data
    if len(dataset) == 0:
        plot.write('At least one dataset need to be selected.')
        return
    elif len(dataset) == 1:
        data = data[data['WORLD'] == dataset[0]]
    data = data.dropna(subset=[y])
    desc_str = f'What\'s the distribution of {y.lower()}'
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
    desc.write(desc_str + '?\n\nYou can drag the year range in the upper chart. And then the lower bar chart would summarizes the data points in selected year range.')
    brush = alt.selection_interval(encodings=['x'])
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('YEAR', axis=alt.Axis(title='Year')),
        y=alt.Y(
            'count(count)',
            axis=alt.Axis(title=f"Count of Characters with Different {y} Feature")
        ),
        color=y,
        tooltip=[y, 'count(count)'],
    ).add_selection(brush)
    bar = alt.Chart(data).transform_filter(brush).transform_joinaggregate(
        TotalCount='count(*)', ).transform_calculate(
        pct='1 / datum.TotalCount').mark_bar().encode(
        x=alt.X(
            'sum(pct):Q',
            axis=alt.Axis(format='%', title=f"Percentage of Characters with Different {y} Feature")),
        y=y,
        tooltip=[y, alt.Tooltip('sum(pct):Q', format='.1%', title="Percentage")],
    )
    plot.altair_chart(chart & bar, use_container_width=True)
    st.markdown('''
    A number of interesting observations:
    - Earlier characters with blue eyes accounts for the majority, and gradually \
    the numbers of characters with brown eyes and black eyes begin to grow.
    - Set `Target feature` to `SEX`, select different `ALIGN` and drag the year range window from \
    left to right. Earlier worlds are dominated by male characters and the distributions \
    of different sex within different `ALIGN` groups have
    become more evenly distributed throughout the years, except for the `Bad Characters`. Well, \
    at least some of us may be expecting charming bad female characters :) 
    ''')


def show_combination(data):
    """以出现次数作为weights 看对于每种align来说比较常见的feature组合"""
    st.write('## Stereotypes')
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
    desc_str = f'What\'s the most common type of {", ".join(y_)}'
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
    desc_str += '\n\nWe estimate the popularity of a feature combination with a function of both the number of characters that match the combination and their number of appearances.'
    desc_str += '\n\nIn earlier years (e.g. before 1951), blue-eyed blonde-hair male characters dominate the comic world. \
    Later (e.g. after 1972), the world favors brown-eyed black/brown hair male characters and women become more influential, \
    but the most popular female characters are still of blue eyes and blonde hair. \
    From 2000 to 2013, the ethnicities of male and female are roughly of the same distribution with respect to popularity.'
    desc_str += '\n\nSexual and gender minorities other than homosexual and bisexual are not in the comic world until 1984.'

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
        plot.write(alt.Chart(data).mark_bar().encode(
            x=alt.X('sum(POPULARITY)', axis=alt.Axis(title='Popularity')),
            y=alt.Y('FEATURE', sort='-x', axis=alt.Axis(title='Feature')),
            color='WORLD',
            column='WORLD',
            tooltip=['FEATURE', 'sum(POPULARITY)', 'WORLD'],
        ).properties(width=MAX_WIDTH // 3))
    except Exception:
        plot.write('No such combination :(')


def show_heatmap(data):
    st.write('## Relationships of features')
    st.markdown('''
    What\'s the correlation between different features?'
    
    The correlation is calculated by corrected Cramer\'s V, and it is a symmetric metric.
    
    - The upper panel demonstrates intra-correlations (correlations between genetic features), \
    and the lower panel demonstrates inter-correlations (correlation between a genetic feature and an acquired identity,`ID` (or `ALIGN`)).
    - The left panel demonstrates the DC world, and the right panel demonstrates the Marvel world.
    ''')
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
        data = orig_data[orig_data['WORLD'] == dataset]

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
        inter_corr = pd.DataFrame(cramer_arr, index=keys, columns=keys).loc[
            y, x].copy().reset_index().melt('index')
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
    st.markdown('''
    For intra-correlations:
    - The pair-wise correlations among {`HAIR`, `EYE`, `SEX`} are quite high. 
    - The correlation between `HAIR` and `EYE` of the Marvel world is consistently higher than then DC world over the years.
    - `GSM` has little correlation with other genetic features except for `SEX`. One possible reason is that most characters are of sexual and gender majority.
    
    For inter-correlations:
    - Inter-correlations are higher in the Marvel world.
    - There is no strong inter-correlation between a single genetic feature and a single acquired identity (`ALIGN`, or `ID`).
    ''')


def show_prediction(feature_importances):
    st.write('## Let\'s make predictions')
    st.markdown('''
    Can we predict one genetic feature given the other features?
    
    We train a decision tree classifier for each of the response variable from {`SEX`, `EYE`, `HAIR`}, and visualize the Gini importance of the explanatory features, \
    as an interpretation of asymmetric correlations. 
    
    The details of Gini importance are give in [the scikit-learn document](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree#sklearn.tree.DecisionTreeClassifier).
    ''')

    # desc = st.empty()
    # collect user input
    plot = st.empty()
    response = st.sidebar.selectbox(
        'Which response variable', list(feature_importances.keys()))

    data = feature_importances[response]
    plot.altair_chart(alt.Chart(data).mark_bar().encode(
        x=alt.X('Importance',
                axis=alt.Axis(title=f"Importances of Different Explanatory Variables in Predicting {response}")
                ),
        y=alt.Y('Variable', axis=alt.Axis(title='Explanatory Variables'),
                sort='-x'),
        tooltip=['Importance'],
    ).properties(height=500).interactive(), use_container_width=True)
    st.markdown('''
    The most decisive factor in predicting genetic features are the debut year \
    and the number of appearances (popularity) of the character. This is in line with our observations in the previous sections \
    that the distribution of genetic features changes over time and the world favors certain ethnicities.
    
    Although previously we show that genetic features share correlations with each other, \
    the other two genetic features contribute little to the prediction of a certain genetic feature. \
    It is possible that the combination of the other two genetic features is predictive of the target response feature, \
    but we are unable to validate this using the APIs of the decision tree model.
    
    Acquired identities are not predictive of the genetic features as well.
    ''')


if __name__ == '__main__':
    st.write('# Comic Dataset Analysis')
    st.markdown('''
        > GitHub project page: https://github.com/CMU-IDS-2020/a3-create-a-new-team

        > Dataset credit: [Kaggle](https://www.kaggle.com/fivethirtyeight/fivethirtyeight-comic-characters-dataset)
    ''')
    data, dc, marvel, feature_importances = load_data()
    function_mapping = {
        'Project description': lambda: show_desc(dc, marvel),
        'Level of activity of DC/Marval': lambda: show_company(data),
        'The most exposed characters': lambda: show_most_appear_name(data),
        'Distribution of genetic features': lambda: show_character_distribution(data),
        'Stereotypes': lambda: show_combination(data),
        'Relationships between features': lambda: show_heatmap(data),
        'Let\'s make predictions': lambda: show_prediction(feature_importances),
    }
    st.sidebar.write('Choose playgrounds 👇🏻 to explore this comic dataset!')
    option = st.sidebar.selectbox(
        "Playground", list(function_mapping.keys())
    )
    st.sidebar.markdown('---')
    st.markdown('---')
    function_mapping[option]()
