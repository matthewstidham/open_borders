import pandas as pd
import numpy as np
from itertools import combinations
import os


class DataGenerator:
    def __init__(self):
        self.parent_folder = '../data'
        self.test = None
        self.data2 = None

    def generator(self):
        # read csvs
        cpi = pd.read_csv(os.path.join(self.parent_folder, 'cpi.csv'))
        edbi = pd.read_csv(os.path.join(self.parent_folder, 'edbiscores.csv'))
        homicide = pd.read_csv(os.path.join(self.parent_folder, 'homicide.csv'))
        pfi = pd.read_csv(os.path.join(self.parent_folder, 'pfi.csv'))[['Country', 'Score']]
        data = pd.read_csv(os.path.join(self.parent_folder, 'world_indicators_all_years.csv'))
        democracy = pd.read_csv(os.path.join(self.parent_folder, 'democracyindex2019.csv'))
        gini = pd.read_csv(os.path.join(self.parent_folder, 'gini.csv')).iloc[:217].replace('..', np.nan)

        # quality of life indicators
        cpi = cpi[['Country', '2015 cpi']]
        cpi = cpi.replace('-', np.nan)
        cpi['2015 cpi'] = [float(x) for x in cpi['2015 cpi']]

        # ease of doing business
        edbi = edbi.replace('Russian Federation', 'Russia')

        # press freedom
        pfi.columns = ['Country', '2018 pfi']
        pfi['2018 pfi'] = [float(x) for x in pfi['2018 pfi']]

        # homicide
        homicide.columns = ['Country', 'Region', 'Subregion', 'Homicide Rate', 'Homicide County', 'Year', 'Source']

        # quality of life indictors into one dataframe
        qol = pd.merge(cpi, pfi, on='Country', how='outer')
        qol = pd.merge(qol, edbi, on='Country', how='outer')
        qol = pd.merge(qol, homicide, on='Country', how='outer')
        qol['score'] = qol['2015 cpi'] / qol['2018 pfi'] * qol['DB 2019']

        # GDP and Population
        data = data[data['Time'] == '2016']
        data = data.replace('Iran, Islamic Rep.', 'Iran').replace('Korea, Dem. People’s Rep.', "North Korea").replace(
            'Lao PDR', 'Laos').replace('Korea, Rep.', 'South Korea').replace('Macedonia, FYR', 'Macedonia').replace(
            'Myanmar', 'Myanmar (Burma)').replace('Russian Federation', 'Russia').replace('Slovak Republic',
                                                                                          'Slovakia').replace(
            'Syrian Arab Republic', 'Syria').replace('Venezuela, RB', 'Venezuela').replace('Yemen, Rep.', 'Yemen')
        data1 = data[['Country Name', 'Population, total [SP.POP.TOTL]', 'Population growth (annual %) [SP.POP.GROW]',
                      'GDP (current US$) [NY.GDP.MKTP.CD]', 'GDP per capita (current US$) [NY.GDP.PCAP.CD]']]
        data1 = data1.replace('..', np.nan)
        for x in [*data1.columns][1:]:
            data1[x] = [float(i) for i in data1[x]]
        data1['Population, total [SP.POP.TOTL]'] = [float(n) for n in data1['Population, total [SP.POP.TOTL]']]
        data1 = pd.merge(qol, data1, left_on='Country', right_on='Country Name', how='outer')

        # treaties
        schengen = ['Austria', 'Belgium', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',
                    'Greece', 'Hungary', 'Iceland', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg',
                    'Malta', 'Netherlands', 'Monaco', 'Norway', 'Poland', 'Portugal', 'San Marino', 'Slovakia',
                    'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Vatican City', 'Ireland', 'United Kingdom']
        unionstate = ['Russia', 'Belarus']
        peaceandfriendship = ['India', 'Bhutan', 'Nepal']
        ca4 = ['El Salvador', 'Honduras', 'Guatemala', 'Nicaragua']
        tasman = ['Australia', 'New Zealand']
        andean = ['Bolivia', 'Ecuador', 'Colombia', 'Peru']
        caricom = ['Antigua and Barbuda', 'Barbados', 'Belize', 'Dominica', 'Grenada', 'Guyana', 'Jamaica',
                   'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Suriname',
                   'Trinidad and Tobago']
        gcc = ['Saudi Arabia', 'Oman', 'Kuwait', 'United Arab Emirates', 'Bahrain']
        eac = ['Kenya', 'Tanzania', 'Uganda', 'South Sudan', 'Rwanda', 'Burundi']
        table = [schengen + unionstate + peaceandfriendship + ca4 + tasman + andean + caricom + gcc + eac,
                 ['Schengen'] * len(schengen) + ['Union State'] * len(unionstate) + ['Peace and Friendship'] * len(
                     peaceandfriendship) + ['Central America 4'] * len(ca4) + ['Trans-Tasman Agreement'] * len(
                     tasman) + ['Andean Community'] * len(andean) + len(caricom) * ['Caricom'] + len(gcc) * [
                     'Gulf Cooperation Council'] + ['East African Community'] * len(eac)]
        treaties = pd.DataFrame(table).transpose()
        treaties.columns = ['Country', 'Treaty']
        notreaty = pd.DataFrame([*set(treaties['Country']) ^ set(data1['Country Name'])])
        notreaty['Treaty'] = None
        notreaty.columns = ['Country', 'Treaty']
        treaties = pd.concat([notreaty, treaties])
        data1 = pd.merge(data1, treaties, how='outer')

        # Democracy Index from the EIU
        democracy.columns = ['Rank', 'Country', 'Democracy Score', 'Electoral process and pluralism',
                             'Functioning of government', 'Political participation',
                             'Political culture', 'Civil liberties', 'Regime type', 'Region[n 1]',
                             'Changes from last year']
        data1 = pd.merge(data1, democracy, on='Country', how='outer')

        # bring it all together
        data1['gini'] = gini[['1990 [YR1990]', '2000 [YR2000]', '2010 [YR2010]', '2011 [YR2011]',
                              '2012 [YR2012]', '2013 [YR2013]', '2014 [YR2014]', '2015 [YR2015]',
                              '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]']].astype(float).mean(
            axis=1)
        combos = list(combinations(data1['Country'], 2))
        borders = pd.DataFrame(combos).drop_duplicates()
        test2 = pd.merge(pd.merge(borders, data1.drop('Rank', 1), left_on=0, right_on='Country Name', how='inner'),
                         data1.drop('Rank', 1), left_on=1, right_on='Country Name', how='inner')
        # More statistics
        test2.columns = ['Country_x', 'Country_y', 'C', '2015 cpi_x', '2018 pfi_x', 'DB_y018_x',
                         'DB_y019_x', 'Region_x', 'Subregion_x', 'Homicide Rate_x',
                         'Homicide County_x', 'Year_x', 'Source_x', 'score_x', 'Country Name_x',
                         'Population_x',
                         'Population growth_x',
                         'GDP_x',
                         'GDP per capita_x', 'Treaty_x',
                         'Democracy Score_x', 'Electoral process and pluralism_x',
                         'Functioning of government_x', 'Political participation_x',
                         'Political culture_x', 'Civil liberties_x', 'Regime type_x',
                         'Region[n_x]_x', 'Changes from last year_x', 'gini_x', 'Country',
                         '2015 cpi_y', '2018 pfi_y', 'DB_y018_y', 'DB_y019_y', 'Region_y',
                         'Subregion_y', 'Homicide Rate_y', 'Homicide County_y', 'Year_y',
                         'Source_y', 'score_y', 'Country Name_y',
                         'Population_y',
                         'Population growth_y',
                         'GDP_y',
                         'GDP per capita_y', 'Treaty_y',
                         'Democracy Score_y', 'Electoral process and pluralism_y',
                         'Functioning of government_y', 'Political participation_y',
                         'Political culture_y', 'Civil liberties_y', 'Regime type_y',
                         'Region[n_x]_y', 'Changes from last year_y', 'gini_y']
        test2['Population difference'] = test2['Population_x'] / test2['Population_y']
        test2['Population growth difference'] = test2['Population growth_x'] / test2['Population growth_y']
        test2['GDP difference'] = test2['GDP_x'] / test2['GDP_y']
        test2['GDP per capita difference'] = test2['GDP per capita_x'] / test2['GDP per capita_y']
        test2['Democracy Score difference'] = test2['Democracy Score_x'] / test2['Democracy Score_y']
        test2['Electoral process and pluralism difference'] = test2['Electoral process and pluralism_x'] / test2[
            'Electoral process and pluralism_y']
        test2['Functioning of government difference'] = test2['Functioning of government_x'] / test2[
            'Functioning of government_y']
        test2['Political participation difference'] = test2['Political participation_x'] / test2[
            'Political participation_y']
        test2['Population difference'] = [1 / x if x > 1 else x for x in test2['Population difference']]
        test2['Population growth difference'] = [1 / x if x > 1 else x for x in test2['Population growth difference']]
        test2['GDP difference'] = [1 / x if x > 1 else x for x in test2['GDP difference']]
        test2['GDP per capita difference'] = [1 / x if x > 1 else x for x in test2['GDP per capita difference']]
        # test2 = pd.merge(test2,df2, left_on='Country_x', right_on='Name of country',how='outer').drop('Borders',1)
        test2['GDP product'] = test2['GDP_x'] * test2['GDP_y']
        test2['GDP per capita in both'] = (test2['GDP_x'] + test2['GDP_y']) / (
                    test2['Population_x'] + test2['Population_y'])
        test2 = test2.drop_duplicates('GDP product').reset_index()
        df1 = test2
        df1['cpi difference'] = df1['2015 cpi_x'] / df1['2015 cpi_y']
        # df1['cpi difference'] = [1/x if x> 1 for x in df1['cpi difference'] else x]
        df1['GDP sum'] = df1['GDP_x'] + df1['GDP_y']
        df1['cpi difference'] = [1 / x if x > 1 else x for x in df1['cpi difference']]
        df1['db difference'] = df1['DB_y019_x'] / df1['DB_y019_y']
        df1['db difference'] = [1 / x if x > 1 else x for x in df1['db difference']]
        df1['pfi difference'] = df1['2018 pfi_x'] / df1['2018 pfi_y']
        df1['pfi difference'] = [1 / x if x > 1 else x for x in df1['pfi difference']]
        df1['homicide difference'] = df1['Homicide Rate_x'] / df1['Homicide Rate_y']
        df1['homicide difference'] = [1 / x if x > 1 else x for x in df1['homicide difference']]
        # df1['trust difference'] = df1['trust1'] / df1['trust2']
        # df1['trust difference'] = [1/x if x> 1 else x for x in df1['pfi difference']]
        df1['Country Similarity'] = (df1['cpi difference'] + df1['db difference'] + df1['pfi difference'] + df1[
            'homicide difference']) / 4
        df1['average cpi'] = df1[['2015 cpi_x', '2015 cpi_y']].mean(axis=1)
        df1['average db'] = df1[['DB_y019_x', 'DB_y019_y']].mean(axis=1)
        df1['average pfi'] = df1[['2018 pfi_x', '2018 pfi_y']].mean(axis=1)
        df1['average homicide'] = df1[['Homicide Rate_x', 'Homicide Rate_y']].mean(axis=1)
        df1['average Democracy Score'] = df1[['Democracy Score_x', 'Democracy Score_y']].mean(axis=1)
        df1['average Electoral process and pluralism'] = df1[
            ['Electoral process and pluralism_x', 'Electoral process and pluralism_y']].mean(axis=1)
        df1['average Functioning of Government'] = df1[
            ['Functioning of government_x', 'Functioning of government_y']].mean(axis=1)
        df1['average Political participation'] = df1[['Political participation_x', 'Political participation_y']].mean(
            axis=1)
        df1['average Civil Liberties'] = df1['Civil liberties_x'] / df1['Civil liberties_y']
        # df1['average trust'] = df1[['trust1','trust2']].mean(axis=1)
        df1['Border Status'] = np.where(df1['Treaty_x'] == df1['Treaty_y'], 'Open', 'Closed')
        df1['Combined Population'] = df1['Population_x'] + df1['Population_y']
        df1['GDP per capita'] = df1['GDP sum'] / df1['Combined Population']
        df1['viability'] = df1['average cpi'] * df1['average db'] / (df1['average pfi'] * df1['average homicide'])
        df1['viability2'] = df1['average cpi'] * df1['average db'] * df1['GDP per capita'] / (
                    df1['average pfi'] * df1['average homicide'])
        df1 = df1.drop_duplicates('Combined Population').sort_values('viability', ascending=False).reset_index().drop(
            'index', 1)
        df1['Status'] = df1['Border Status'].replace('Open', 1).replace('Closed', 0)
        df1 = df1.replace(np.inf, np.nan)
        df1['Same Region'] = df1['Region_x'] == df1['Region_y']
        df1['Same Subregion'] = df1['Subregion_x'] == df1['Subregion_y']
        df1['Same Region'] = df1['Same Region'].replace(False, 0)
        df1['Same Subregion'] = df1['Same Subregion'].replace(False, 0)
        test = df1.dropna(
            subset=['Country_x', 'Country_y', '2018 pfi_x', '2018 pfi_y', 'GDP_x', 'GDP_y', 'GDP per capita_x',
                    'GDP per capita_y', '2015 cpi_x', '2015 cpi_y', 'DB_y018_x', 'DB_y018_y', 'Homicide Rate_x',
                    'Homicide Rate_y', 'Democracy Score_x', 'Democracy Score_y', 'Status'])[
            ['Country_x', 'Country_y', 'GDP_x', 'GDP_y', 'GDP per capita_x', 'GDP per capita_y', '2015 cpi_x',
             '2015 cpi_y', 'DB_y018_x', 'DB_y018_y', 'Homicide Rate_x', 'Homicide Rate_y', 'Democracy Score_x',
             'Democracy Score_y', 'Population_x', 'Population_y', '2018 pfi_x', '2018 pfi_y', 'Status', 'Same Region',
             'Same Subregion']]
        for x in ['2015 cpi', 'DB_y018', '2018 pfi', 'Homicide Rate', 'Democracy Score', 'Population']:
            test['Average ' + x] = test[[x + '_x', x + '_y']].mean(axis=1)
        test['Population'] = test['Population_x'] + test['Population_y']
        test['GDP'] = test['GDP_x'] + test['GDP_y']
        test['GDP per capita'] = test['GDP'] / test['Population']
        test['worse cpi'] = test[['2015 cpi_y', '2015 cpi_x']].min(axis=1)
        test['worse pfi'] = test[['2018 pfi_x', '2018 pfi_y']].max(axis=1)
        test['worse DB'] = test[['DB_y018_y', 'DB_y018_x']].min(axis=1)
        test['worse homicide'] = test[['Homicide Rate_y', 'Homicide Rate_x']].max(axis=1)
        test['worse GDP per capita'] = test[['GDP per capita_y', 'GDP per capita_x']].min(axis=1)
        test['better cpi'] = test[['2015 cpi_y', '2015 cpi_x']].max(axis=1)
        test['better pfi'] = test[['2018 pfi_x', '2018 pfi_y']].min(axis=1)
        test['better DB'] = test[['DB_y018_y', 'DB_y018_x']].max(axis=1)
        test['better homicide'] = test[['Homicide Rate_y', 'Homicide Rate_x']].min(axis=1)
        test['better GDP per capita'] = test[['GDP per capita_y', 'GDP per capita_x']].max(axis=1)
        test['worse Democracy Score'] = test[['Democracy Score_y', 'Democracy Score_x']].min(axis=1)
        test['better Democracy Score'] = test[['Democracy Score_y', 'Democracy Score_x']].max(axis=1)
        self.data2 = data1.dropna(subset=['Country', 'Population, total [SP.POP.TOTL]'])[~data1['Country'].isin(
            ['Guam', 'South Asia', 'American Samoa', 'Europe & Central Asia', 'East Asia & Pacific',
             'Middle East & North Africa', 'Latin America & Caribbean', 'Sub-Saharan Africa'])].drop_duplicates(
            'Country').sort_values('Population, total [SP.POP.TOTL]', ascending=False).reset_index(drop=True)
        self.test = test
