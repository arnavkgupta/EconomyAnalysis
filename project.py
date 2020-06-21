"""
This module contains several functions for analyzing
economic data of 162 countries in the world.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('/home/economicdata2017-2017.csv')
master_data = pd.read_csv('/home/master_data.csv')


def size_of_government(data):
    """
    Generates a scatter plot for the relationship between the
    average score of a country's living conditions and
    the size of the government. The plot is saved in plot7.png file
    """
    data = data[['Money growth',
                 'Reliability of police',
                 'Ownership of banks',
                 'Starting a business',
                 'Size of Government',
                 'Integrity of the legal system',
                 'Rank',
                 'Impartial courts',
                 'Transfers and subsidies']]
    data['avg'] = data[['Money growth',
                        'Reliability of police',
                        'Ownership of banks',
                        'Starting a business',
                        'Integrity of the legal system',
                        'Rank',
                        'Impartial courts',
                        'Transfers and subsidies']].mean(axis=1)
    data = data[['Size of Government', 'avg']]
    data.dropna()
    sns.regplot(x='Size of Government', y='avg', data=data)
    plt.title("Relationship between size of government and living standards")
    plt.xlabel('Size of Government')
    plt.ylabel('Average Score')
    plt.savefig('plot7.png',  bbox_inches='tight')


def plot_average_score_and_index(data):
    """
    Generate a scatter plot of the relationship between the
    average score among Freedom to own foreign currency bank accounts,
    Freedom of foreigners to visit, Freedom to Trade
    Internationally, Judicial independence versus
    Economic Freedom Summary Index with a linear
    regression line. The plot is save in plot1.png file
    """
    data = data[['Freedom to own foreign currency bank accounts',
                 'Freedom of foreigners to visit',
                 'Freedom to Trade Internationally',
                 'Judicial independence',
                 'Economic Freedom Summary Index',
                 'Countries']]
    data['avg'] = data[['Freedom to own foreign currency bank accounts',
                        'Freedom of foreigners to visit',
                        'Freedom to Trade Internationally',
                        'Judicial independence']].mean(axis=1)
    data = data[['Countries', 'avg', 'Economic Freedom Summary Index']]
    data.dropna()
    sns.regplot(x='avg', y='Economic Freedom Summary Index', data=data)
    plt.title("Relationship between Average Score and \
              Economic Freedom Summary Index")
    plt.xlabel('Economic Freedom Summary Index')
    plt.ylabel('Average Score')
    plt.savefig('plot1.png',  bbox_inches='tight')


def greater_than_8(data):
    """
    Generate a scatter plot of the countries that have
    Economic Freedom Summary Index greater than 5.5
    as plot3.png file.
    """
    data = data[['Freedom to own foreign currency bank accounts',
                 'Freedom of foreigners to visit',
                 'Freedom to Trade Internationally',
                 'Judicial independence',
                 'Economic Freedom Summary Index',
                 'Countries']]
    data['avg'] = data[['Freedom to own foreign currency bank accounts',
                        'Freedom of foreigners to visit',
                        'Freedom to Trade Internationally',
                        'Judicial independence']].mean(axis=1)
    data = data[['Countries', 'avg', 'Economic Freedom Summary Index']]
    data = data[data['Economic Freedom Summary Index'] >= 8]
    data.dropna()
    sns.relplot(x='avg', y='Economic Freedom Summary Index',
                hue='Countries', data=data)
    plt.title("Countries that have Economic Freedom \
              Summary Index greater than 8")
    plt.xlabel('Economic Freedom Summary Index')
    plt.ylabel('Average Score')
    plt.savefig('plot2.png',  bbox_inches='tight')


def less_than_5_and_half(data):
    """
    Generate a scatter plot of the countries that have
    Economic Freedom Summary Index less than 5.5
    as plot3.png file.
    """
    data = data[['Freedom to own foreign currency bank accounts',
                 'Freedom of foreigners to visit',
                 'Freedom to Trade Internationally',
                 'Judicial independence',
                 'Economic Freedom Summary Index',
                 'Countries']]
    data['avg'] = data[['Freedom to own foreign currency bank accounts',
                        'Freedom of foreigners to visit',
                        'Freedom to Trade Internationally',
                        'Judicial independence']].mean(axis=1)
    data = data[['Countries', 'avg', 'Economic Freedom Summary Index']]
    data = data[data['Economic Freedom Summary Index'] < 5.5]
    # print(data)
    data.dropna()
    sns.relplot(x='avg', y='Economic Freedom Summary Index',
                hue='Countries', data=data)
    plt.title("Countries that have Economic Freedom \
              Summary Index less than 5.5")
    plt.xlabel('Economic Freedom Summary Index')
    plt.ylabel('Average Score')
    plt.savefig('plot3.png',  bbox_inches='tight')


def highest_freedom_index(data):
    """
    Return the country that has the highest economic freedom
    index in 2017. This country is presented as a tuple of
    length 2 where the first element is the name of the
    country and the second is the economic freedom index
    of that country.

    If there are two coutry with the same highest economic freedom
    index, returns the country that appears first in the file.
    """
    max_freedom = data['Economic Freedom Summary Index'].max()
    countries = data[data['Economic Freedom Summary Index'] == max_freedom]
    first_index = countries['Economic Freedom Summary Index'].idxmin()
    result = countries[['Countries',
                        'Economic Freedom Summary Index']].loc[first_index]
    return result['Countries'], result['Economic Freedom Summary Index']


def lowest_freedom_index(data):
    """
    Return the country that has the lowest economic freedom
    index  in 2017. This country is presented as a tuple of
    length 2  where the first element is the name of the
    country and the second is the economic freedom index
    of that country.

    If there are two coutry with the same lowest economic freedom
    index, returns the country that appears first in the file.
    """
    min_freedom = data['Economic Freedom Summary Index'].min()
    countries = data[data['Economic Freedom Summary Index'] == min_freedom]
    first_index = countries['Economic Freedom Summary Index'].idxmin()
    result = countries[['Countries',
                        'Economic Freedom Summary Index']].loc[first_index]
    return result['Countries'], result['Economic Freedom Summary Index']


def plot_HK_rank_overtime(master_data):
    """
    Generate a scatter plot of the results of how the economy
    rank of Hong Kong change between 1970 and 2017 (inclusive)
    as plot4.png file.
    """
    master_data = master_data[master_data['Countries'] == 'Hong Kong']
    sns.relplot(x='Year', y='Rank', size='Economic Freedom Summary Index',
                kind='scatter', data=master_data)
    plt.title("Economy Rank of Hong Kong over time")
    plt.xlabel('Year')
    plt.ylabel('Rank')
    plt.xticks(rotation=-45)
    plt.xticks([1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015])
    plt.savefig('plot4.png', bbox_inches='tight')


def plot_Ven_rank_overtime(master_data):
    """
    Generate a scatter plot of the results of how the economy
    rank of Venezuela change between 1970 and 2017 (inclusive)
    as plot5.png file.
    """
    master_data = master_data[master_data['Countries'] == 'Venezuela']
    sns.relplot(x='Year', y='Rank', size='Economic Freedom Summary Index',
                kind='scatter', data=master_data)
    plt.title("Economy Rank of Venezuela over time")
    plt.xlabel('Year')
    plt.ylabel('Rank')
    plt.xticks(rotation=-45)
    plt.xticks([1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015])
    plt.savefig('plot5.png',  bbox_inches='tight')


def plot_US_rank_overtime(master_data):
    """
    Generate a scatter plot of the results of how the economy
    rank of United States change between 1970 and 2017 (inclusive)
    as plot6.png file.
    """
    master_data = master_data[master_data['Countries'] == 'United States']
    sns.relplot(x='Year', y='Rank', size='Economic Freedom Summary Index',
                kind='scatter', data=master_data)
    plt.title("Economy Rank of United States over time")
    plt.xlabel('Year')
    plt.ylabel('Rank')
    plt.xticks([1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015])
    plt.xticks(rotation=-45)
    plt.savefig('plot6.png', bbox_inches='tight')


def fit_and_predict_rank(data):
    """
    Return the accuracy of the machine learning
    model which predicts the economy freedom quartile
    to which a country belongs. One may change the
    variable test_size for different data split.
    """
    data = data[['Government consumption', 'Government investment',
                 'Labor market regulations', 'Legal System & Property Rights',
                 'Top marginal tax rate', 'Inflation: Most recent year',
                 'Freedom to Trade Internationally',
                 'Hiring regulations and minimum wage',
                 'Hiring and firing regulations', 'Business regulations',
                 'Quartile']]
    data = data.dropna()

    X = data.loc[:, data.columns != 'Quartile']
    y = data['Quartile']
    # change test_size for different data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # y_train_pred = model.predict(X_train)
    # y_test_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return accuracy


def main():
    size_of_government(data)
    greater_than_8(data)
    less_than_5_and_half(data)
    plot_average_score_and_index(data)
    plot_HK_rank_overtime(master_data)
    plot_Ven_rank_overtime(master_data)
    plot_US_rank_overtime(master_data)

    print('The country that has the highest freedom index:')
    print(highest_freedom_index(data))
    print('The country that has the lowest freedom index:')
    print(lowest_freedom_index(data))
    print('The accuracy of the model:')
    print(fit_and_predict_rank(data))


if __name__ == "__main__":
    main()
