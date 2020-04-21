import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Dataset obsahuje nasledujici promenne:
 'Age' - vek v rocich
 'Fare' - cena jizdenky
 'Name' - jmeno cestujiciho
 'Parch' - # rodicu/deti daneho cloveka na palube
 'PassengerId' - Id
 'Pclass' - Trida, 1 = 1. trida, 2 = 2.trida, 3 = 3.trida
 'Sex' - pohlavi
 'SibSp' - # sourozencu/manzelu daneho cloveka na ppalube
 'Survived' - 0 = Neprezil, 1 = Prezil
 'Embarked' - Pristav, kde se dany clovek nalodil. C = Cherbourg, Q = Queenstown, S = Southampton
 'Cabin' - Cislo kabiny
 'Ticket' - Cislo tiketu
"""


def load_dataset(train_file_path, test_file_path):
    """
    Napiste funkci, ktera nacte soubory se souboru zadanych parametrem a vytvori dva separatni DataFrame. Pro testovani vyuzijte data 'data/train.csv' a 'data/test.csv'
    Ke kazdemu dataframe pridejte sloupecek pojmenovaný jako "Label", ktery bude obsahovat hodnoty "Train" pro train.csv a "Test" pro test.csv.

    1. Pote slucte oba dataframy.
    2. Z vysledneho slouceneho DataFramu odstraňte sloupce  "Ticket", "Embarked", "Cabin".
    3. Sloučený DataDrame bude mít index od 0 do do počtu řádků.
    4. Vratte slouceny DataDrame.
    """

    train_set = pd.read_csv(train_file_path)
    test_set = pd.read_csv(test_file_path)

    train_set['Label'] = 'Train'
    test_set['Label'] = 'Test'

    frames = [train_set, test_set]

    result = pd.concat(frames)

    result = result.drop(columns=['Ticket', 'Embarked', 'Cabin'])
    result.reset_index(drop=True, inplace=True)

    return result


def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ze zadaneho dataframu zjistete chybejici hodnoty. Vytvorte DataFrame, ktery bude obsahovat v indexu jednotlive promenne
    a ve prvnim sloupci bude promenna 'Total' obsahujici celkovy pocet chybejicich hodnot a ve druhem sloupci promenna 'Percent',
    ve ktere bude procentualni vyjadreni chybejicich hodnot vuci celkovemu poctu radku v tabulce.
    DataFrame seradte od nejvetsich po nejmensi hodnoty.
    Vrattre DataFrame chybejicich hodnot a celkovy pocet chybejicich hodnot.

    Priklad:

               |  Total  |  Percent
    "Column1"  |   34    |    76
    "Column2"  |   0     |    0

    """

    columns = df.columns

    result = pd.DataFrame(index=columns,
                          columns=['Total', 'Percent'])

    result['Total'] = pd.isna(df[columns]).sum()
    result['Percent'] = round((result['Total']/df.shape[0]) * 100)
    result = result.astype('int32')
    result.sort_values(by='Total',
                       ascending=False,
                       inplace=True,
                       ignore_index=True)
    return result


def substitute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chybejici hodnoty ve sloupecku "Age" nahradte meanem hodnot z "Age".
    Chybejici hodnoty ve sloupecku "Fare" nahradte meadianem hodnot z "Fare".
    V jednom pripade pouzijte "loc" a ve druhem "fillna".
    Zadany DataFrame neupravujte, ale vytvorte si kopii.
    Vratte upraveny DataFrame.
    """

    df_copy = df.copy()

    df_copy['Age'].fillna(df_copy['Age'].mean(), inplace=True)
    df_copy.loc[pd.isna(df_copy.loc[:, 'Fare']), 'Fare'] = df_copy.loc[:, 'Fare'].median()

    return df_copy


def get_correlation(df: pd.DataFrame) -> float:
    """
    Spocitejte korelaci pro "Age" a "Fare" a vratte korelaci mezi "Age" a "Fare".
    """
    return df['Age'].corr(df['Fare'])


def get_survived_per_class(df: pd.DataFrame,
                           group_by_column_name: str) -> pd.DataFrame:
    """
    Spocitejte prumer z promenne "Survived" pro kazdou skupinu zadanou parametrem "group_by_column_name".
    Hodnoty seradte od nejvetsich po mejmensi.
    Hodnoty "Survived" zaokhroulete na 2 desetinna mista.
    Vratte pd.DataFrame.

    Priklad:

    get_survived_per_class(df, "Sex")

                 Survived
    Male     |      0.32
    Female   |      0.82

    """

    result = df.groupby(df[group_by_column_name]).mean()[['Survived']]
    result = result.round(2)

    result.reset_index(inplace=True)

    result.sort_values(by='Survived',
                       ascending=False,
                       inplace=True,
                       ignore_index=True)

    return result


def get_outliers(df: pd.DataFrame) -> (int, str):
    """
    Vyfiltrujte odlehle hodnoty (outliers) ve sloupecku "Fare" pomoci metody IRQ.
    Tedy spocitejte rozdil 3. a 1. kvantilu, tj. IQR = Q3 - Q1.
    Pote odfiltrujte vsechny hodnoty nesplnujici: Q1 - 1.5*IQR < "Fare" < Q3 + 1.5*IQR.
    Namalujte box plot pro sloupec "Fare" pred a po vyfiltrovani outlieru.
    Vratte tuple obsahujici pocet outlieru a jmeno cestujiciho pro nejvetsi outlier.
    """

    third_quantile = df['Fare'].quantile(0.75)
    first_quantile = df['Fare'].quantile(0.25)

    delta = third_quantile - first_quantile

    # df.boxplot('Fare')

    is_outlier = np.logical_or(df['Fare'] < first_quantile - 1.5 * delta,
                               df['Fare'] > third_quantile + 1.5 * delta)

    outliers = df[is_outlier]
    max_outlier = df.iloc[outliers['Fare'].idxmax()]

    df = df[np.logical_not(is_outlier)]

    # df.boxplot('Fare')
    # plt.show()

    return (outliers.shape[0], max_outlier['Name'])


def normalise(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Naskalujte sloupec "col" zadany parametrem pro kazdou "Pclass" hodnotu z dataframu "df" zvlast.
    Pouzijte vzorec: scaled_x_i = (x_i - min(x)) / (max(x) - min(x)), kde "x_i" prestavuje konkretni hodnotu ve sloupeci "col".
    Vratte naskalovany dataframe.
    """

    by_class = df.groupby('Pclass')[col]

    by_class_min = by_class.min()
    by_class_max = by_class.max()

    def normalised(row):
        min_value = by_class_min[row['Pclass']]
        max_value = by_class_max[row['Pclass']]
        delta = max_value - min_value

        row[col] = (row[col] - min_value)/delta
        return row

    df = df.apply(normalised, axis=1)

    return df


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvorte 3 nove promenne:
    1. "Fare_scaled" - vytvorte z "Fare" tak, aby mela nulovy prumer a jednotkovou standartni odchylku.
    2. "Age_log" - vytvorte z "Age" tak, aby nova promenna byla logaritmem puvodni "Age".
    3. "Sex" - Sloupec "Sex" nahradte: "female" -> 1, "male" -> 0, kde 0 a 1 jsou integery.

    Nemodifikujte predany DataFrame, ale vytvorte si novy, upravte ho a vratte jej.
    """

    df['Fare_scaled'] = (df['Fare'] - df['Fare'].mean())/df['Fare'].std()
    df['Age_log'] = np.log(df['Age'])
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)

    return df


def determine_survival(df: pd.DataFrame, n_interval: int, age: float, sex: str) -> float:
    """
    Na zaklade statistickeho zpracovani dat zjistete pravdepodobnost preziti Vami zadaneho cloveka (zadava se vek a pohlavi pomoci parametru "age" a "sex")

    Vsechny chybejici hodnoty ve vstupnim DataFramu ve sloupci "Age" nahradte prumerem.
    Rozdelte "Age" do n intervalu zadanych parametrem "n_interval". Napr. pokud bude Age mit hodnoty [2, 13, 18, 25] a mame jej rozdelit do 2 intervalu,
    tak bude vysledek:

    0    (1.977, 13.5]
    1    (1.977, 13.5]
    2     (13.5, 25.0]
    3     (13.5, 25.0]

    Pridejte k rozdeleni jeste pohlavi. Tj. pro kazdou kombinaci pohlavi a intervalu veku zjistete prumernou
    pravdepodobnost preziti ze sloupce "Survival" a tu i vratte.

    Vysledny DataFrame:

    "AgeInterval"   |    "Sex"    |   "Survival Probability"
       (0-10)       | "male"      |            0.21
       (0-10)       | "female"    |            0.28
       (10-20)      | "male"      |            0.10
       (10-20)      | "female"    |            0.15
       atd...

    Takze vystup funkce determine_survival(df, n_interval=20, age = 5, sex = "male") bude 0.21. Tato hodnota bude navratovou hodnotou funkce.

    """

    df = substitute_missing_values(df)

    min_value, max_value = df['Age'].min(), df['Age'].max()

    if age > max_value or sex not in ['male', 'female']:
        return 0

    df['Age'] = pd.cut(df['Age'], n_interval)

    grouped = df.groupby(['Age', 'Sex'])['Survived'].mean()

    return grouped[age, sex]
