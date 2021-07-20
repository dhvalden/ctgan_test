import warnings
import pandas as pd
from sdv.tabular import CTGAN

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    data = pd.read_csv('ESS8e02.1_F1.csv')
    vars_to_use = ['idno',
                   'cntry',
                   'nwspol',
                   'netusoft',
                   'netustm',
                   'ppltrst',
                   'pplfair',
                   'pplhlp',
                   'polintr',
                   'psppsgva',
                   'actrolga',
                   'psppipla',
                   'cptppola',
                   'trstprl',
                   'trstlgl',
                   'trstplc',
                   'trstplt',
                   'trstprt',
                   'trstep',
                   'trstun',
                   'vote']
    data = data[vars_to_use]
    print(data.head())
    model = CTGAN(
        primary_key='idno'
    )
    model.fit(data)
    new_data = model.sample(50000)
    new_data.to_csv('synth_data.csv')
    print(new_data.head())


if __name__ == "__main__":
    main()
