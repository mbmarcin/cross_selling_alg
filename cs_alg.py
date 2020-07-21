# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:05:14 2020

@author: marcin
"""

##import numpy as np 
import pandas as pd
from efficient_apriori import apriori


# https://www.wpdesk.pl/blog/analiza-koszykowa-algorytm-apriori/
# https://pypi.org/project/efficient-apriori/
# https://medium.com/@deepak.r.poojari/apriori-algorithm-in-python-recommendation-engine-5ba89bd1a6da
# https://efficient-apriori.readthedocs.io/en/latest/?badge=latest


class AprioriAalgorithm:

    def __init__(self, df_sc, col_group, col_to_group_and_split,
                 name_cols_split, thresh=2, mode=0, min_supp=0.05,
                 min_conf=0.05, rule_lhs=1, rule_rhs=1, sort_by='rule.support'):

        self.df_sc = df_sc
        self.col_group = col_group
        self.col_to_group_and_split = col_to_group_and_split
        self.name_cols_split = name_cols_split
        self.thresh = thresh
        self.mode = mode
        self.min_supp = min_supp
        self.min_conf = min_conf
        self.rule_lhs = rule_lhs
        self.rule_rhs = rule_rhs
        self.sort_by = sort_by

    def __group_by_id_and_split_list(self):

        """
        :param df_sc: df
        :param col_group: str (name of col)
        :param col_to_group_and_split: str (name of col)
        :param name_cols_split: str (name of cols)
        :return: df with split ids
        """

        # note that the apply function here takes a series made up of the values
        # for each group. We then call the .tolist() method on the series to make
        # it into a list

        print('preparing data...')
        df_ = self.df_sc.groupby(self.col_group)[self.col_to_group_and_split].apply(
            lambda group_series: list(set(group_series))).reset_index()
        df_ = pd.DataFrame([pd.Series(x) for x in df_.loc[:, self.col_to_group_and_split]])
        df_.columns = ['{}_{}'.format(self.name_cols_split, i) for i in df_.columns]

        return df_.dropna(thresh=self.thresh)  # thresh?

    def __prep_list_of_tuples(self):
        """
        Parameters
        ----------
        df_tr : df with skus per transactions
    
        Returns
        -------
        List of tuples 
    
        """
        # x = self.__group_by_id_and_split_list

        transactions = [tuple(row) for row in self.__group_by_id_and_split_list().values.tolist()]

        l1 = list()

        for i in transactions:
            l2 = list()
            l1.append(l2)
            for j in i:
                if str(j) == 'nan':
                    pass
                else:
                    l2.append(str(j))

        return [tuple(i) for i in l1]

    def alg_apriori(self):

        print('starting analysis apriori')

        itemsets, rules = apriori(self.__prep_list_of_tuples(), min_support=self.min_supp, min_confidence=self.min_conf)

        if self.mode == 0:
            return rules
        elif self.mode == 1:
            return itemsets
        elif self.mode == 2:
            rules_rhs = filter(lambda rule: len(rule.lhs) == self.rule_lhs and len(rule.rhs) == self.rule_rhs, rules)

            cols = ['id1', 'id2', 'supp', 'conf', 'lift']
            lst_data = list()

            for rule in sorted(rules_rhs, key=lambda rule: self.sort_by):
                lst_data.append([rule.lhs, rule.rhs, rule.support, rule.confidence, rule.lift])
            return pd.DataFrame(lst_data, columns=cols)

        else:
            print('Put val mode...')

# DEBUG
# #data = pd.read_csv('C://Users//marcin//Documents//Python Scripts//online_retail.txt')
# #data = pd.read_excel('online_retail.xlsx') 
# #data = data.iloc[:, [0, 1]]


# data_sc = pd.read_csv('retail_sample.txt') 

# # Dropping the rows without any invoice number 
# # data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True) 
# # data['InvoiceNo'] = data['InvoiceNo'].astype('str') 

# # # Dropping all transactions which were done on credit 
# # data = data[~data['InvoiceNo'].str.contains('C')] 

# # basket_France = data[data['Country'] =="France"] 

# #basket_France.to_csv('retail_sample.txt', index=False)

# data = data_sc.iloc[:, [0, 1]].astype(str)


# apri = Apriori_algorithm(data,'InvoiceNo','StockCode', 'sku', mode=2)


# print(
# #alg_apriori(data, mode=2, rule_lhs=1)
# apri.alg_apriori()
# )
