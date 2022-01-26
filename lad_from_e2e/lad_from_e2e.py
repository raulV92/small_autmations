# -*- coding: utf-8 -*-

import os
# import numpy as np
import pandas as pd
from pathlib import Path

def read_e2e():
    """Designed to work with E2E(V2)"""
    cols=['Country Id', 'SMS ID','Period Group', 'Period ID', 'Planned  Audit Date', 'Visit Audit date']
    #cols=['Country Id','Period ID']
    file_name=Path('E2E',os.listdir(Path('E2E'))[0])
    e2e = pd.read_csv(file_name,sep='\t',encoding='utf_16_le')
    print(file_name)
    return e2e[cols]

def get_lad(linea):
    visita='Visit Audit date'
    planeada='Planned  Audit Date'
    if len(linea[visita]) < 4:
        return linea[planeada]
    else:
        return linea[visita]

def main():

    periodos_sur=    {
        'Bi-Monthly_Food':1,
        'Bi-Monthly_Drug':2,
        'Monthly':3
        }
    rep= read_e2e()

    rep=rep[rep['SMS ID'].str.len()==10].drop_duplicates()
    print(rep)
    rep['LAD']=rep.apply(get_lad,axis=1)
    rep['Period Group'].replace(periodos_sur,inplace=True)

    cols_order=['Country Id', 'SMS ID','Period Group', 'Period ID','LAD', 'Planned  Audit Date', 'Visit Audit date']
    return rep[cols_order]
if __name__=='__main__':
    df=main()
    df.to_csv('lad_from_E2E.csv')
