# -*- coding: utf-8 -*-
"""
* Version: 28-Dic-2022
* TODO Agregar avance de auditoria por Auditor
* TODO Lineas con auditores en varios clusters

"""

import time

init = time.perf_counter()

import pretty_errors

import pandas as pd
import numpy as np

from pathlib import Path
from inputReaderClass import InputReader

import logging
logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s',
                    filename="log.txt",
                    filemode='w',)
logging.warning('Auditor with errors:')


def add_group_cols(data_set: pd.DataFrame, nivel: str, kpi='Date_'):
    """
    Default to date compliance  options kpi= Date_, Auditor_
    original Col names:
                'Date_Compliant','Date_NonCompliant',
                'Auditor_Compliant','Auditor_NonCompliant'
    """
    grupos = data_set.groupby(nivel)

    # define new cols name
    new_perc = kpi + 'Percent_' + nivel
    new_comp = kpi + 'Compliant_' + nivel
    new_nonComp = kpi + 'NonCompliant_' + nivel

    # init new cols
    data_set[new_perc] = 0
    data_set[new_comp] = 0
    data_set[new_nonComp] = 0

    for i, df in grupos:
        comp_stores = df[kpi + 'Compliant'].sum()
        non_comp_stores = df[kpi + 'NonCompliant'].sum()
        block_percent = comp_stores / (non_comp_stores + comp_stores)
        # print(i,comp_stores,non_comp_stores,block_percent)
        # breakpoint()
        data_set[new_perc].where(data_set[nivel] != i,
                                 other=block_percent,
                                 inplace=True)
        data_set[new_comp].where(data_set[nivel] != i,
                                 other=comp_stores,
                                 inplace=True)
        data_set[new_nonComp].where(data_set[nivel] != i,
                                    other=non_comp_stores,
                                    inplace=True)

    # return data_set


def add_represent(data_set: pd.DataFrame, nivel: str):
    grupos = data_set.groupby(nivel)

    col_name = nivel + '_representativeness'
    data_set[col_name] = data_set['Date_NonCompliant'] / data_set[
        'Date_NonCompliant_' + nivel]

    # return data_set


def tso_comp(row,tso_ranges):


    limite_inf = tso_ranges[row['PERIOD_ID']][0]
    limite_sup = tso_ranges[row['PERIOD_ID']][1]
    ## NOTE
    # Caso ED Brasil :
    if row['COUNTRY_ID'] == 'Brasil':
        if row['PERIOD_ID'][0:4] == 'MONT':
            #breakpoint()
            limite_inf = 27
            limite_sup = 33
        elif row['PERIOD_ID'][0:4] == 'DRUG' or row['PERIOD_ID'][0:4] == 'FOOD':
            limite_inf = 56
            limite_sup = 65
        else:
            raise ValueError('Not a valid period for Brasil')

    ###

    if row['STORE_STATUS'] not in ['FULLAUDIT', 'CANS']:
        return np.nan
        # TODO agragar filtro de actividad
    elif 'SOT' in row['Activity Local Name']:
        return np.nan

    else:
        return limite_inf <= row['ELAPSED_DAYS'] and row[
            'ELAPSED_DAYS'] <= limite_sup


def count_comp(serie: pd.Series, valor: bool):

    try:
        conteo = serie.value_counts().loc[valor]
    except KeyError:
        conteo = 0
    # breakpoint()
    #if serie.isna().sum()>0:
    #    breakpoint()
    return conteo


def kpi_comp_(planes, audit_summary):

    # audit_summary.rename(e2e_renames,inplace=True)
    extend_plan = pd.merge(planes,
                           audit_summary,
                           on=['SMS ID', 'Frequency'],
                           indicator=True)

    planes.to_csv('plan_prog.csv')
    audit_summary.to_csv('e2e_prog.csv')
    #print('columnas extendida:',extend_plan.columns)
    cols_output = [
        'SMS ID', 'Activity Local Name', 'State', 'City', 'Frequency',
        'Associate', 'Associate Cdar ID', 'Visit CDAR ID', 'Type',
        'COUNTRY_ID', 'PERIOD_ID', 'CLUSTER_NM', 'SUPERVISOR_NM',
        'STORE_STATUS', 'Planned Date', 'VISIT_DATE', 'ELAPSED_DAYS', '_merge'
    ]

    # Calcula verdaderos y falsos para auditor y date compliance
    extend_plan['date_compliance'] = extend_plan[
        'Planned Date'] == extend_plan['VISIT_DATE']
    extend_plan['auditor_compliance'] = extend_plan[
        'Associate Cdar ID'] == extend_plan['Visit CDAR ID']

    extend_plan['tso_compliance'] = extend_plan.apply(lambda x: tso_comp(x,InputReader.tso_ranges), axis=1)

    # print(extend_plan.columns);breakpoint()
    extend_plan[cols_output[:-1]+['tso_compliance']].to_csv(Path('updated_output','weekly',
                                                                 'tabla_extendida.csv'),index=False)

    # extend_plan.to_csv('tabla_extendida_TEST.csv')
    print(
        '3) Exportados:\n\tplan_prog.csv\n\te2e_prog.csv\n\ttabla_extendida.csv <--->Subir a Sharepoint'
    )

    #extend_plan.to_csv('tabla_true_false_FULL.csv', index = False)
    #print(extend_plan['date_compliance'].value_counts())
    #print(extend_plan['auditor_compliance'].value_counts())

    # filtra a solo tiendas visitadas
    visited_stores = extend_plan[~extend_plan['VISIT_DATE'].isna()]

    # Export true/false tab
    true_false_export = visited_stores[
        (~visited_stores['date_compliance']) |
        (~visited_stores['auditor_compliance']) |
        (visited_stores['tso_compliance'] == False)]

    legacy_columns = {
        'COUNTRY_ID': 'Country Id',
        'PERIOD_ID': 'Period ID',
        'CLUSTER_NM': 'Cluster Name',
        'SUPERVISOR_NM': 'Supervisor Name',
        'STORE_STATUS': 'Store status',
        'VISIT_DATE': 'Visit Audit date',
        'ELAPSED_DAYS': 'Elapsed Days',
    }

    true_false_export.rename(columns=legacy_columns).to_csv(Path(
        'updated_output', 'weekly','tabla_true_false.csv'),
                                                            index=False)

    print("4) Exportado: 'tabla_true_false.csv' <---> Subir a sharepoint")

    audit_perc={i:len(visited_stores[visited_stores['COUNTRY_ID']==i])/ \
                len(extend_plan[extend_plan['COUNTRY_ID']==i]) \
                for i in extend_plan['COUNTRY_ID'].unique()}
    #print(audit_perc)

    auditors = visited_stores.groupby('Associate')

    # nombres correspondinetes con funcion add_group_cols
    # definicion antes para hacer la comparacion contra lineas irregulares
    kpi_comp_columns = [
        'Country', 'QCT', 'Cluster', 'Auditor', 'Date_Compliant',
        'Date_NonCompliant', 'Auditor_Compliant', 'Auditor_NonCompliant',
        'TSO_Compliant', 'TSO_NonCompliant'
    ]
    kpi_comp = []

    # breakpoint()
    # for i in auditors:
    for nm_auditor, df_auditor in auditors:
        #if nm_auditor=='VANESSA KATTY PEREZ TECHERA':
        #    breakpoint()
        #    pass

        country = list(df_auditor['COUNTRY_ID'].unique())

        # en caso de un auditor con mas de 1 QCT...
        if len(df_auditor['SUPERVISOR_NM'].unique()) > 1:
            logging.warning('multiple QCT: '+country[0]+''+nm_auditor)
            # get qct with most stores with him
            qct_s = df_auditor['SUPERVISOR_NM'].value_counts()
            qct = qct_s.index.tolist()[0:1]
            # breakpoint()
        else:
            qct = list(df_auditor['SUPERVISOR_NM'].unique())

        # en caso de un auditor asignado a mas de 1 Cluster...
        if len(df_auditor['CLUSTER_NM'].unique()) > 1:
            logging.warning('multiple Cluster: '+country[0]+'-'+nm_auditor)
            # get qct with most stores with him
            cluster_s = df_auditor['CLUSTER_NM'].value_counts()
            cluster = cluster_s.index.tolist()[0:1]
            # breakpoint()
        else:
            cluster = list(df_auditor['CLUSTER_NM'].unique())
        # TODO : validar que no cree relacion QCT-Cluster inexistentes

        # cluster = list(df_auditor['CLUSTER_NM'].unique()) # NOTE Legacy
        auditor = [nm_auditor]

        # compliance_count=df_auditor['date_comp_aud'].value_counts().values.tolist()
        date_comp_count = [
            count_comp(df_auditor['date_compliance'], True),
            count_comp(df_auditor['date_compliance'], False)
        ]

        auditor_comp_count = [
            count_comp(df_auditor['auditor_compliance'], True),
            count_comp(df_auditor['auditor_compliance'], False)
        ]

        tso_comp_count = [
            count_comp(df_auditor['tso_compliance'], True),
            count_comp(df_auditor['tso_compliance'], False)
        ]

        #if i == 'SANTIAGO CALLE':
        #    pass
        #breakpoint()
        new_row = country + qct + cluster + auditor + date_comp_count + auditor_comp_count + tso_comp_count
        if len(new_row) != 10:
            # breakpoint()
            print('Error en columnas:')
            logging.warning('Error en columnas:')
            for i, j in zip(kpi_comp_columns, new_row):
                print('\t', i, '\t', j)
                logging.warning('\t', i, '\t', j)
        else:
            kpi_comp.append(new_row)

        #if 'AMANDA VERONICA' in nm_auditor:
        #    breakpoint()
    #breakpoint()
    #res=pd.DataFrame([filter_e2e['date_comp_aud'].value_counts().T,
    #                  filter_e2e['date_comp_aud'].value_counts(normalize=True).T])

    # nombres correspondinetes con funcion add_group_cols
    kpi_comp_columns = [
        'Country', 'QCT', 'Cluster', 'Auditor', 'Date_Compliant',
        'Date_NonCompliant', 'Auditor_Compliant', 'Auditor_NonCompliant',
        'TSO_Compliant', 'TSO_NonCompliant'
    ]

    #breakpoint()
    kpi_comp = pd.DataFrame(kpi_comp, columns=kpi_comp_columns)

    #print(pd.DataFrame(kpi_comp).head(10))
    #return kpi_comp

    #kpi_comp['Percentage']=kpi_comp['aud_Compliant']/(kpi_comp['aud_Compliant']+kpi_comp['aud_NonCompliant'])

    add_group_cols(kpi_comp, 'Cluster')
    add_group_cols(kpi_comp, 'QCT')
    add_group_cols(kpi_comp, 'Country')

    add_group_cols(kpi_comp, 'Cluster', kpi='Auditor_')
    add_group_cols(kpi_comp, 'QCT', kpi='Auditor_')
    add_group_cols(kpi_comp, 'Country', kpi='Auditor_')

    add_group_cols(kpi_comp, 'Cluster', kpi='TSO_')
    add_group_cols(kpi_comp, 'QCT', kpi='TSO_')
    add_group_cols(kpi_comp, 'Country', kpi='TSO_')

    add_represent(kpi_comp, 'Country')
    add_represent(kpi_comp, 'Cluster')

    kpi_comp.sort_values(
        by=['Country', 'QCT', 'Cluster', 'Country_representativeness'],
        axis=0,
        inplace=True)

    # extra cols:
    kpi_comp['Date_Threshold'] = 0.8
    kpi_comp['Auditor_Threshold'] = 0.9

    kpi_comp['Audit_perc'] = kpi_comp['Country'].copy()

    kpi_comp['Audit_perc'].replace(to_replace=audit_perc, inplace=True)

    try:
        # ***************   Output  **********************
        kpi_comp.to_csv(Path('updated_output', 'weekly', 'kpi_comp.csv'),
                        index=False)
        print("5) Exportado: 'kpi_comp.csv' <---> Subir a Sharepoint")
    except:
        print(
            'Por favor verifica que el archivo "final_output.csv" no este abierto'
        )

    kpi_comp['Country'].drop_duplicates().to_csv(Path('updated_output','countryData',
                                                      'Countries.csv'),
                                                 header='Country',
                                                 index=False)
    kpi_comp['QCT'].drop_duplicates().to_csv(Path('updated_output','countryData',
                                                 'QCT.csv'),
                                             header='QCT',
                                             index=False)
    kpi_comp['Cluster'].drop_duplicates().to_csv(Path('updated_output','countryData',
                                                      'Cluster.csv'),
                                                 header='Cluster',
                                                 index=False)

    kpi_comp[['Country', 'QCT', 'Cluster'
              ]].drop_duplicates().to_csv(Path('updated_output','countryData',
                                               'Country_data.csv'),
                                          header='Country_data',
                                          index=False)

    kpi_comp[['Country', 'Cluster'
              ]].drop_duplicates().to_csv(Path('updated_output','countryData',
                                               'Country_cluster.csv'),
                                          header='Country_cluster',
                                          index=False)

    kpi_comp[['Country',
              'QCT']].drop_duplicates().to_csv(Path('updated_output','countryData',
                                                    'Country_QCT.csv'),
                                               header='Country_QCT',
                                               index=False)

    return kpi_comp


def historicos(planes,audit_summary,periodos,reader):

    # periodos = ['Jul-22', 'Jun-22']
    secuencia = np.arange(len(periodos), 0, -1)
    

    info_cols=['indice', 'Period', 'Planes', 'E2E']
    info = pd.DataFrame(zip(secuencia, periodos, planes,audit_summary),
                        columns=info_cols)

    kpi_cols = [
        'Period', 'Country', 'QCT', 'Cluster', 'Date_Percent_Cluster',
        'Date_Percent_QCT', 'Date_Percent_Country', 'Auditor_Percent_Cluster',
        'Auditor_Percent_QCT', 'Auditor_Percent_Country',
        'TSO_Percent_Cluster', 'TSO_Percent_QCT', 'TSO_Percent_Country'
    ]

    base_frame = pd.DataFrame(columns=kpi_cols)

    try:

        history_data = pd.read_csv(Path('updated_output', 'historical','history_data.csv'))
        base_frame = pd.concat([base_frame, history_data], ignore_index=True)
        print('-->Encontro archivo archivo de historia previo')
        calculated_periods = history_data['Period'].unique()

    except Exception as e:
        print('algun error:  ', e)
        print('-->No se leyo archivo de historia previo')
        calculated_periods = []


    restantes=info[~info['Period'].isin(calculated_periods)]

    for mes, plan, e2e in restantes[info_cols[1:]].itertuples(index=False):
        print('calculating...: ',mes)

        plan_leido = reader.read_plan(Path('Planes',plan)) # CAUTION Hardcoded
        e2e_leido = reader.read_audit_summary(Path('updated_E2E_input',e2e)) # CAUTION Hardcoded

        kpi_comp = kpi_comp_(plan_leido, e2e_leido)

        kpi_comp = kpi_comp[kpi_cols[1:]].drop_duplicates()
        kpi_comp['Period'] = mes

        kpi_comp = kpi_comp[kpi_cols]

        kpi_comp.reset_index(inplace=True, drop=True)
        base_frame = pd.concat([base_frame, kpi_comp], ignore_index=True)

    base_frame.to_csv(Path('updated_output', 'historical','history_data.csv'), index=False)
    print('-->Importado "history_data.csv"')
    secuencia = info[['indice', 'Period']]
    secuencia.to_csv(Path('updated_output','historical', 'Period_sequence.csv'),
                     index=False)


def main(historico=False):

    # input_reader= InputReader(Path(r'C:\Users\vara9003\Documents\dashboard_update\EMEA'))
    input_reader = InputReader(Path('C:\\Users\\vara9003\\Documents\\dashboard_update'))

    if historico:

        # planes = ['sep', 'ago', 'jul']
        # audit_summary = ['sep', 'ago', 'jul']
        # periodos = ['Sep-22', 'Ago-22', 'Jul-22']

        planes = ['Planes_Diciembre','Ajustados_Nov2','Planes_Oct22','planes_Sep22_br', 'Planes_Agosto']#,'planes_junio' # Todos]
        audit_summary = ['cierre_dic','nov_cierre','oct_cierre','cierre Sep', 'cierre_ago']#,'Cierre June'] # Todos
        periodos = ['Dic-22','Nov-22','Oct-22','Sep-22', 'Ago-22'] #, 'Jun-22'] # Todos

        if len(planes)!=len(audit_summary) or len(planes)!=len(periodos):
            raise ValueError('Error en las carpetas especificadas para leer historicos')

        historicos(planes,audit_summary,periodos,input_reader)


    audit_summary = input_reader.read_plan(Path('Planes','Planes_Diciembre'))
    planes = input_reader.read_audit_summary(Path('updated_E2E_input','cierre_dic'))
    kpi_comp_(planes,audit_summary)





if __name__ == '__main__':
    print('\n->Running script to update dashboard, please wait...')

    main(historico=False)

    excec_time_minutes = round((time.perf_counter() - init)/60,3)
    print("tiempo de ejecucion: ",excec_time_minutes ," mins")
    #
