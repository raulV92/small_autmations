# -*- coding: utf-8 -*-
"""
* Version: 08-Dic-2022
* TODO Agregar avance de auditoria por Auditor
* TODO Lineas con auditores en varios clusters

EMEA DASHBOARD

"""

import time

init = time.perf_counter()

import pretty_errors

import os
import pandas as pd
import numpy as np

from pathlib import Path


def read_plan(plan_dir):
    ## **************** Folder Planes *********************************
    work_folder = Path('Planes', plan_dir)
    # work_folder= 'Planes'
    all_files = os.listdir(work_folder)
    excel_files = filter(lambda x: x.endswith('.xls') or x.endswith('.xlsx'),
                         all_files)
    excel_files = list(excel_files)
    plan_cols = [
        'SMS ID', 'Frequency', 'State', 'City', 'Planned Date', 'Associate',
        'Associate Cdar ID', 'Type', 'Activity Local Name'
    ]
    # selecciona carpeta de planes
    '''
    for i in excel_files:
        try:
        #pass
            print(i)
            pl=pd.read_excel(Path(work_folder,i),usecols=plan_cols)
            pl.drop_duplicates()

            #print(pl['CLUSTER_NM'].unique())
            #print(pl['Activity Local Name'].unique())

            #pl=pl[pl['Type']=='ASSIGNED']
            # tipos={'SMS ID':np.int64,'Associate Cdar ID':np.int32}

            # normaliza tipos de sms id y codigo de auditor:
            tipos = {'SMS_ID': str, 'AUDTIOR_ID': np.int32}
            breakpoint()
            df = df.astype(tipos)

            pl=pl.astype(tipos)
            pl=pl[pl['SMS ID']>10**9]

            pl['Planned Date']=pd.to_datetime(pl['Planned Date'],format='%m/%d/%Y')
        except Exception as e:
            print(i,'----------------------------------------------------')
            print(e)


    print('archivos individuales')
    breakpoint()'''

    all_plan = [
        pd.read_excel(Path(work_folder, i), usecols=plan_cols)
        for i in excel_files
    ]

    #breakpoint()
    ## Termina de leer archivo en local

    # eliminar dups ->quitar unassign ->convertir a fecha ->ordenar por fecha
    all_plan = pd.concat(all_plan, ignore_index=True).drop_duplicates()
    all_plan = all_plan[all_plan['Type'] == 'ASSIGNED']

    tipos = {'SMS ID': str, 'Associate Cdar ID': np.int32}
    all_plan = all_plan.astype(tipos)

    # df=df[df.A.apply(lambda x: len(str(x))==10]
    all_plan = all_plan[all_plan['SMS ID'].apply(lambda x: len(x)) == 10]
    all_plan['Planned Date'] = pd.to_datetime(all_plan['Planned Date'],
                                              format='%m/%d/%Y',
                                              errors='raise')
    all_plan.sort_values(by=['SMS ID', 'Planned Date'])

    all_plan = all_plan[
        ~all_plan.duplicated(subset=['SMS ID', 'Frequency'], keep='last')] # quita primeras fechas
    all_plan.to_csv('concatenado_planes.csv', index=False)
    print("1) Exportado: 'concatenado_planes.csv'")
    return all_plan


def read_e2e(e2e_dir):
    ''' Funcion para leer los Audit Summary en la carpeta con nombre del argumento de funcion
    '''
    work_folder = Path('updated_E2E_input', e2e_dir)

    all_files = os.listdir(work_folder)
    csv_files = filter(lambda x: x[-4:] == '.csv', all_files)
    csv_files = list(csv_files)
    # print(csv_files)

    #try:
    #all_csv=[pd.read_csv(Path(work_folder,i),sep='\t',encoding='utf_16_le') for i in csv_files]

    #except:
    #    print('Por favor verifica que la carpeta "updated_E2E_input" exista')

    all_csv = []
    for f, i in enumerate(csv_files):
        print('Leyendo archivo: ', i)

        individual_csv = pd.read_csv(Path(work_folder, i), sep='\t', encoding='utf_16_le')

        ## NOTE
        # Caso Brasil:
        if 76 in individual_csv['COUNTRY_ID'].unique():
            # quita 'Antecipa Compras'
            individual_csv = individual_csv[individual_csv['ACTIVITY_LOCAL_NM']!='Antecipa Compras']

        ############
        all_csv.append(individual_csv)
        print('Periodos: ', all_csv[f]['PERIOD_ID'].unique())

    try:
        all_csv = pd.concat(all_csv, ignore_index=True)
    except:
        print(
            'Por favor verifica que la carpeta "updated_E2E_input" no este vacia'
        )

    # Termina de leer local

    countries = {
        32: 'Argentina',
        858: 'Uruguay',
        484: 'Mexico',
        76: 'Brasil',
        68: 'Bolivia',
        152: 'Chile',
        218: 'Ecuador',
        170: 'Colombia',
        862: 'Venezuela',
        222: 'El Salvador',
        320: 'Guatemala',
        340: 'Honduras',
        558: 'Nicaragua',
        591: 'Panama',
        600: 'Paraguay',
        604: 'Peru',
        630: 'Puerto Rico',
        214: 'Republica Dominicana',
        188: 'Costa Rica',
        682: 'KSA'
    }

    e2e_cols = [
        'COUNTRY_ID', 'PERIOD_GROUP', 'PERIOD_ID', 'CLUSTER_NM', 'AUDTIOR_ID',
        'SUPERVISOR_NM', 'STORE_STATUS', 'SMS_ID', 'VISIT_DATE', 'ELAPSED_DAYS'
    ]

    df = all_csv[e2e_cols].copy()

    # normaliza tipos de sms id y codigo de auditor:
    tipos = {'SMS_ID': str, 'AUDTIOR_ID': np.int32}
    # breakpoint()
    # df = df.astype(tipos)

    # cambia nombre de columnas para join, limpia sms id

    df.rename(columns={
        'PERIOD_GROUP': 'Frequency',
        'AUDTIOR_ID': 'Visit CDAR ID',
        'SMS_ID': 'SMS ID'
    },
              inplace=True)

    # elimina contratacion, elimina parciales, codigo de pais a nombre:
    # df = df[df['SMS ID'].apply(lambda x: len(x)) == 10]

    # elimina contratacion, elimina parciales, codigo de pais a nombre:
    df = df[df['SMS ID'].apply(lambda x: len(str(x))) == 10]
    # elminar numeros a 10 digitos 'negativos'
    df = df[~df['SMS ID'].apply(lambda x: str(x).startswith('-'))]

    df['VISIT_DATE'].where(df['STORE_STATUS'] != 'PARTIAL',
                           other=' ',
                           inplace=True)

    df['COUNTRY_ID'].replace(to_replace=countries, inplace=True)

    df['VISIT_DATE'] = pd.to_datetime(df['VISIT_DATE'].str.replace(' ', ''),
                                      format='%m/%d/%Y',
                                      errors='coerce')

    df.sort_values(by=['SMS ID', 'VISIT_DATE'])
    df = df[~df.duplicated(subset=['SMS ID', 'Frequency'], keep='last')]
    df.to_csv('concat_E2E.csv', index=False)
    print("2) Exportado: 'concat_E2E.csv'")
    return df


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


def tso_comp(row):
    tso_ranges = {
        'MONT_1206': (28, 34),  #'ene',
        'MONT_1210': (25, 31),  #'feb',
        'MONT_1214': (28, 34),  #'mar',
        'MONT_1218': (27, 33),  #'abr',
        'MONT_1223': (28, 34),  #'may',
        'MONT_1227': (27, 33),  #'jun',
        'MONT_1232': (28, 34),  #'jul',
        'MONT_1236': (28, 34),  #'ago',
        'MONT_1240': (27, 33),  #'sep',
        'MONT_1245': (28, 34),  #'oct',
        'MONT_1249': (27, 33),  #'nov',
        'MONT_1253': (28, 34),  #'dic',
        #
        'CRMO_1227': (27, 33),  #'jun',
        'GTMO_1227': (27, 33),  #'jun',
        'HNMO_1227': (27, 33),  #'jun',
        'NIMO_1227': (27, 33),  #'jun',
        'PAMO_1227': (27, 33),  #'jun',
        'SVMO_1227': (27, 33),  #'jun',
        #
        'CRMO_1232': (28, 34),  #'jul',
        'GTMO_1232': (28, 34),  #'jul',
        'HNMO_1232': (28, 34),  #'jul',
        'NIMO_1232': (28, 34),  #'jul',
        'PAMO_1232': (28, 34),  #'jul',
        'SVMO_1232': (28, 34),  #'jul',
        #
        'CRMO_1236': (28, 34),  #'ago',
        'GTMO_1236': (28, 34),  #'ago',
        'HNMO_1236': (28, 34),  #'ago',
        'NIMO_1236': (28, 34),  #'ago',
        'PAMO_1236': (28, 34),  #'ago',
        'SVMO_1236': (28, 34),  #'ago',
        #
        'CRMO_1240': (27, 33),  #'sep',
        'GTMO_1240': (27, 33),  #'sep',
        'HNMO_1240': (27, 33),  #'sep',
        'NIMO_1240': (27, 33),  #'sep',
        'PAMO_1240': (27, 33),  #'sep',
        'SVMO_1240': (27, 33),  #'sep',
        #
        'CRMO_1245': (28, 34),  #'oct',
        'GTMO_1245': (28, 34),  #'oct',
        'HNMO_1245': (28, 34),  #'oct',
        'NIMO_1245': (28, 34),  #'oct',
        'PAMO_1245': (28, 34),  #'oct',
        'SVMO_1245': (28, 34),  #'oct',
        #
        'CRMO_1249': (27, 33),  #'nov',
        'GTMO_1249': (27, 33),  #'nov',
        'HNMO_1249': (27, 33),  #'nov',
        'NIMO_1249': (27, 33),  #'nov',
        'PAMO_1249': (27, 33),  #'nov',
        'SVMO_1249': (27, 33),  #'nov',
        #
        'CRMO_1253': (28, 34),  #'dic',
        'GTMO_1253': (28, 34),  #'dic',
        'HNMO_1253': (28, 34),  #'dic',
        'NIMO_1253': (28, 34),  #'dic',
        'PAMO_1253': (28, 34),  #'dic',
        'SVMO_1253': (28, 34),  #'dic',
        # enero 2023:
        'FOOD_1258': (57, 67),  #'ene'
        #
        'FOOD_1206': (57, 67),  #'ene'
        'FOOD_1214': (54, 64),  #'mar'
        'FOOD_1223': (56, 66),  #'may'
        'FOOD_1232': (56, 66),  #'jul'
        'FOOD_1240': (56, 66),  #'sep'
        'FOOD_1249': (56, 66),  #'nov'
        #
        'DRUG_1210': (54, 64),  #'feb'
        'DRUG_1218': (56, 66),  #'abr'
        'DRUG_1227': (56, 66),  #'jun'
        'DRUG_1236': (57, 67),  #'ago'
        'DRUG_1245': (56, 66),  #'oct'
        'DRUG_1253': (56, 66)  #'dic'
    }

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


def main(planes, audit_summary):

    plan = read_plan(planes)
    raw_e2e = read_e2e(audit_summary)

    extend_plan = pd.merge(plan,
                           raw_e2e,
                           on=['SMS ID', 'Frequency'],
                           indicator=True)

    plan.to_csv('plan_prog.csv')
    raw_e2e.to_csv('e2e_prog.csv')
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

    extend_plan['tso_compliance'] = extend_plan.apply(tso_comp, axis=1)

    # print(extend_plan.columns);breakpoint()
    extend_plan[cols_output[:-1]+['tso_compliance']].to_csv(Path('updated_output',
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
        'updated_output', 'tabla_true_false.csv'),
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
    for i in auditors:
        #if i[0]=='VANESSA KATTY PEREZ TECHERA':
        #    breakpoint()
        #    pass
        country = list(i[1]['COUNTRY_ID'].unique())
        # en caso de un auditor con mas de 1 QCT...
        if len(i[1]['SUPERVISOR_NM'].unique()) > 1:
            qct = list(i[1]['SUPERVISOR_NM'].unique()[0:1])  #
        else:
            qct = list(i[1]['SUPERVISOR_NM'].unique())
        #elif len(i[1]['SUPERVISOR_NM'].unique())>1:
        #    qct=list(i[1]['SUPERVISOR_NM'].unique()[0]) #

        cluster = list(i[1]['CLUSTER_NM'].unique())
        auditor = [i[0]]

        # compliance_count=i[1]['date_comp_aud'].value_counts().values.tolist()
        date_comp_count = [
            count_comp(i[1]['date_compliance'], True),
            count_comp(i[1]['date_compliance'], False)
        ]

        auditor_comp_count = [
            count_comp(i[1]['auditor_compliance'], True),
            count_comp(i[1]['auditor_compliance'], False)
        ]

        tso_comp_count = [
            count_comp(i[1]['tso_compliance'], True),
            count_comp(i[1]['tso_compliance'], False)
        ]

        #if i == 'SANTIAGO CALLE':
        #    pass
        #breakpoint()
        new_row = country + qct + cluster + auditor + date_comp_count + auditor_comp_count + tso_comp_count
        if len(new_row) != 10:

            print('Error en columnas:')
            for i, j in zip(kpi_comp_columns, new_row):
                print('\t', i, '\t', j)
        else:
            kpi_comp.append(new_row)

        #if 'AMANDA VERONICA' in i[0]:
        #    breakpoint()
    #breakpoint()
    #res=pd.DataFrame([filter_e2e['date_comp_aud'].value_counts().T,
    #                  filter_e2e['date_comp_aud'].value_counts(normalize=True).T])

    # noombres correspondinetes con funcion add_group_cols
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
        kpi_comp.to_csv(Path('updated_output', 'kpi_comp.csv'),
                        index=False)
        print("5) Exportado: 'kpi_comp.csv' <---> Subir a Sharepoint")
    except:
        print(
            'Por favor verifica que el archivo "final_output.csv" no este abierto'
        )

    kpi_comp['Country'].drop_duplicates().to_csv(Path('updated_output',
                                                      'Countries.csv'),
                                                 header='Country',
                                                 index=False)
    kpi_comp['QCT'].drop_duplicates().to_csv(Path('updated_output', 'QCT.csv'),
                                             header='QCT',
                                             index=False)
    kpi_comp['Cluster'].drop_duplicates().to_csv(Path('updated_output',
                                                      'Cluster.csv'),
                                                 header='Cluster',
                                                 index=False)

    kpi_comp[['Country', 'QCT', 'Cluster'
              ]].drop_duplicates().to_csv(Path('updated_output',
                                               'Country_data.csv'),
                                          header='Country_data',
                                          index=False)

    kpi_comp[['Country', 'Cluster'
              ]].drop_duplicates().to_csv(Path('updated_output',
                                               'Country_cluster.csv'),
                                          header='Country_cluster',
                                          index=False)

    kpi_comp[['Country',
              'QCT']].drop_duplicates().to_csv(Path('updated_output',
                                                    'Country_QCT.csv'),
                                               header='Country_QCT',
                                               index=False)

    return kpi_comp


def historicos(planes,audit_summary,periodos):

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

        history_data = pd.read_csv(Path('updated_output', 'history_data.csv'))
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
        kpi_comp = main(plan, e2e)

        kpi_comp = kpi_comp[kpi_cols[1:]].drop_duplicates()
        kpi_comp['Period'] = mes

        kpi_comp = kpi_comp[kpi_cols]

        kpi_comp.reset_index(inplace=True, drop=True)
        base_frame = pd.concat([base_frame, kpi_comp], ignore_index=True)

    base_frame.to_csv(Path('updated_output', 'history_data.csv'), index=False)
    print('-->Importado "history_data.csv"')
    secuencia = info[['indice', 'Period']]
    secuencia.to_csv(Path('updated_output', 'Period_sequence.csv'),
                     index=False)


if __name__ == '__main__':
    print('Running script to update dashboard...')

    planes_ = 'Planes_Diciembre' # Carpeta donde se guarda los planes current
    audit_summary_ = 'cierre_dic' # Carpeta de E2E Current


    planes = [

        'Ajustados_Nov2','Planes_Oct22','planes_Sep22_br', 'Planes_Agosto', 'planes_july'#,'planes_junio' # Todos
        # 'planes_july',
    ]
    audit_summary = ['nov_cierre','oct_cierre','cierre Sep', 'cierre_ago', 'Cierre Julio']#,'Cierre June'] # Todos
    # audit_summary = ['Cierre Julio', 'Cierre June']
    periodos = ['Nov-22','Oct-22','Sep-22', 'Ago-22', 'Jul-22']#, 'Jun-22'] # Todos

    # historicos(planes,audit_summary,periodos)
    kpi_comp = main(planes_, audit_summary_)


    print("tiempo de ejecucion: ", time.perf_counter() - init)
    # input('input standby...')
