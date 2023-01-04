# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from pathlib import Path


class InputReader:

    @classmethod
    def convert_legacy_e2e(cls, legacy_e2e):

        header_convert = dict(zip(cls.legacy_e2e_cols, cls.e2e_cols))
        # breakpoint()
        return legacy_e2e.rename(columns=header_convert)

    @staticmethod
    def read_file(file_path, format_):
        if format_ == '.csv':
            file_ = pd.read_csv(file_path, sep='\t', encoding='utf_16_le')

        elif format_ == '.xlsx':
            file_ = pd.read_excel(file_path)

        return file_

    def __init__(self, input_path: Path):
        self.input_path = input_path
        pass

    def read_audit_summary(self, e2e_dir):
        ''' Funcion para leer los Audit Summary en la carpeta con nombre del argumento de funcion
        '''
        work_folder = Path(self.input_path, e2e_dir)

        all_files = os.listdir(work_folder)
        # all_files = filter(lambda x: x.endswith(format_), all_files) # DEPRECATED
        # csv_files = list(csv_files) # DEPRECATED
        # print(csv_files)

        all_csv = []  # placeholder for concatenated Audit Summary
        for f, i in enumerate(all_files):
            print('·Leyendo archivo: ', i)

            format_ = os.path.splitext(i)[1]
            individual_csv = InputReader.read_file(Path(work_folder, i),
                                                   format_)

            actual_e2e_cols = set(individual_csv.columns)

            if set(self.e2e_cols).issubset(actual_e2e_cols):
                legacy = False
            elif set(self.legacy_e2e_cols).issubset(actual_e2e_cols):
                legacy = True
            else:
                raise ValueError(f'{i} file doesnt have valid columns.')

            if legacy:
                individual_csv = InputReader.convert_legacy_e2e(individual_csv)

            ## NOTE
            # Caso Brasil:
            if 76 in individual_csv['COUNTRY_ID'].unique():
                # quita 'Antecipa Compras'
                # breakpoint()
                individual_csv = individual_csv[
                    individual_csv['ACTIVITY_LOCAL_NM'] != 'Antecipa Compras']
            # ### ### ###

            # Filtra solo a columnas necesarias
            indicidual_csv = individual_csv[self.e2e_cols]

            all_csv.append(individual_csv)
            print('·Periodos: ', all_csv[f]['PERIOD_ID'].unique())
            print('--------')

        try:
            all_csv = pd.concat(all_csv, ignore_index=True)
        except:
            print(
                'Por favor verifica que la carpeta "updated_E2E_input" no este vacia'
            )

        # Termina de leer local

        df = all_csv.copy()

        # normaliza tipos de sms id y codigo de auditor:
        tipos = {'SMS_ID': str, 'AUDTIOR_ID': np.int32}
        df = df.astype(tipos)

        # cambia nombre de columnas para join, limpia sms id

        df.rename(columns={
            'PERIOD_GROUP': 'Frequency',
            'AUDTIOR_ID': 'Visit CDAR ID',
            'SMS_ID': 'SMS ID'
        },
                  inplace=True)

        # elimina contratacion, elimina parciales, codigo de pais a nombre:
        df = df[df['SMS ID'].apply(lambda x: len(x)) == 10]
        # elminar numeros a 10 digitos 'negativos'
        df = df[~df['SMS ID'].apply(lambda x: x.startswith('-'))]

        df['VISIT_DATE'].where(df['STORE_STATUS'] != 'PARTIAL',
                               other=' ',
                               inplace=True)

        df['COUNTRY_ID'].replace(to_replace=self.countries, inplace=True)

        df['VISIT_DATE'] = pd.to_datetime(df['VISIT_DATE'].str.replace(
            ' ', ''),
                                          format='%m/%d/%Y',
                                          errors='coerce')

        df.sort_values(by=['SMS ID', 'VISIT_DATE'])
        df = df[~df.duplicated(subset=['SMS ID', 'Frequency'], keep='last')]
        df.to_csv('concat_E2E.csv', index=False)
        print("2) Exportado: 'concat_E2E.csv'")
        return df

    def read_plan(self, plan_dir):
        work_folder = Path(self.input_path, plan_dir)

        all_files = os.listdir(work_folder)
        excel_files = filter(
            lambda x: x.endswith('.xls') or x.endswith('.xlsx'), all_files)
        excel_files = list(excel_files)
        plan_cols = [
            'SMS ID', 'Frequency', 'State', 'City', 'Planned Date',
            'Associate', 'Associate Cdar ID', 'Type', 'Activity Local Name'
        ]

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

        all_plan = all_plan[~all_plan.duplicated(
            subset=['SMS ID', 'Frequency'
                    ], keep='last')]  # quita primeras fechas
        all_plan.to_csv('concatenado_planes.csv', index=False)
        print("1) Exportado: 'concatenado_planes.csv'")

        return all_plan

    # Class Variables:
    # a.k.a. Info I couldnt find a better place to store

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
        682: 'Arabia Saudita',
        414: 'Kuwait',  # aun no migra a CIP, mensual
        634: 'Qatar'  # aun no migra a CIP
    }

    e2e_cols = [
        'COUNTRY_ID', 'PERIOD_GROUP', 'PERIOD_ID', 'CLUSTER_NM', 'AUDTIOR_ID',
        'SUPERVISOR_NM', 'STORE_STATUS', 'SMS_ID', 'VISIT_DATE', 'ELAPSED_DAYS'
    ]

    legacy_e2e_cols = [
        'Country Id', 'Period Group', 'Period ID', 'Cluster Name',
        'Resource CDAR_ID', 'Supervisor Name', 'Store status', 'SMS ID',
        'Visit Audit date', 'Elapsed Days'
    ]

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
        # 2023
        'MONT_1258': (28, 34),  #'ene',
        'MONT_1262': (25, 31),  #'feb',
        'MONT_1266': (28, 34),  #'mar',
        'MONT_1271': (27, 33),  #'abr',
        'MONT_1275': (28, 34),  #'may',
        'MONT_1279': (27, 33),  #'jun',
        'MONT_1284': (28, 34),  #'jul',
        'MONT_1288': (28, 34),  #'ago',
        'MONT_1292': (27, 33),  #'sep',
        'MONT_1297': (28, 34),  #'oct',
        'MONT_1301': (27, 33),  #'nov',
        'MONT_1306': (28, 34),  #'dic',

        # 2023 - Bi
        'FOOD_1258': (57, 67),  #'ene'
        'FOOD_1266': (54, 64),  #'mar'
        'FOOD_1275': (56, 66),  #'may'
        'FOOD_1284': (56, 66),  #'jul'
        'FOOD_1292': (56, 66),  #'sep'
        'FOOD_1301': (56, 66),  #'nov'
        'DRUG_1262': (54, 64),  #'feb'
        'DRUG_1271': (56, 66),  #'abr'
        'DRUG_1279': (56, 66),  #'jun'
        'DRUG_1288': (57, 67),  #'ago'
        'DRUG_1297': (56, 66),  #'oct'
        'DRUG_1206': (56, 66),  #'dic'

        # CAM 2023
        #
        'CRMO_1258': (28, 34),  #'ene',
        'GTMO_1258': (28, 34),  #'ene',
        'HNMO_1258': (28, 34),  #'ene',
        'NIMO_1258': (28, 34),  #'ene',
        'PAMO_1258': (28, 34),  #'ene',
        'SVMO_1258': (28, 34),  #'ene',

        #
        'CRMO_1262': (25, 31),  #'feb',
        'GTMO_1262': (25, 31),  #'feb',
        'HNMO_1262': (25, 31),  #'feb',
        'NIMO_1262': (25, 31),  #'feb',
        'PAMO_1262': (25, 31),  #'feb',
        'SVMO_1262': (25, 31),  #'feb',

        #
        'CRMO_1266': (28, 34),  #'mar',
        'GTMO_1266': (28, 34),  #'mar',
        'HNMO_1266': (28, 34),  #'mar',
        'NIMO_1266': (28, 34),  #'mar',
        'PAMO_1266': (28, 34),  #'mar',
        'SVMO_1266': (28, 34),  #'mar',


        # CAM 2022
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

### Kate text editor alert syntax
# ALERT
# ATTENTION
# DANGER
# HACK
# SECURITY
# BUG
# FIXME
# DEPRECATED
# TASK
# TODO
# TBD
# WARNING
# CAUTION
# NOLINT
# ###
# NOTE
# NOTICE
# TEST
# TESTING
#
# TODO msg
# TODO <- NO-BREAK SPACE (nbsp)
