import logging
from route_calculator.data import data_loader
import os
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import time


class Data:
    def __init__(self, path_to_data, load_from='csv', plandate=None):
        if load_from == 'csv':
            data = data_loader.load_data_from_csv(path_to_data)
        elif load_from == 'excel':
            # TODO Вынести структуру файлов в отдельную функцию загрузки
            # data_dir = os.path.join("..", "data_examples")
            # excel_filepath = os.path.join(data_dir, "Воронеж.xlsx")
            # market_market_filepath = os.path.join(data_dir, "матрица расстояний от магазина 1.xlsx")
            # dc_market_filepath = os.path.join(data_dir, "матрица расстояний от склада.xlsx")
            # data = data_loader.read_excel(excel_filepath=excel_filepath,
            #                   market_market_filepath=market_market_filepath,
            #                   dc_market_filepath=dc_market_filepath)
            data_dir = os.path.join("..", "data_examples")

            excel_filepath = os.path.join(data_dir, "Тюмень.xlsx")
            market_market_filepath = os.path.join(data_dir, "матрица расстояний от магазина.xlsx")
            dc_market_filepath = os.path.join(data_dir, "матрица расстояний от склада.xlsx")
            parking_market_filepath = os.path.join(data_dir, "матрица расстояний от стоянки.xlsx")
            data = data_loader.read_excel(excel_filepath=excel_filepath,
                                          market_market_filepath=market_market_filepath,
                                          dc_market_filepath=dc_market_filepath,
                                          parking_market_filepath=parking_market_filepath)

            data = data_loader.prepare_data(data, datetime(2019, 4, 8, 0, 0), vehicle_params={"ГВМ малоформата": 16})

        self.plandate = plandate

        # Данные по РЦ
        self.store = data['store']#.sort_values(by='order_id', ascending=False)[:20]
        # self.store_enrichment()

        # Данные по категориям
        self.category = data["category"]

        # Данные по заказам
        self.order = data["order"]
        self.order_enrichment()

        # Данные по ТС
        # TODO Необходимо обогатить кол-во ТС возможностью расцеплять сцепки и считать их отдельными
        # TODO При расчете необходимо будет исключать по Г/н
        self.vehicle = data["vehicle"]
        self.vehicle_enrichment()

        # Данные по расстояниям М-М
        self.market_market_dist = data["market_market_dist"]
        self.market_market_enrichment()

        # Данные по расстояниям РЦ-М
        self.rc_market_dist = data["rc_market_dist"]
        self.dc_market_enrichment()

        # Данные по расстояниям ПАРКОВКА-М
        self.parking_market_dist = data["parking_market_dist"]
        self.parking_market_enrichment()

        self.plandate_order_volume = self.calculate_plandate_order_volume()
        self.calculate_store_windows_capacity_on_plandate()

    def order_enrichment(self):
        self.order['Время выгрузки, темп (мин)'] = \
            self.order.apply(lambda x: x['Темп, заказ, пм'] * x['Выгр реф (в минутах)'], 1)

        self.order['Время выгрузки, нетемп (мин)'] = \
            self.order.apply(lambda x: x['Не темп, заказ, пм'] * x['Выгр сух (в минутах)'], 1)

        self.order['Время выгрузки (мин)'] = \
            self.order.apply(lambda x: x['Время выгрузки, темп (мин)'] + x['Время выгрузки, нетемп (мин)'], 1)

        # TODO добавить нормативы по другим операциям, маневрирование и т.д.
        self.order['Время обслуживания'] = \
            self.order.apply(lambda x: x['Время выгрузки (мин)'] +
                                       x['Время выгрузки, нетемп (мин)'], 1)

        self.order['Основная часть'] = 1

    def market_market_enrichment(self):
        """
        Функция обогащает данные по плечам магазин-магазин
        :return:
        """
        self.add_RTO(self.market_market_dist, 'ret_time')

    # def store_enrichment(self):
    #     ADDITIONAL_PREPARATION_TIME = 30
    #     self.store['Мин время отправления'] = self.store['Дата и время отгрузки до'] + \
    #                                           timedelta(minutes=ADDITIONAL_PREPARATION_TIME)

    def dc_market_enrichment(self):
        """
        Функция обогащает данные по плечам РЦ-магаз
        :return:
        """

        self.add_RTO(self.rc_market_dist, 'ret_time')

    def parking_market_enrichment(self):
        """
        Функция обогащает данные по плечам ПАРКТОВКА-магаз
        :return:
        """

        self.add_RTO(self.parking_market_dist, 'ret_time')

    def vehicle_enrichment(self):
        """
        Функция добавляет в vehicles дополнительную запись для каждого ТС, которое можно расцепить
        и использовать как одиночку, также добавляет поле "Расцеплен" со значениями 0 и 1 (не расцеплен/расцеплен)
        :return:
        """

        loc_vehicles = self.vehicle
        loc_vehicles["Заполнен пм, темп приц"] = 0
        loc_vehicles["Заполнен пм, не темп приц"] = 0
        loc_vehicles["Заполнен пм, темп тяг"] = 0
        loc_vehicles["Заполнен пм, не темп тяг"] = 0
        loc_vehicle_no_trailer = loc_vehicles[(loc_vehicles['ГВМ приц'] != 0) & (loc_vehicles['ГВМ тяг'] != 0)].copy()
        loc_vehicle_no_trailer['ГВМ приц'] = 0

        loc_vehicles['Расцеплен'] = 0
        loc_vehicle_no_trailer['Расцеплен'] = 1

        loc_vehicle_no_trailer["Модель приц"] = 0
        loc_vehicle_no_trailer["ГВМ пм"] = loc_vehicle_no_trailer['ГВМ тяг']

        self.vehicle = loc_vehicles.append(loc_vehicle_no_trailer)
        self.vehicle.reset_index(drop=True, inplace=True)

        return self.vehicle

    def add_RTO(self, arc_df, duration_column_name, inplace=True):
        """
        Функция добавляет два поля времени на основе времени на маршруте, учитывая нормы РТО и МО
        Нормы:
            Если это первое плечо маршрута:
                Если путь на маршруте больше 4 часов, то положен отдых 30 мин
            Если второе или последующее:
                Если путь на маршруте больше 2 часов, то положен отдых 15 минут каждые 2 часа
        :return:
        """

        if inplace:
            loc_arc = arc_df
        else:
            loc_arc = arc_df.copy()

        MINUTES_IN_4_HOURS = 4 * 60
        MINUTES_IN_2_HOURS = 2 * 60
        FIRST_REST_DUR = 30
        SECOND_REST_DUR = 15

        # Считаем, насколько длительность плеча больше 4 часов
        loc_arc['RTO1_duration'] = loc_arc['ret_time'] - MINUTES_IN_4_HOURS

        # Если первое плечо больше 4 часов, то добавляем 30 мин к длительности + 15 минут каждые два часа на отдых
        loc_arc.loc[loc_arc['RTO1_duration'] > 0, 'RTO1_duration'] = \
            loc_arc['ret_time'] + \
            FIRST_REST_DUR + \
            np.floor(loc_arc['RTO1_duration'] / MINUTES_IN_2_HOURS) * SECOND_REST_DUR

        # Если первое плечо меньше 4 часов, то просто оставляем длительность как есть
        loc_arc.loc[loc_arc['RTO1_duration'] <= 0, 'RTO1_duration'] = loc_arc['ret_time']

        # Считаем, какая будет длительность плеча, если ранее по маршруту было плечо длинее 4 часов
        loc_arc['RTO2_duration'] = loc_arc['ret_time'] + \
                                   (np.floor(loc_arc['ret_time'] / MINUTES_IN_2_HOURS) * SECOND_REST_DUR)

        if not inplace:
            return loc_arc

    def df_crossjoin(self, df1, df2, **kwargs):
        """
        Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
        Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
        See: https://github.com/pydata/pandas/issues/5401
        :param df1 dataframe 1
        :param df1 dataframe 2
        :param kwargs keyword arguments that will be passed to pd.merge()
        :return cross join of df1 and df2
        """
        df1['_tmpkey'] = 1
        df2['_tmpkey'] = 1

        res = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)

        df1.drop('_tmpkey', axis=1, inplace=True)
        df2.drop('_tmpkey', axis=1, inplace=True)

        return res

    @property
    def orders_on_plandate(self):
        return self.order[self.order["Дата разнарядки"] == self.plandate]

    @property
    def vehicles_on_plandate(self):
        return self.vehicle[self.vehicle["Дата разнарядки"] == self.plandate]

    @property
    def rule_order_vehicle_temp_type(self):
        """
        Таблица допустимых ТС по типу темп/нетемп для заказов (ТЗ у2.1)
        Правила:
            Темп заказ -> ТС реф
            Нетемп заказ -> ТС реф/изотерм
            Темп + Нетемп -> ТС реф
        :return: Соотвествие между кодом магазина и Г/н ТС, которое может выполнить заказ этого магазина
        """
        loc_order = self.order[['order_id', 'Код ММ', 'Темп, заказ, пм', 'Не темп, заказ, пм']].copy()
        loc_vehicle = self.vehicle[['Г/н ТС', 'Тип утепления']].copy()
        rule_df = self.df_crossjoin(loc_order, loc_vehicle)

        # Считаем, что тип утепления тягача совпадает с типом утепления прицепа

        def rule(rule_row):
            if rule_row['Тип утепления'] == 'рефрижератор':
                return True
            if rule_row['Тип утепления'] == 'термос':
                if rule_row['Темп, заказ, пм'] == 0:
                    return True
            return False

        rule_df = rule_df[rule_df.apply(rule, axis=1)]
        return rule_df[['order_id', 'Код ММ', 'Г/н ТС']].drop_duplicates()

    @property
    def shops_on_plandate(self):
        return self.orders_on_plandate['Код ММ'].drop_duplicates()

    @property
    def rule_shop_vehicle_compatibility(self) -> pd.DataFrame:
        """
        Таблица допустимых ТС по типу подъездности к магазину (ТЗ у.2.2) ['Код ММ', 'Г/н ТС']
        :return:['Код ММ', 'Г/н ТС']
        """
        loc_vehicle = self.vehicle[['Г/н ТС', 'Модель тяг', 'Модель приц']]
        loc_order = self.order[['Код ММ', 'ИД категории ММ']]
        loc_category = self.category

        rule_df = pd.merge(loc_order, loc_category, left_on=['ИД категории ММ'],
                           right_on=['Идентификатор категории ОД'])
        rule_df = pd.merge(rule_df, loc_vehicle,
                           left_on=['Идентификатор модели тягача', 'Идентификатор модели прицепа'],
                           right_on=['Модель тяг', 'Модель приц'])

        return rule_df[['Код ММ', 'Г/н ТС']].drop_duplicates()

    # @property
    def rule_vehicle_load_interval_compatibility(self, store=None, vehicle=None):
        """
        Сопоставляет интервалам отгрузки склада ТС с учетом времени готовности
        :return:
        """
        if store is None:
            store = self.store
        if vehicle is None:
            vehicle = self.vehicle

        loc_store = store[["Дата и время отгрузки до", "Дата и время отгрузки от",
                                "Минимальное время отправления", "Минимальное время отправления (мин)"]].copy()

        loc_vehicle = vehicle[['Г/н ТС', 'Дата и время готовности ТС']].copy()
        loc_vehicle = loc_vehicle.drop_duplicates()  # Т.к номера будут дублироваться для расцепленных

        rule_df = self.df_crossjoin(loc_vehicle, loc_store)
        rule_df['Ожидание'] = rule_df["Дата и время отгрузки до"] - rule_df["Дата и время готовности ТС"]

        def rule(rule_row):
            if rule_row['Дата и время готовности ТС'] < rule_row["Дата и время отгрузки от"]:
                return True
            return False

        rule_df = rule_df[rule_df.apply(rule, axis=1)]
        return rule_df

    @property
    def rule_dc_market_compatibility(self):
        """
        Таблица плеч РЦ-Магазин с наборов дополнительных параметров
        :return:
        """

        loc_dc_market = self.rc_market_dist.copy()

        loc_store = self.store[['Дата и время отгрузки до', 'Минимальное время отправления',
                                'Минимальное время отправления (мин)']].copy()
        rule_df = self.df_crossjoin(loc_dc_market, loc_store)

        loc_order = self.order[['order_id', 'Код ММ', 'ИД категории ММ',
                                'Строго ГПД', 'Начало ГПД (в минутах)', 'Окончание ГПД (в минутах)',
                                'Можно бить', 'Время обслуживания', 'Срок',
                                'Темп, заказ, пм', 'Не темп, заказ, пм', 'Весь заказ, пм',
                                'Перегруз (в минутах)', 'Подъезд (в минутах)', 'Отъезд (в минутах)',
                                'Оставить прицеп (в минутах)', 'Забрать прицеп (в минутах)']]
        rule_df = pd.merge(rule_df, loc_order,
                           left_on=['to_code'],
                           right_on=['Код ММ'],
                           suffixes=['', '_М'])

        rule_df.rename(columns=dict(zip(loc_order.columns, [x + '_М' for x in loc_order.columns])), inplace=True)

        rule_df['Мин прибытие к М'] = rule_df['Минимальное время отправления (мин)'] + rule_df['ret_time']
        rule_df['Мин время отправления_М'] = rule_df['Начало ГПД (в минутах)_М'] + rule_df['Время обслуживания_М']
        rule_df['Макс время отправления_РЦ'] = rule_df['Окончание ГПД (в минутах)_М'] - rule_df['ret_time']
        rule_df['Мин время отправления_РЦ'] = rule_df['Начало ГПД (в минутах)_М'] - rule_df['ret_time']

        rule_df['cat1'] = 0 # Категория заказа в точке 1 = 0 для РЦ
        rule_df['overlap'] = 0 # Доп поле для комбинирования с market_market_compatibility
        rule_df['wait_time'] = 0 # Доп поле для комбинирования с market_market_compatibility

        # Доп поле для комбинирования с market_market_compatibility. Здесь можно учитывать сервисное время РЦ 30 мин
        rule_df['Время обслуживания_РЦ'] = 0
        return rule_df[['order_id_М', 'from_code', 'to_code', 'cat1', 'ИД категории ММ_М',
                        'Мин время отправления_М', 'Макс время отправления_РЦ', 'Мин время отправления_РЦ',
                        'Начало ГПД (в минутах)_М', 'Окончание ГПД (в минутах)_М', 'ret_time',
                        'overlap', 'wait_time', 'Время обслуживания_РЦ', 'Время обслуживания_М', 'Срок_М',
                        'Темп, заказ, пм_М', 'Не темп, заказ, пм_М', 'Весь заказ, пм_М', 'Можно бить_М', 'ret_distance',
                        'Перегруз (в минутах)_М', 'Подъезд (в минутах)_М', 'Отъезд (в минутах)_М',
                        'Оставить прицеп (в минутах)_М', 'Забрать прицеп (в минутах)_М'
                        ]]

    @property
    def rule_market_market_compatibility(self):
        """
        Таблица совместимых магазинов для последовательной доставки
        :return: Датафрейм с набором параметров, который позволяет принять решение о совместимости
        """
        loc_market_market_dist = self.market_market_dist
        loc_order = self.order[['Код ММ', 'ИД категории ММ',
                                'Строго ГПД', 'Начало ГПД (в минутах)', 'Окончание ГПД (в минутах)',
                                'Можно бить', 'Время обслуживания', 'Срок',
                                'Темп, заказ, пм', 'Не темп, заказ, пм', 'Весь заказ, пм',
                                'Перегруз (в минутах)', 'Подъезд (в минутах)', 'Отъезд (в минутах)',
                                'Оставить прицеп (в минутах)', 'Забрать прицеп (в минутах)'
                                ]]

        rule_df = pd.merge(loc_market_market_dist, loc_order,
                           left_on=['from_code'],
                           right_on=['Код ММ'],
                           suffixes=['', '_М1'])
        rule_df.rename(columns=dict(zip(loc_order.columns, [x + '_М1' for x in loc_order.columns])), inplace=True)

        rule_df = pd.merge(rule_df, loc_order,
                           left_on=['to_code'],
                           right_on=['Код ММ'],
                           suffixes=['', '_М2'])
        rule_df.rename(columns=dict(zip(loc_order.columns, [x + '_М2' for x in loc_order.columns])), inplace=True)

        # TODO Пересчитать ret_time с учетом РТО

        # Время прибытия ко второму магазину
        rule_df['Мин прибытие к М2'] = rule_df['Начало ГПД (в минутах)_М1'] + rule_df['Время обслуживания_М1'] + \
                                       rule_df['ret_time']

        rule_df['Макс прибытие к М2'] = rule_df['Окончание ГПД (в минутах)_М1'] + rule_df['Время обслуживания_М1'] + \
                                        rule_df['ret_time']

        # Начало границы прибытия ко второму магазину
        rule_df['o_tw'] = rule_df[['Мин прибытие к М2', 'Начало ГПД (в минутах)_М2']].max(axis=1)

        # Конец границы прибытия ко Дата и время готовности ТСвторому магазину
        rule_df['c_tw'] = rule_df[['Макс прибытие к М2', 'Окончание ГПД (в минутах)_М2']].min(axis=1)

        # Перекрытие окна ГПД М2
        rule_df['overlap'] = rule_df['c_tw'] - rule_df['o_tw']

        # Время ожидания
        rule_df['wait_time'] = rule_df['Начало ГПД (в минутах)_М2'] - rule_df['Мин прибытие к М2']

        # Обновление данных согласно соответствию времени прибытия в М2 и ГПД М2

        rule_df.loc[
            rule_df['Макс прибытие к М2'] > rule_df['Окончание ГПД (в минутах)_М2'], ['o_tw', 'c_tw', 'wait_time',
                                                                                      'overlap']] = 0, 0, 0, 0
        rule_df = rule_df[rule_df['overlap'] > 0]

        return rule_df[['Код ММ_М1', 'Код ММ_М2', 'Мин прибытие к М2', 'Макс прибытие к М2',
                         'Начало ГПД (в минутах)_М2', 'Окончание ГПД (в минутах)_М2',
                         'ИД категории ММ_М1', 'ИД категории ММ_М2', 'ret_time', 'overlap',
                         'o_tw', 'c_tw', 'wait_time', 'Время обслуживания_М1', 'Время обслуживания_М2', 'Срок_М2',
                         'Темп, заказ, пм_М2', 'Не темп, заказ, пм_М2', 'Весь заказ, пм_М1', 'Весь заказ, пм_М2', 'Можно бить_М2',
                         'ret_distance', 'Перегруз (в минутах)_М2', 'Подъезд (в минутах)_М2', 'Отъезд (в минутах)_М2',
                         'Оставить прицеп (в минутах)_М2', 'Забрать прицеп (в минутах)_М2'
                         ]]

    @property
    def market_parking_market_compatibility(self):
        """
        Таблица с предрасчитанными данными по использованию стоянки для перегруза или оставления прицепа
        :return:
        """
        # Для каждой пары market-market необходимо выбрать ближайшую стоянку (M1-P-M2 расстояние должно быть минимальным)
        # Предполагаем, что в колонке to_code таблицы parking_market стоит магазин

        logger = logging.getLogger(__name__ + ".MPMC")

        logger.info("Creating local data")
        loc_mm = self.market_market_dist.copy()
        logger.info('Number of loc_mm records: {}'.format(len(loc_mm)))
        loc_pm = self.parking_market_dist.copy()
        logger.info('Number of loc_pm records: {}'.format(len(loc_pm)))
        loc_o = self.orders_on_plandate.copy()

        # loc_mm.rename(columns=dict(zip(loc_mm.columns, [x + '_М1М2' for x in loc_mm.columns])), inplace=True)

        logger.info('Merging MMC with PM_dist on first M code')
        rule_df = pd.merge(loc_mm,
                           loc_pm.rename(columns=dict(zip(loc_pm.columns, [x + '_ПМ1' for x in loc_pm.columns]))),
                           left_on='from_code', right_on='to_code_ПМ1')
        logger.info('Number of records after merge: {}'.format(len(rule_df)))

        logger.info('Merging MMCPM with PM_dist on second M code')
        rule_df = pd.merge(rule_df,
                           loc_pm.rename(columns=dict(zip(loc_pm.columns, [x + '_ПМ2' for x in loc_pm.columns]))),
                           left_on='to_code', right_on='to_code_ПМ2')
        rule_df = rule_df[rule_df['from_code_ПМ1'] == rule_df['from_code_ПМ2']]
        logger.info('Number of records after merge: {}'.format(len(rule_df)))

        # ToDo Определить нормативы по времени работы с прицепом и добавить в качестве дополнительного параметра
        rule_df['ret_time_М1ПМ2'] = rule_df['ret_time_ПМ1'] + rule_df['ret_time_ПМ2']
        rule_df['ret_distance_М1ПМ2'] = rule_df['ret_distance_ПМ1'] + rule_df['ret_distance_ПМ2']
        # Группируем для поиска минимального ret_time_М1ПМ2

        rule_df = rule_df \
            .sort_values(by=['ret_time_М1ПМ2']) \
            .drop_duplicates(['from_code', 'to_code'], keep='first')

        RTO1_limit = 60 * 4
        rule_df.loc[rule_df['RTO1_duration_ПМ1'] >= RTO1_limit, 'RTO1_duration_М1ПМ2'] = \
            rule_df['RTO1_duration_ПМ1'] + rule_df['RTO2_duration_ПМ2']

        rule_df.loc[rule_df['RTO1_duration_ПМ1'] < RTO1_limit, 'RTO1_duration_М1ПМ2'] = \
            rule_df['RTO1_duration_ПМ1'] + rule_df['RTO1_duration_ПМ2']

        rule_df['RTO2_duration_М1ПМ2'] = \
            rule_df['RTO2_duration_ПМ1'] + rule_df['RTO2_duration_ПМ2']

        rule_df = pd.merge(rule_df,
                           loc_o.rename(columns=dict(zip(loc_o.columns, [x + '_М2' for x in loc_o.columns]))),
                           left_on=['to_code'],
                           right_on=['Код ММ_М2'])

        rule_df = pd.merge(rule_df,
                           loc_o.rename(columns=dict(zip(loc_o.columns, [x + '_М1' for x in loc_o.columns]))),
                           left_on=['from_code'],
                           right_on=['Код ММ_М1'])

        # Время прибытия к магазину
        rule_df['Мин прибытие к М2'] = rule_df['Начало ГПД (в минутах)_М1'] + rule_df['Время обслуживания_М1'] + \
                                       rule_df['RTO1_duration_М1ПМ2'] + rule_df['Оставить прицеп (в минутах)_М2']

        rule_df['Макс прибытие к М2'] = rule_df['Окончание ГПД (в минутах)_М1'] + rule_df['Время обслуживания_М1'] + \
                                        rule_df['RTO1_duration_М1ПМ2'] + rule_df['Оставить прицеп (в минутах)_М2']

        # Начало границы прибытия ко второму магазину
        rule_df['o_tw'] = rule_df[['Мин прибытие к М2', 'Начало ГПД (в минутах)_М2']].max(axis=1)

        # Конец границы прибытия ко Дата и время готовности ТСвторому магазину
        rule_df['c_tw'] = rule_df[['Макс прибытие к М2', 'Окончание ГПД (в минутах)_М2']].min(axis=1)

        # Перекрытие окна ГПД М2
        rule_df['overlap'] = rule_df['c_tw'] - rule_df['o_tw']

        # Время ожидания
        rule_df['wait_time'] = rule_df['Начало ГПД (в минутах)_М2'] - rule_df['Мин прибытие к М2']

        # Обновление данных согласно соответствию времени прибытия в М2 и ГПД М2

        rule_df.loc[
            rule_df['Макс прибытие к М2'] > rule_df['Окончание ГПД (в минутах)_М2'], ['o_tw', 'c_tw', 'wait_time',
                                                                                      'overlap']] = 0, 0, 0, 0
        rule_df = rule_df[rule_df['overlap'] > 0]

        # print(rule_df.iloc[100])

        return rule_df[['Код ММ_М1', 'Код ММ_М2', 'Мин прибытие к М2', 'Макс прибытие к М2',
                        'Начало ГПД (в минутах)_М2', 'Окончание ГПД (в минутах)_М2',
                        'ИД категории ММ_М1', 'ИД категории ММ_М2', 'ret_time', 'overlap',
                        'o_tw', 'c_tw', 'wait_time', 'Время обслуживания_М1', 'Время обслуживания_М2', 'Срок_М2',
                        'Темп, заказ, пм_М2', 'Не темп, заказ, пм_М2', 'Весь заказ, пм_М1', 'Весь заказ, пм_М2',
                        'Можно бить_М2',
                        'ret_distance', 'Перегруз (в минутах)_М2', 'Подъезд (в минутах)_М2', 'Отъезд (в минутах)_М2',
                        'Оставить прицеп (в минутах)_М2', 'Забрать прицеп (в минутах)_М2']]

    @property
    def rule_dc_parking_market_compatibility(self):
        """
        Функция определяет подходящую тройку РЦ - Парковка - Магазин
        :return:
        """
        logger = logging.getLogger(__name__ + 'DCPMC')

        loc_PM = self.parking_market_dist
        logger.info("Number of records in local P-M dist: {}".format(len(loc_PM)))
        loc_PDC = self.parking_dc_dist

        loc_store = self.store[['Дата и время отгрузки до', 'Минимальное время отправления',
                                'Минимальное время отправления (мин)']].copy()
        loc_store.rename(columns=dict(zip(loc_store.columns, [x + '_РЦ' for x in loc_store.columns])), inplace=True)

        loc_order = self.order[['order_id', 'Код ММ', 'ИД категории ММ',
                                'Строго ГПД', 'Начало ГПД (в минутах)', 'Окончание ГПД (в минутах)',
                                'Можно бить', 'Время обслуживания', 'Срок',
                                'Темп, заказ, пм', 'Не темп, заказ, пм', 'Весь заказ, пм',
                                'Перегруз (в минутах)', 'Подъезд (в минутах)', 'Отъезд (в минутах)',
                                'Оставить прицеп (в минутах)', 'Забрать прицеп (в минутах)']].copy()

        loc_order.rename(columns=dict(zip(loc_order.columns, [x + '_М' for x in loc_order.columns])), inplace=True)

        rule_df = pd.merge(loc_PDC.rename(columns=dict(zip(loc_PDC.columns, [x + '_ПРЦ' for x in loc_PDC.columns]))),
                           loc_PM.rename(columns=dict(zip(loc_PM.columns, [x + '_ПМ' for x in loc_PM.columns]))),
                           left_on=['from_code_ПРЦ'],
                           right_on=['from_code_ПМ'])

        rule_df = self.df_crossjoin(rule_df, loc_store)

        rule_df = pd.merge(rule_df, loc_order,
                           left_on=['to_code_ПМ'],
                           right_on=['Код ММ_М'])

        logger.info("Number of records in DC-P-M table: {}".format(len(rule_df)))

        rule_df['ret_time'] = rule_df['ret_time_ПРЦ'] + \
                              rule_df['ret_time_ПМ'] + \
                              rule_df['Оставить прицеп (в минутах)_М'] + \
                              rule_df['Подъезд (в минутах)_М']
        rule_df['ret_distance'] = rule_df['ret_distance_ПРЦ'] + rule_df['ret_distance_ПМ']

        rule_df['rto1_duration'] = rule_df['RTO1_duration_ПРЦ'] + rule_df['RTO1_duration_ПМ'] + rule_df[
            'Оставить прицеп (в минутах)_М']
        rule_df.loc[rule_df['ret_time_ПРЦ'] > 4 * 60, 'rto1_duration'] = rule_df['RTO1_duration_ПРЦ'] + rule_df[
            'RTO2_duration_ПМ'] + rule_df['Оставить прицеп (в минутах)_М']
        rule_df['rto2_duration'] = rule_df['RTO2_duration_ПРЦ'] + rule_df['RTO2_duration_ПМ'] + rule_df[
            'Оставить прицеп (в минутах)_М']

        rule_df['Мин прибытие к М'] = rule_df['Минимальное время отправления (мин)_РЦ'] + rule_df['ret_time']

        rule_df['Мин время отправления_М'] = rule_df['Начало ГПД (в минутах)_М'] + rule_df['Время обслуживания_М']
        rule_df['Макс время отправления_РЦ'] = rule_df['Окончание ГПД (в минутах)_М'] - rule_df['ret_time']

        rule_df['Мин время отправления_РЦ'] = rule_df['Начало ГПД (в минутах)_М'] - rule_df['ret_time']

        rule_df['cat1'] = 0  # Категория заказа в точке 1 = 0 для РЦ
        rule_df['overlap'] = 0  # Доп поле для комбинирования с market_market_compatibility
        rule_df['wait_time'] = 0  # Доп поле для комбинирования с market_market_compatibility
        rule_df['Время обслуживания_РЦ'] = 0

        rule_df.rename(columns={'to_code_ПМ': 'to_code', 'to_code_ПРЦ': 'from_code'}, inplace=True)

        print(rule_df.iloc[0])

        return rule_df[['order_id_М', 'from_code', 'to_code', 'cat1', 'ИД категории ММ_М', 'Начало ГПД (в минутах)_М',
                        'Мин время отправления_М', 'Макс время отправления_РЦ', 'Мин время отправления_РЦ',
                        'Начало ГПД (в минутах)_М', 'Окончание ГПД (в минутах)_М', 'ret_time',
                        'overlap', 'wait_time', 'Время обслуживания_РЦ', 'Время обслуживания_М', 'Срок_М',
                        'Темп, заказ, пм_М', 'Не темп, заказ, пм_М', 'Весь заказ, пм_М', 'Можно бить_М', 'ret_distance',
                        'Перегруз (в минутах)_М', 'Подъезд (в минутах)_М', 'Отъезд (в минутах)_М',
                        'Оставить прицеп (в минутах)_М', 'Забрать прицеп (в минутах)_М']]

    def RTO_calculation(self, veh_arc_merged:pd.DataFrame, RTO_dict = None):
        """
        Функция рассчитывает соотвествие ТС и плеч, согласно правилам расчета РТО
        РТО включает МО и РО
        :param veh_MM_merged:
        :return:
        """
        print("Start RTO calculation")
        # Если на входе нет словаря с параметрами, то используем захардкоженные
        if RTO_dict is None:
            RTO_dict = {
                            'Первые N часа непрерывного управления ТС':4*60,
                            'Следующие N часа непрерывного управления ТС': 2*60,
                            'Рабочий отдых 1': 30,
                            'Рабочий отдых 2': 15,
                            'Рабочая смена одного водителя': 14*60,
                            'Время управления одного водителя': 10*60,
                            'Отдых межсменный 1ВЭ': 9*60,
                            'Рабочая смена 2х водителей': 16*60,
                            'Время управления 2х водителей': 16*60,
                            'Отдых межсменный 2ВЭ': 8*60,
                            'Технологические операции': 30
                        }

        total_shift_dur = 'Длительность смены на предыдущих плечах'
        total_driving_dur = 'Продолжительность управления на предыдущих плечах'
        drivers_num = 'Количество ВЭ'
        exceed_four_hours = 'Была ли арка больше 4 часов' # 0 or 1

        current_arc_time = 'ret_time'
        current_arc_time_ro = 'ro_ret_time'
        new_arc_time = 'RTO_ret_time'
        mo_number = 'mo_num'
        before_mo = 'before_mo'
        before_mo_wo_ro = 'before_mo_wo_ro'
        left_after_mo = 'left_after_mo'
        left_after_all_mo = 'left_after_all_mo'

        # Расчет длительности текущей арки с учетом РО
        veh_arc_merged.loc[veh_arc_merged[exceed_four_hours] == 1, current_arc_time_ro] = \
            self.add_RTO(veh_arc_merged[[current_arc_time]], current_arc_time, inplace=False)['RTO2_duration']
        veh_arc_merged.loc[veh_arc_merged[exceed_four_hours] == 0, current_arc_time_ro] = \
            self.add_RTO(veh_arc_merged[[current_arc_time]], current_arc_time, inplace=False)['RTO1_duration']

        #Сколько ехать до межсменного отдыха
        veh_arc_merged[veh_arc_merged[drivers_num] == 1, before_mo] = \
            np.max(np.min(RTO_dict['Время управления одного водителя'] - veh_arc_merged[total_driving_dur],
                   RTO_dict['Рабочая смена одного водителя'] - veh_arc_merged[total_shift_dur]), 0)

        veh_arc_merged[veh_arc_merged[drivers_num] == 2, before_mo] = \
            np.min(RTO_dict['Время управления 2х водителей'] - veh_arc_merged[total_driving_dur],
                   RTO_dict['Рабочая смена 2х водителей'] - veh_arc_merged[total_shift_dur])

        # Больше ли отрезок до МО 4-х часов
        veh_arc_merged[before_mo_wo_ro] = veh_arc_merged[before_mo] - RTO_dict['Первые N часа непрерывного управления ТС']

        # Сколько удается проехать до МО от чистого времени маршрута
        veh_arc_merged[veh_arc_merged[exceed_four_hours], before_mo_wo_ro] = \
            (veh_arc_merged[before_mo] - veh_arc_merged[before_mo] // RTO_dict["Следующие N часа непрерывного управления ТС"]) * RTO_dict['Рабочий отдых 2']

        veh_arc_merged[~veh_arc_merged[exceed_four_hours], before_mo_wo_ro] = \
            veh_arc_merged[before_mo] - \
            (veh_arc_merged[before_mo] // RTO_dict['Первые N часа непрерывного управления ТС']) * RTO_dict['Рабочий отдых 1'] - \
            (veh_arc_merged[before_mo_wo_ro] // RTO_dict['Следующие N часа непрерывного управления ТС']) * RTO_dict['Рабочий отдых 2']

        # Сколько останется проехать чистого времени после первого МО
        veh_arc_merged[left_after_mo] = veh_arc_merged[current_arc_time] - veh_arc_merged[before_mo_wo_ro]

        # Считаем кол-во межсменных отдыхов в зависимости от кол-ва водителей
        veh_arc_merged.loc[veh_arc_merged[drivers_num] == 1, mo_number] = \
            np.maximum(
                np.floor((veh_arc_merged[total_shift_dur] +
                        veh_arc_merged[current_arc_time_ro])%RTO_dict['Рабочая смена одного водителя']),
                np.floor((veh_arc_merged[total_driving_dur] +
                          veh_arc_merged[current_arc_time_ro]) % RTO_dict['Время управления одного водителя'])
            )

        veh_arc_merged.loc[veh_arc_merged[drivers_num] == 2, mo_number] = \
            np.maximum(
                np.floor((veh_arc_merged[total_shift_dur] +
                          veh_arc_merged[current_arc_time_ro]) % RTO_dict['Рабочая смена 2х водителей']),
                np.floor((veh_arc_merged[total_driving_dur] +
                          veh_arc_merged[current_arc_time_ro]) % RTO_dict['Время управления 2х водителей'])
            )

        # Сколько останется проехать после последнего МО
        veh_arc_merged[left_after_all_mo] = veh_arc_merged[left_after_mo] - np.maximum(veh_arc_merged[mo_number] - 1, 0)*RTO_dict['Время управления одного водителя']

        # Сколько займет маршрут с учетом МО, там где они есть
        veh_arc_merged.loc[(veh_arc_merged[mo_number] > 0)&(veh_arc_merged[drivers_num] == 1), new_arc_time] = \
            veh_arc_merged[before_mo] + \
            RTO_dict['Отдых межсменный 1ВЭ'] * veh_arc_merged[mo_number] + \
            np.maximum(veh_arc_merged[mo_number] - 1, 0) * (RTO_dict['Время управления одного водителя']*(1 + RTO_dict['Рабочий отдых 2']/RTO_dict['Следующие N часа непрерывного управления ТС'])) + \
            self.add_RTO(veh_arc_merged, left_after_all_mo, inplace=False)['RTO1_duration']

        veh_arc_merged.loc[(veh_arc_merged[mo_number] > 0)&(veh_arc_merged[drivers_num] == 2), new_arc_time] = \
            veh_arc_merged[before_mo] + \
            RTO_dict['Отдых межсменный 2ВЭ'] * veh_arc_merged[mo_number] + \
            np.maximum(veh_arc_merged[mo_number] - 1, 0) * (RTO_dict['Время управления 2х водителей']*(1 + RTO_dict['Рабочий отдых 2']/RTO_dict['Следующие N часа непрерывного управления ТС'])) + \
            self.add_RTO(veh_arc_merged, left_after_all_mo, inplace=False)['RTO1_duration']


        # Добавляем ограничение
        veh_arc_merged = veh_arc_merged[veh_arc_merged['Окончание окна ГПД (мин)_М2'] - veh_arc_merged['Начало окна ГПД (мин)_М1'] < veh_arc_merged[new_arc_time]]
        print("End RTO calculation")
        return veh_arc_merged




    def weight_limit_to_pm_limit(self, veh_order_df):
        current_car_weight = 'current_car_weight'
        current_trailer_weight = 'current_trailer_weight'
        car_weight_limit = 'Ограничение по весу ТС'
        trailer_weight_limit = 'Ограничение по весу прицепа'
        order_car_volume = 'Объем заказа тяг'
        order_trailer_volume = 'Объем заказа приц'
        split = 'Можно бить'
        new_order_car_weight = 'order_car'
        new_order_trailer_weight = 'order_trailer'
        max_trailer_order_vol = 'max_trailer_order_vol'
        max_car_order_vol = 'max_car_order_vol'

        temp_density = 'temp_density'
        netemp_density = 'netemp_density'

        veh_order_df[temp_density] = veh_order_df['Темп, заказ, кг']/veh_order_df['Темп, заказ, пм']
        veh_order_df[netemp_density] = veh_order_df['Не темп, заказ, кг']/veh_order_df['Не темп, заказ, пм']

        veh_order_df.loc[veh_order_df['Утепл тяг'] == 2, max_trailer_order_vol] = \
            np.floor((veh_order_df[trailer_weight_limit] - veh_order_df[current_trailer_weight])/veh_order_df[netemp_density])

        veh_order_df.loc[veh_order_df['Утепл тяг'] == 4, max_trailer_order_vol] = \
            np.floor((veh_order_df[trailer_weight_limit] - veh_order_df[current_trailer_weight])/veh_order_df[temp_density])

        veh_order_df.loc[veh_order_df['Утепл тяг'] == 2, max_car_order_vol] = \
            np.floor((veh_order_df[car_weight_limit] - veh_order_df[current_car_weight]) / veh_order_df[
                netemp_density])

        veh_order_df.loc[veh_order_df['Утепл тяг'] == 4, max_car_order_vol] = \
            np.floor((veh_order_df[car_weight_limit] - veh_order_df[current_car_weight]) / veh_order_df[
                temp_density])


        # Тут можно либо вернуть этот DF в основной расчет и там использовать поле макс загрузки, либо продолжить считать здесь


        return veh_order_df




    @property
    def closest_parkings(self):
        loc_pm = self.parking_market_dist.copy()
        loc_pm = loc_pm.sort_values(by=['ret_time']).drop_duplicates(['to_code'], keep='first')
        return loc_pm



    def calculate_plandate_order_volume(self):
        return self.orders_on_plandate[["Темп, заказ, пм", "Не темп, заказ, пм"]].values.sum()

    def calculate_store_windows_capacity_on_plandate(self):
        fixed_volume = self.store['Возможность РЦ (%)'].values.sum()
        dynamic_volume = self.plandate_order_volume - fixed_volume
        self.store['Возможность РЦ (п/м)'] = self.store.apply(lambda x: dynamic_volume * x['Возможность РЦ (%)']
        if x['Возможность РЦ (п/м)'] == 0
        else x['Возможность РЦ (п/м)'], axis=1)
