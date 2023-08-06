# Prepilot

### Сбор данных
1.   Для исследования, на этапе препилота, необходимы данные, которые содержат **customer_rk**
2.   По customer_rk собираются данные для стратификации с учетом параметров стратификации
    (на этапе сбора данных используются, заданные в параметрах, периоды стратификации).
    Для сбора данных используется **stratification.checks_preprocessing.collect_guests_data()**
3.   По customer_rk собираются метрики с использованием **post_analysis.guests_metrics_collector.GuestsMetricsCollector**
    На данном этапе используются post_analysis.params.FinParams заданные в ../post_analysis/configs/post_analysis_config.yaml
4.   Данные фильтруются post_analysis.params.FraudFilterParams заданные в ../post_analysis/configs/post_analysis_config.yaml

### Препилот
1. Для собранных на предыдущем этапе данных:
    *  Для каждой исследуемой метрики
        * Для каждого исследуемого inject
            * Создается столбец метрика*inject
    * Для каждого исследуемого размера групп:
        * Создается split с использованием **stratification.split_builder.build_target_control_groups()**
            * Split повторяется заданное n раз
2. Создается grid экспериментов
    * Для каждой метрики
        * Для каждого размера групп
            * Для каждого inject
                * Заданное n раз
    Таким образом, для 1 метрики, 3 размеров групп, 10 injects, с n=5
    количество экспериментов  = 1 * 3 * 10 * 5
3. По каждому элементу grid рассчитывается ошибка первого и второго рода
    Для расчета используется post_analysis.stat_test.PeriodStatTest.calculate_period_effect()
    При расчете ошибки первого рода inject не используется
    Результат группируется по метрике, размеру группы и inject

### Как запускать
1. Пример запуска в ../notebooks/prepilot.ipynb
Для корректного эксперимента необходимо задать параметры стратификации и препилота.(пример в notebook)
Также необходимо задать FinParams, FraudFilterParams, PeriodStatTestParams.
Как правило эти параметры задаются из ../post_analysis/configs/post_analysis_config.yaml
Все пять наборов параметров ожидаются на вход класса prepilot.prepilot_experiment_builder.PrepilotExperimentBuilder.
Ошибка второго рода может быть выведена в двух вариантах: аналитической и полной.
