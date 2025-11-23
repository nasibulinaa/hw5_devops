import pandas as pd
import deepchecks.tabular
from deepchecks.tabular.suites import data_integrity
import evidently.report
import evidently.metric_preset

def deepcheck_analyze(df: pd.DataFrame):
    dataset = deepchecks.tabular.Dataset(df, label='target', cat_features=[])
    integrity_suite = deepchecks.tabular.suites.data_integrity()
    integrity_result = integrity_suite.run(dataset)
    integrity_result.save_as_html('reports/deepchecks_report.html')
    with open('reports/deepchecks_report.json', 'w', encoding='utf-8') as f:
        f.write(integrity_result.to_json())
    return integrity_result


def evidently_analyze(df: pd.DataFrame):
    """Анализ дрейфа данных с EvidentlyAI"""
    reference_data = df.sample(frac=0.7, random_state=42)
    current_data = df.drop(reference_data.index)
    report = evidently.report.Report(metrics=[evidently.metric_preset.DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html('reports/evidently_report.html')
    return report

