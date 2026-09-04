import re

import yaml


def test_project_feature_and_label_lists_match_categories():
    with open("configs/projects/bmra_student_project.yaml", encoding="utf-8") as file:
        project = yaml.safe_load(file)

    source_name = lambda name: re.sub(r"\.\d+$", "", name)
    expected_features = (
        project["Daily habits Groups"]
        + project["Observations"]
        + project["Diagnosis"]
        + project["Problem Area"]
    )
    expected_labels = project["Treatment"] + project["Exercises"] + project["Supplements"]

    assert [source_name(name) for name in project["features"]] == expected_features
    assert [source_name(name) for name in project["labels"]] == expected_labels
    assert not set(project["features"]) & set(project["labels"])
